# utils/log.py
import os
import sys
import logging
import inspect
from typing import Optional
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MAIN_LOG_FOLDER: Optional[str] = None  # set by main script

# -------------------------------
# Formatter
# -------------------------------
FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
)

# -------------------------------
# Loggers dictionary
# -------------------------------
LOGGERS = {
    '__main__': {'handlers': ['console', 'file', 'stream'], 'propagate': True},
    'nrsp.algs.nx': {'handlers': ['console', 'file', 'stream'], 'propagate': True},
    'nrsp.algs.cu': {'handlers': ['console', 'file', 'stream'], 'propagate': True},
    'nrsp.datasets.dummy': {'handlers': ['console', 'file'], 'propagate': True},
    'nrsp.utils': {'handlers': ['console', 'file'], 'propagate': True},
    'nxkernel': {'handlers': ['console', 'file'], 'propagate': True},
    'nx_kernel': {'handlers': ['console', 'file', 'stream'], 'propagate': True},
}

# -------------------------------
# Internal state
# -------------------------------
_configured_loggers = set()
_std_redirected = False

# -------------------------------
# Utilities
# -------------------------------
def _ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)

def _create_file_handler(module_name: str, folder: str) -> logging.Handler:
    subfolder = os.path.join(folder, *module_name.split('.')[:-1])
    _ensure_folder(subfolder)
    file_base = module_name.split('.')[-1] if module_name != "__main__" else "main"
    filename = os.path.join(subfolder, f"{file_base}.log")
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(FORMATTER)
    return handler

def _create_console_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(FORMATTER)
    return handler

def _detect_caller_module() -> str:
    frame = inspect.stack()[2]
    module = inspect.getmodule(frame[0])
    if module is None or module.__name__ == "__main__":
        return "__main__"
    return module.__name__

# -------------------------------
# Stream redirection
# -------------------------------
class StreamToLogger:
    """
    File-like stream object that redirects writes both to a logger and the original stream.
    """
    def __init__(self, logger, level, orig_stream):
        self.logger = logger
        self.level = level
        self.orig_stream = orig_stream  # keep original sys.stdout / sys.stderr

    def write(self, buf):
        # Always forward to console
        self.orig_stream.write(buf)
        self.orig_stream.flush()

        # Also log line-by-line (avoid blank lines)
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        self.orig_stream.flush()

def _redirect_std_streams(base_logger: logging.Logger):
    """Redirect sys.stdout and sys.stderr into logging, while keeping console output."""
    global _std_redirected
    if _std_redirected:
        return

    sys.stdout = StreamToLogger(base_logger, logging.INFO, sys.__stdout__)
    sys.stderr = StreamToLogger(base_logger, logging.ERROR, sys.__stderr__)
    _std_redirected = True

# -------------------------------
# Main function
# -------------------------------
def get_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """
    Get a logger. Automatically detects calling module.
    The main script must call get_logger(log_dir="...") first.
    """
    global MAIN_LOG_FOLDER

    module_name = _detect_caller_module()

    # Set MAIN_LOG_FOLDER if called by main script
    if module_name == "__main__" and log_dir is not None:
        _ensure_folder(log_dir)
        MAIN_LOG_FOLDER = log_dir

    if MAIN_LOG_FOLDER is None:
        raise ValueError(
            "Main log folder not set. The main script must call get_logger(log_dir='...') first."
        )

    logger = logging.getLogger(module_name)

    if module_name in _configured_loggers:
        return logger

    # --- Find the longest matching LOGGERS prefix ---
    matched_prefix = None
    longest_len = -1
    for key in LOGGERS:
        if module_name == key or module_name.startswith(key + "."):
            if len(key) > longest_len:
                matched_prefix = key
                longest_len = len(key)
    if matched_prefix is None:
        matched_prefix = module_name
    config = LOGGERS.get(matched_prefix, {'handlers': ['console'], 'propagate': False})

    # --- Attach handlers ---
    for h in config['handlers']:
        if h == 'console':
            logger.addHandler(_create_console_handler())
        elif h == 'file':
            logger.addHandler(_create_file_handler(module_name, MAIN_LOG_FOLDER))
        elif h == 'stream':
            # Redirect stdout/stderr into the main logger
            _redirect_std_streams(logger)

    logger.setLevel(logging.DEBUG)
    logger.propagate = config.get('propagate', False)
    _configured_loggers.add(module_name)

    return logger

# -------------------------------
# Silence noisy libraries globally
# -------------------------------
for lib in ["paramiko", "asyncssh", "matplotlib"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

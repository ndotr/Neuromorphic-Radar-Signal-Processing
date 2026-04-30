# radial_utils_simple.py
# Minimal helpers for RADIal (DBReader + labels_CVPR.csv)
# Label file is hardcoded relative to THIS file: radial/labels_CVPR.csv

import numpy as np
from pathlib import Path
from nrsp.datasets.radial_sdk.DBReader import SyncReader
import nrsp.datasets.radial as radial

# Radar parameter
NUM_SAMPLES = 512
NUM_RX_PER_CHIP = 4
NUM_CHIRPS = 256
NUM_RX_ANT = 16
NUM_TX_ANT = 12
NUM_DOPPLER = 16
NUM_CHIRPS_PER_LOOP = 16

RANGE_RESOLUTION = 0.2            # meters per range bin (placeholder)
VELOCITY_RESOLUTION = 0.1                 # max Doppler velocity (placeholder)

# -----------------------------
# Hardcoded label path (relative to this utils file)
# -----------------------------
_THIS_DIR = Path(__file__).resolve().parent
LABEL_FILE = (_THIS_DIR / "radial_sdk" / "labels_CVPR.csv").resolve()

# Global label cache (simple)
_LABELS_CACHE = {}  # resolved_path -> list[dict]


# -----------------------------
# Reader
# -----------------------------
def init_reader(data_folder: str, tolerance: int = 40000):
    return SyncReader(data_folder, tolerance=tolerance)


# -----------------------------
# Small parsing helpers
# -----------------------------
def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


# -----------------------------
# Labels
# -----------------------------
def load_cvpr_labels(path: Path = LABEL_FILE):
    """
    Very simple, robust parser for labels_CVPR.csv.

    - Detects delimiter: tab > comma > whitespace
    - Returns list[dict]
    - Caches results in-memory
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {p}")

    spath = str(p)
    if spath in _LABELS_CACHE:
        return _LABELS_CACHE[spath]

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        _LABELS_CACHE[spath] = []
        return []

    first = lines[0]
    if "\t" in first:
        delim = "\t"
    elif "," in first:
        delim = ","
    else:
        delim = None  # whitespace split

    header = None
    rows = []

    for line in lines:
        parts = line.split(delim) if delim is not None else line.split()

        # Header detect (first token not numeric => header)
        if header is None:
            try:
                float(parts[0])
                # no header row; use default header (matches common CVPR labels)
                header = [
                    "numSample","x1_pix","y1_pix","x2_pix","y2_pix",
                    "laser_X_m","laser_Y_m","radar_X_m","radar_Y_m",
                    "radar_R_m","radar_A_deg","radar_D_mps","radar_P_db",
                    "dataset","index","Annotation","Difficult"
                ]
                # fallthrough: treat current line as data
            except ValueError:
                header = parts
                continue

        if len(parts) < len(header):
            # keep it simple: skip malformed rows
            continue

        rows.append({k: v for k, v in zip(header, parts[:len(header)])})

    _LABELS_CACHE[spath] = rows
    return rows


def labels_for_frame(
    record_name: str,
    frame_idx: int,
    *,
    frame_key: str = "index",     # or "numSample"
    dataset_key: str = "dataset",
):
    """
    Filter labels by dataset name + frame key.
    Always uses LABEL_FILE.

    Returns list[dict] per object:
      R, D, A, P, ann, diff
    """
    out = []
    labels_all = load_cvpr_labels(LABEL_FILE)

    for r in labels_all:
        if r.get(dataset_key, "") != record_name:
            continue

        key_val = to_int(r.get(frame_key, None), default=None)
        if key_val is None or key_val != frame_idx:
            continue

        out.append({
            "R":   to_float(r.get("radar_R_m", np.nan)),
            "D":   to_float(r.get("radar_D_mps", np.nan)),
            "A":   to_float(r.get("radar_A_deg", np.nan)),
            "P":   to_float(r.get("radar_P_db", np.nan)),
            "ann": r.get("Annotation", ""),
            "diff": r.get("Difficult", ""),
        })

    return out


# -----------------------------
# Radar frame build
# -----------------------------
def build_radar_frame(
    adc0, adc1, adc2, adc3,
    *,
    numSamplePerChirp: int,
    numRxPerChip: int,
    numChirps: int,
):
    """
    Returns complex64 cube: (samples, chirps, rx_total)
    """
    adc0 = np.asarray(adc0)
    adc1 = np.asarray(adc1)
    adc2 = np.asarray(adc2)
    adc3 = np.asarray(adc3)

    frame0 = np.reshape(adc0[0::2] + 1j * adc0[1::2],
                        (numSamplePerChirp, numRxPerChip, numChirps),
                        order="F").transpose((0, 2, 1))
    frame1 = np.reshape(adc1[0::2] + 1j * adc1[1::2],
                        (numSamplePerChirp, numRxPerChip, numChirps),
                        order="F").transpose((0, 2, 1))
    frame2 = np.reshape(adc2[0::2] + 1j * adc2[1::2],
                        (numSamplePerChirp, numRxPerChip, numChirps),
                        order="F").transpose((0, 2, 1))
    frame3 = np.reshape(adc3[0::2] + 1j * adc3[1::2],
                        (numSamplePerChirp, numRxPerChip, numChirps),
                        order="F").transpose((0, 2, 1))

    out = np.concatenate([frame3, frame0, frame1, frame2], axis=2)
    return out.astype(np.complex64, copy=False)


# -----------------------------
# Windowing + RD
# -----------------------------
def window_2d(num_samples: int, num_chirps: int, apply_window: bool = True):
    if not apply_window:
        return 1.0

    # Hamming-style (matches your snippet)
    w_r = (0.54 - 0.46 * np.cos((2 * np.pi * np.arange(num_samples)) / (num_samples - 1)))
    w_d = (0.54 - 0.46 * np.cos((2 * np.pi * np.arange(num_chirps)) / (num_chirps - 1)))
    return (w_r[:, None] * w_d[None, :])[:, :, None].astype(np.float32)


def rd_map(raw_frame: np.ndarray, apply_window: bool = True, reduce_tx: bool = True):
    """
    raw_frame: (samples, chirps, rx)
    returns: float32 map
    """
    raw_frame = np.asarray(raw_frame)
    num_samples, num_chirps_total = raw_frame.shape

    win = window_2d(num_samples, num_chirps_total, apply_window)
    rd = np.fft.fft2(raw_frame * win[...,0], axes=(0, 1))
    if reduce_tx:
        rd = np.fft.fftshift(rd, axes=(1,))
    #rd = np.sum(np.abs(rd), axis=2)  # (samples, chirps)

    if reduce_tx:
        rd = np.sum(rd.reshape(NUM_SAMPLES, NUM_DOPPLER, NUM_CHIRPS_PER_LOOP), axis=1)
        #rd = rd.reshape(NUM_SAMPLES, NUM_DOPPLER, NUM_CHIRPS_PER_LOOP, -1)[:,0,:]  # keep tx dimension separate

    return rd.astype(np.float32, copy=False)


# -----------------------------
# One-call getters
# -----------------------------
def get_raw_frame(
    db,
    frame_idx: int,
    *,
    numSamplePerChirp: int,
    numRxPerChip: int,
    numChirps: int,
):
    sample = db.GetSensorData(frame_idx)
    return build_radar_frame(
        sample["radar_ch0"]["data"],
        sample["radar_ch1"]["data"],
        sample["radar_ch2"]["data"],
        sample["radar_ch3"]["data"],
        numSamplePerChirp=numSamplePerChirp,
        numRxPerChip=numRxPerChip,
        numChirps=numChirps,
    )


def get_radial_data(
    db,
    frame_idx: int,
    *,
    record_name: str,
    frame_key: str = "index",
    # keep these defaults in one place:
    numSamplePerChirp: int = NUM_SAMPLES,
    numRxPerChip: int = NUM_RX_PER_CHIP,
    numChirps: int = NUM_CHIRPS,
):
    """
    Returns: raw_frame, camera_frame, ann
    Always uses LABEL_FILE.
    """
    ann = labels_for_frame(record_name, frame_idx, frame_key=frame_key)

    sample = db.GetSensorData(frame_idx)

    raw_frame = build_radar_frame(
        sample["radar_ch0"]["data"],
        sample["radar_ch1"]["data"],
        sample["radar_ch2"]["data"],
        sample["radar_ch3"]["data"],
        numSamplePerChirp=numSamplePerChirp,
        numRxPerChip=numRxPerChip,
        numChirps=numChirps,
    )

    camera_frame = sample["camera"]["data"] if "camera" in sample else None
    return raw_frame, camera_frame, ann
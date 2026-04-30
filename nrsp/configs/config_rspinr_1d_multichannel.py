import numpy as np
from nrsp.configs.config_radar_basics import n_samples, n_ranges, n_channels

# === Algorithm ===
n_runs_for_output = n_samples + 2
n_runs_for_input = n_samples

# Tile shapes per group type
TILE_SHAPES = {
"input_eth": (1,),
"input_buf": (1,),
"spinr": (n_ranges,),
"output": ((n_ranges, n_channels,1),),
}

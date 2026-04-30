import numpy as np
from nrsp.configs.config_radar_basics import n_samples, n_ranges, n_velocities, n_chirps

# === Algorithm ===
n_runs_for_output = n_chirps * n_samples + 1 #+ n_layers + 5 # 64 + 5 + 5
n_runs_for_input = n_chirps * n_samples

# Tile shapes per group type
TILE_SHAPES = {
"input_eth": (1,),
"input_buf": (n_samples,),
"spinr": (n_ranges*n_velocities//118,),
"output_eth": (n_ranges*n_velocities,),
}
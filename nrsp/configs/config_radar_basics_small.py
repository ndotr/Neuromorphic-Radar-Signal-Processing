# === Radar Sensor ===
n_samples = 512
n_chirps = 64
n_channels = 1

# === Targets ===
rs =        [4,                      128,           500, 200, 300, 400]
amps =      [0.2,                    0.4,           0.6,   0.2,   0.3,   0.1]
thetas =    [2/n_channels,  4/n_channels,  8/n_channels, 100, 100, 100]
vs =        [0,                         0,            0, 100, 100, 100]
noise_std = None

# === Algorithm ===
n_velocities = n_chirps
n_ranges = n_samples // 2 + 1


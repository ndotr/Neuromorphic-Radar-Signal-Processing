# === Radar Sensor ===
n_samples = 512
n_chirps = 256
n_channels = 1

# === Targets ===
rs =        [4,                      120,           500, 180, 300, 400]
amps =      [0.2,                    0.4,           0.6,   0.2,   0.3,   0.1]
thetas = [0/n_channels, 0/n_channels, 0/n_channels, 0/n_channels, 100, 100]
vs =        [0, 12, 20, -8, -12, -24]
noise_std = 0.1

rs =        [4,                      8]
amps =      [1,                    1]
thetas = [0/n_channels, 0/n_channels ]
vs =        [4, 12]
noise_std = 0.1

# === Algorithm ===
n_velocities = n_chirps
n_ranges = n_samples//2


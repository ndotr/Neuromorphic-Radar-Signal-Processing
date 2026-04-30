# NRSP – Neuromorphic Radar Signal Processing

This repository contains research code for **neuromorphic radar signal processing (NRSP)**, including implementations on conventional GPU-based systems as well as deployment on Intel’s Loihi 2 neuromorphic hardware. 
The focus is on developing and evaluating radar processing methods that leverage neuromorphic principles such as sparsity, temporal dynamics, and locality, with the aim of improving efficiency in resource-constrained and real-time scenarios. 
The repository is structured per publication, with clearly separated scripts and results for each paper, while all implementations are built on a shared core Python module `nrsp`.

🚧 **Ongoing Work:**
- Real-time processing and vehicle demonstrator
- Cloud access for simulated FMCW radar data (In the meantime, contact me directly for data access)
---

# 📄 Publications

---

## Paper: Spiking Neural Resonator with Activity-Gated Sparsity

---

### Scope of This Paper

This paper introduces and evaluates:
- Spiking Neural Resonator with Activity-Gated Sparsity
- GPU Implementation
- Intel Loihi 2 Implementation

![SpiNR dynamics with AGS](results/ags/plots/spinr_dynamics_with_ags.pdf)

*Figure: Exemplary dynamics of the SpiNR model with AGS.*

#### Scripts (`scripts/ags/`)
Contains:
- `evaluations/` → experiment pipelines used to generate quantitative results
- `plots/` → scripts to reproduce figures in the paper
- `helpers/` → utility scripts specific to this paper

These scripts are the **only entry points** used to generate the published results.

Example:
```
python scripts/ags/evaluations/<evaluation_script>.py
```

#### Results (`results/ags/`)
Contains:
- Raw outputs from experiments
- Processed metrics
- Final plots used in the publication

Each result file corresponds directly to a script in `scripts/ags/`.

---

# 📁 Core Library Structure

```
nrsp/
├── algs/        # Algorithm implementations
├── rsp/         # Radar signal processing primitives
├── datasets/    # Data loading
├── metrics/     # Evaluation metrics
├── utils/       # Utilities
└── configs/     # Configurations
```

The `nrsp/` module is **shared across papers**, while `scripts/` and `results/` are paper-specific.

---

# ⚙️ Installation

```bash
git clone <repo-url>
cd Neuromorphic-Radar-Signal-Processing

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional GPU support:
```bash
pip install cupy-cuda12x
```

---

# 📌 Citation

- Introduction of the SpiNR model:
```
@article{reeb2025rangeSpiNR,
  author  = {Reeb, Nico and López-Randulfe, Javier and Dietrich, Robin and Knoll, Alois C.},
  title   = {Range and angle estimation with spiking neural resonators for {FMCW} radar},
  journal = {Neuromorphic Computing and Engineering},
  volume  = {5},
  number  = {2},
  pages   = {024009},
  year    = {2025},
  month   = may,
  doi     = {10.1088/2634-4386/adcf46},
  publisher = {IOP Publishing}
}
```

- Extension with AGS:

- Towards Real-Time Neuromorphic Radar Processing:

---

## 📬 Contact

For correspondence regarding this repository, especially the dataset:

**Nico Reeb**  
nico.reeb@tum.de
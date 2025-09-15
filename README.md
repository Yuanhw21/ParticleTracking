# Particle Tracking

A lightweight pipeline for particle detection and z-regression with:

- Synthetic training data with ground truth
- U-Net heatmap detector training
- Patch-based z regressor training
- Interactive inference browsing (Jupyter / VS Code Notebook)

Project layout:

```
models/                 # Trained/example models (.keras)
particles_data/         # Datasets (train/val/test .npz), can be generated
src/                    # Source code (data, models, training, inference utils)
TwoStep.ipynb           # Main notebook: data -> training -> inference
requirements.txt        # Python dependencies
```

## Environment & Installation

Recommended: Python 3.10–3.11 in a virtual environment.

Generic install (CPU or NVIDIA GPU environments):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Apple Silicon (M1/M2/M3) suggestion:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install tensorflow-macos tensorflow-metal  # use native TF for macOS
pip install -r requirements.txt --no-deps      # install the remaining deps only
```

> If you prefer the TensorFlow package pinned in `requirements.txt`, just follow the generic install.

## Quick Start (Notebook)

1. Open `TwoStep.ipynb`.
2. Run the first cell to import modules and set config.
3. Optional: run data generation and training (can be time-consuming).
4. Run inference cells to load models from `models/` and launch the interactive viewer.

Main adjustable parameters in the notebook:

- NMS threshold (`thr`), pixel tolerance (`px_tol`)
- Patch size for z regressor input (auto-resized to the training size)

## Command-line Usage

Build datasets (saved in `particles_data/`):

```python
from src.build_datasets import build_datasets_with_gt

build_datasets_with_gt(
	generator_path="src/Particle_Tracking_Training_Data.py",
	out_dir="./particles_data",
	n_train=2000, n_val=300, n_test=300,
	Nt=1, heat_sigma=2.0, seed=42,
)
```

Train the detector:

```python
from src.train import train_detector
import os

tr_npz = os.path.join("particles_data", "train_with_gt.npz")
va_npz = os.path.join("particles_data", "val_with_gt.npz")
det_model, hist = train_detector(tr_npz, va_npz, epochs=20, batch_size=8, lr=1e-3)
# Best model is saved to particles_data/detector.keras
```

Train the z regressor:

```python
from src.train import train_z_regressor
import os

tr_npz = os.path.join("particles_data", "train_with_gt.npz")
va_npz = os.path.join("particles_data", "val_with_gt.npz")
z_model, hist = train_z_regressor(tr_npz, va_npz, patch_size=21, epochs=20, batch_size=64, lr=1e-3)
# Best model is saved to particles_data/z_regressor.keras
```

## Dependencies

Key third-party packages:

- TensorFlow 2.x (or `tensorflow-macos` + `tensorflow-metal` on Apple Silicon)
- numpy, scipy
- matplotlib, ipywidgets (visualization and interactive UI)
- tqdm (progress bars for data generation)

## Data & Models

`*_with_gt.npz` files created by `src/build_datasets.py` include:

- `imgs`: (N,H,W,1) float images
- `heatmaps`: (N,H,W,1) label heatmaps
- `gt_xy_list`: per-image GT points (variable length, object array)
- `gt_z_list`: per-image GT z values (variable length, object array)

Training scripts save best models as:

- `particles_data/detector.keras`
- `particles_data/z_regressor.keras`

You may copy/move them into `models/` for the notebook to load directly.

## Troubleshooting

- No module named 'tensorflow': ensure the venv is activated and TF is installed. On Apple Silicon, prefer `tensorflow-macos` + `tensorflow-metal`.
- ipywidgets not showing in Jupyter/VS Code: ensure `ipywidgets>=8` is installed; in VS Code, enable the Jupyter and Widgets extensions and select the correct Python kernel.
- High memory usage: reduce synthetic dataset size or lower the batch size.

---

MIT License
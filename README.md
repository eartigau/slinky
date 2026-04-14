# Slinky: Telluric Correction and Spectral Processing Toolkit

Slinky is a Python toolkit for telluric correction and spectral processing, designed for use with high-resolution spectrographs (e.g., NIRPS, SPIRou). It shares configuration and YAML parameter files with the [telluric_fit](https://github.com/yourorg/telluric_fit) repository for seamless integration.

## Features
- Batch telluric correction
- Residual PCA and pixel-level modeling
- Flexible YAML-driven configuration (see `yamls/`)
- Utilities for plotting, database management, and more

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourorg/slinky.git
cd slinky
```

### 2. Set up the Python environment (recommended: conda)

Create and activate a new environment (Python 3.12 recommended):
```bash
conda create -n slinky python=3.12
conda activate slinky
```

### 3. Install dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

> **Note:** If you use the [telluric_fit](https://github.com/yourorg/telluric_fit) repo, you can share YAML config files between both projects. Place shared YAMLs in the `yamls/` folder.

## Usage

- Main batch processing: `python batch_slinky.py`
- Residual PCA: `python residual_pca.py`
- Plotting tools: `python plot_slinky.py`
- See individual scripts for more details and options.

## Configuration

- YAML parameter files are in `yamls/` (copied/shared from `telluric_fit`)
- Example: `yamls/params_tellu05.yaml`, `yamls/params_tellu15B.yaml`
- For instrument-specific settings, see the `telluric_config.yaml` in the `telluric_fit` repo

## Directory Structure

- `yamls/` — YAML parameter/config files (shared with telluric_fit)
- `fiber_diff/` — Example FITS files for testing
- `obsolete/` — Legacy scripts (not maintained)
- `plots/` — Output plots
- Main scripts: `batch_slinky.py`, `residual_pca.py`, `plot_slinky.py`, etc.

## License
MIT License

## Acknowledgments
- This project shares configuration and methodology with [telluric_fit](https://github.com/yourorg/telluric_fit)
- Developed by Étienne Artigau and collaborators

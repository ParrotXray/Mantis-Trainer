# NetGuardia-Trainer

A machine learning training pipeline for network intrusion detection models. Trains Deep Autoencoder and LightGBM classifier on multiple datasets and exports to ONNX format for deployment with [NetGuardia](https://github.com/ParrotXray/NetGuardia).

## Overview

NetGuardia-Trainer provides a complete pipeline for training network anomaly detection and classification models:

1. **Data Preprocessing** - Loads multiple datasets (CIC-IDS-2017, CIC-IDS-2018, CIC-UNSW-NB15), maps to a unified 26-feature schema, and normalizes labels
2. **Deep Autoencoder** - Trains an autoencoder on 15 common features for unsupervised anomaly detection
3. **LightGBM Classifier** - Trains a gradient boosted tree classifier on all 36 features for multi-class attack classification
4. **ONNX Export** - Exports trained models to ONNX format for cross-platform inference

## Supported Datasets

Datasets are **auto-downloaded** from Kaggle via [kagglehub](https://github.com/Kaggle/kagglehub). No manual download required.

| Dataset | Kaggle ID | Use Features | Use File |
|---------|-----------|----------|----------|
| CIC-IDS-2017 | `chethuhn/network-intrusion-dataset` | 26 | `all` |
| CIC-IDS-2018 | `dhoogla/csecicids2018` | 26 | `all` |
| CIC-UNSW-NB15 | `yasserhessein/cic-unsw-nb15-augmented-dataset` | 26 | `CICFlowMeter.csv` |

When merged, the unified schema has **26 features** total. Missing features are NaN (handled natively by LightGBM).

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support (recommended)
- 50 GB or more free disk space for datasets and models 
- 40 GB RAM or more (for training on full datasets)

### Dependencies

- PyTorch and PyTorch Lightning for deep learning
- LightGBM for gradient boosted classification
- scikit-learn for preprocessing
- imbalanced-learn for SMOTE oversampling
- ONNX, onnxruntime, onnxmltools for model export
- kagglehub for dataset auto-download
- pandas and numpy for data processing

See `requirements.txt` for the complete list.

## Installation

```bash
git clone https://github.com/ParrotXray/NetGuardia-Trainer.git
cd NetGuardia-Trainer
pip3 install -r requirements.txt
```

## Usage

```bash
cd src
chmod +x main.py
```

### Run Complete Pipeline

```bash
# Single dataset (auto-download)
./main.py -s cic2018 -a

# Merge two datasets
./main.py -s cic2018,unsw -a

# All three datasets
./main.py -s cic2017,cic2018,unsw -a
```

### Run Individual Stages

```bash
# Data preprocessing only
./main.py -s cic2018 -dp

# Train Deep Autoencoder only
./main.py -da

# Train LightGBM Classifier only
./main.py -cl

# Export models to ONNX only
./main.py -ep
```

### Use Local Dataset Paths

If you already have datasets downloaded locally, override auto-download with `--path`:

```bash
./main.py -s cic2018,unsw --path "/data/cic2018,/data/unsw" -a
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `-s, --set` | Dataset name(s), comma-separated (`cic2017`, `cic2018`, `unsw`) |
| `--path` | Optional: local path(s) to dataset directories (overrides auto-download) |
| `-a, --all` | Run complete training pipeline |
| `-dp, --datapreprocess` | Run data preprocessing |
| `-da, --deepautoencoder` | Train Deep Autoencoder model |
| `-cl, --classifier` | Train LightGBM classifier |
| `-ep, --export` | Export models to ONNX format |

## Docker

### Pull Pre-built Image

```bash
docker pull ghcr.io/parrotxray/netguardia-trainer:master
```

### Run with Docker

```bash
# Full pipeline with auto-download
docker run --gpus all \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/metadata:/app/src/metadata \
  -v ./src/plots:/app/src/plots \
  -v ./src/exports:/app/src/exports \
  -v ./src/logs:/app/src/logs \
  -e DATASET="cic2018,cic2017,cicdos2017,cic2019,unsw" \
  -e ALL=true \
  ghcr.io/parrotxray/netguardia-trainer:master

# With local dataset (mount data directory)
docker run --gpus all \
  -v /path/to/datasets:/data \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/metadata:/app/src/metadata \
  -v ./src/plots:/app/src/plots \
  -v ./src/exports:/app/src/exports \
  -v ./src/logs:/app/src/logs \
  -e DATASET="cic2018,cic2017,cicdos2017,cic2019,unsw" \
  -e DATAPATH="/data/cic2018,/data/cic2017,/data/cicdos2017,/data/cic2019,/data/unsw" \
  -e ALL=true \
  ghcr.io/parrotxray/netguardia-trainer:master
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DATASET` | Dataset name(s), comma-separated |
| `DATAPATH` | Optional: local path(s) to datasets (overrides auto-download) |
| `ALL` | Run complete pipeline (`true`/`false`) |
| `DATAPREPROCESS` | Run preprocessing (`true`/`false`) |
| `DEEPAUTOENCODER` | Train autoencoder (`true`/`false`) |
| `CLASSIFIER` | Train classifier (`true`/`false`) |
| `EXPORT` | Export to ONNX (`true`/`false`) |

### Build Docker Image Locally

```bash
docker build -t netguardia-trainer .
```

## Output Directories

| Directory | Contents |
|-----------|----------|
| `outputs/` | Processed CSV files (benign/attack splits) |
| `artifacts/` | Trained model files (PyTorch, LightGBM) |
| `metadata/` | Model configurations, label encoders, stats |
| `plots/` | Training visualizations and analysis plots |
| `exports/` | ONNX models and inference config JSON |
| `log/` | Process output records |

## Acknowledgments

- Canadian Institute for Cybersecurity for the CIC-IDS datasets
- UNSW Canberra for the UNSW-NB15 dataset
- PyTorch and PyTorch Lightning teams
- LightGBM team

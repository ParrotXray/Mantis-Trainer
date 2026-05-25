# Mantis-Trainer

A machine learning training pipeline for the [Mantis](https://github.com/ParrotXray/Mantis) network intrusion detection system. Trains an LSTM Autoencoder on real-world BENIGN traffic and exports to ONNX format for deployment.

## Overview

Mantis-Trainer implements a two-stage pipeline:

1. **Data Preprocessing**: Loads multiple datasets, maps to a unified 39-feature schema, and normalizes labels
2. **LSTM Autoencoder**: Trains an unsupervised autoencoder on BENIGN-only traffic for anomaly detection
3. **ONNX Export**: Exports the trained model and inference configuration for deployment with Mantis

The autoencoder learns the normal traffic distribution from real laboratory traffic. During inference, flows with high reconstruction error are flagged as anomalies.

## Architecture

```
Input (39 features, window=10)
    → LSTM Encoder (2 layers, hidden=128)
    → Bottleneck (encoding_dim=32)
    → LSTM Decoder (2 layers, hidden=128)
    → Reconstruction (39 features)

Anomaly Score = MSE(input, reconstruction)
```

Training uses a latent norm penalty (`||z||²`) to compress BENIGN latent vectors toward the origin, improving separation from unseen attack flows.

## Inference work flow

<img width="1473" height="513" alt="image" src="https://github.com/user-attachments/assets/02f244ff-ea1b-4a84-a46b-693261418578" />

## Supported Datasets

Datasets are **auto-downloaded** from Kaggle via [kagglehub](https://github.com/Kaggle/kagglehub).

### Training (BENIGN only)

| Dataset | Kaggle ID | Description |
|---------|-----------|-------------|
| LAB-301 | `ruiluncai/lab301-timestamp-benign-dataset` | Real lab traffic (YouTube, Speedtest, browsing) with timestamps |

### Testing (Attack evaluation)

| Dataset | Kaggle ID | Description |
|---------|-----------|-------------|
| CIC-IDS-2017 + FLNET2023 | `ruiluncai/attack-test-dataset` | CIC-IDS-2017 standard benchmark + FLNET2023 modern attacks (DoS, DDoS, Infiltration, SQL Injection, XSS, Command Injection) |

### Unified Feature Schema (39 features)

| Category | Features |
|----------|----------|
| Flow | `flow_duration`, `flow_bytes_per_sec`, `flow_pkts_per_sec` |
| Packet counts | `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes` |
| Packet length | `fwd_pkt_len_mean/std`, `bwd_pkt_len_mean/std`, `pkt_len_mean/std` |
| IAT | `fwd_iat_mean`, `bwd_iat_mean`, `flow_iat_mean/std/max/min` |
| Window | `fwd_win_bytes`, `bwd_win_bytes` |
| Flags | `psh_flag_cnt`, `ack_flag_cnt`, `syn_flag_cnt`, `fin_flag_cnt`, `rst_flag_cnt` |
| Misc | `dst_port`, `protocol`, `fwd_seg_size_min`, `fwd_act_data_pkts`, `down_up_ratio` |
| Active/Idle | `active_mean/std/max/min`, `idle_mean/std/max/min` |

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support (recommended)
- 32 GB RAM or more
- 20 GB free disk space

### Dependencies

- PyTorch and PyTorch Lightning for deep learning
- scikit-learn for preprocessing
- ONNX, onnxruntime for model export
- kagglehub for dataset auto-download
- pandas, numpy for data processing

See `requirements.txt` for the complete list.

## Installation

```bash
git clone https://github.com/ParrotXray/Mantis-Trainer.git
cd Mantis-Trainer
pip3 install -r requirements.txt
```

## Usage

```bash
cd src
chmod +x main.py
```

### Run Complete Pipeline

```bash
./main.py -s lab301,test -a
```

### Run Individual Stages

```bash
# Data preprocessing only
./main.py -s lab301,test -dp

# Train LSTM Autoencoder only
./main.py -da

# Export model to ONNX only
./main.py -ep
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `-s, --set` | Dataset name(s), comma-separated (`lab301`, `test`) |
| `--path` | Optional: local path(s) to dataset directories (overrides auto-download) |
| `-a, --all` | Run complete pipeline |
| `-dp, --datapreprocess` | Run data preprocessing |
| `-da, --deepautoencoder` | Train LSTM Autoencoder |
| `-ep, --export` | Export model to ONNX |

## Threshold Selection

After training, the pipeline prints a threshold analysis table derived from the **BENIGN-only validation set** (no attack labels involved):

```
Threshold Analysis [Val Set — BENIGN only]:
Name           Threshold       Val FPR
------------------------------------------
90             0.012345        10.00%
91             0.015678         9.00%
...
Q3+1.5IQR      0.011234        10.02%
Q3+3.0IQR      0.018901         5.62%
```

All threshold candidates are exported to `artifacts/deep_ae_config.pkl` under the key `ae_thresholds`. Select the threshold that meets your FPR tolerance and configure it in Mantis's `inference_config.json`.

## Docker

### Pull Pre-built Image

```bash
docker pull ghcr.io/parrotxray/mantis-trainer:master
```

### Run with Docker

```bash
docker run --gpus all \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/plots:/app/src/plots \
  -v ./src/exports:/app/src/exports \
  -v ./src/logs:/app/src/logs \
  -e DATASET="lab301,test" \
  -e ALL=true \
  ghcr.io/parrotxray/mantis-trainer:master
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DATASET` | Dataset name(s), comma-separated |
| `DATAPATH` | Optional: local path(s) to datasets (overrides auto-download) |
| `ALL` | Run complete pipeline (`true`/`false`) |
| `DATAPREPROCESS` | Run preprocessing (`true`/`false`) |
| `DEEPAUTOENCODER` | Train autoencoder (`true`/`false`) |
| `EXPORT` | Export to ONNX (`true`/`false`) |

### Build Docker Image Locally

```bash
docker build -t mantis-trainer .
```

## Output Directories

| Directory | Contents |
|-----------|----------|
| `outputs/` | Processed CSV files (benign/attack splits, AE scores) |
| `artifacts/` | Trained model files (PyTorch checkpoint, ONNX, config pkl) |
| `plots/` | Training visualizations (ROC, PR curve, t-SNE, score distribution) |
| `exports/` | ONNX model and inference config JSON |
| `logs/` | Process output records |

## Limitations

- Attack evaluation uses CIC-IDS-2017 (2017) and FLNET2023 (2023) benchmark datasets generated in controlled environments. Some attack categories (Exploitation, Reconnaissance) exhibit flow-level statistical features similar to BENIGN traffic, resulting in lower detection rates. This is a known limitation of flow-based NIDS datasets [Dube 2024, Engelen et al. 2021].
- The autoencoder is trained on BENIGN-only traffic. Detection capability for novel attacks depends on how much their flow statistics deviate from normal behaviour.

## Acknowledgments

- Canadian Institute for Cybersecurity for the CIC-IDS datasets
- FLNET2023 dataset authors
- PyTorch and PyTorch Lightning teams

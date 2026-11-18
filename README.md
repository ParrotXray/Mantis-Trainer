# Mantis-Trainer

A machine learning training pipeline for the [Mantis](https://github.com/ParrotXray/Mantis) network intrusion detection system. Trains an LSTM Autoencoder on real-world BENIGN traffic and exports to ONNX format for deployment.

For step-by-step operations (CLI, API, Docker, parameters, workflows, troubleshooting), see the Operations Guide: [English](docs/GUIDE.en.md) · [繁體中文](docs/GUIDE.md).

## Overview

Mantis-Trainer implements a two-stage pipeline:

1. **Data Preprocessing**: Loads multiple datasets, maps to a unified 39-feature schema, and normalizes labels
2. **LSTM Autoencoder**: Trains an unsupervised autoencoder on BENIGN-only traffic for anomaly detection
3. **ONNX Export**: Exports the trained model and inference configuration for deployment with Mantis

The autoencoder learns the normal traffic distribution from real laboratory traffic. During inference, flows with high reconstruction error are flagged as anomalies.

## Architecture

```
Input (39 features, window=10)
    → LSTM Encoder (4 layers, hidden=128)
    → Bottleneck (encoding_dim=32)
    → LSTM Decoder (4 layers, hidden=128)
    → Reconstruction (39 features)

Anomaly Score = MSE(input, reconstruction)
```

Training uses a latent norm penalty (`||z||²`) to compress BENIGN latent vectors toward the origin, improving separation from unseen attack flows.

## Inference work flow

<img width="1473" height="513" alt="image" src="https://github.com/user-attachments/assets/02f244ff-ea1b-4a84-a46b-693261418578" />

## Supported Datasets

Datasets are local CSV files in one pool per node: `src/datasets/` (override with `DATASET_ROOT`, used by both the CLI and the API). Copy CSVs there or upload them with `POST /api/v1/datasets`; stage `dp` reads every CSV in the pool, subdirectories included.

### Training (BENIGN only)

| Dataset | Kaggle dataset (download the CSVs into `src/datasets/`) | Description |
|---------|-----------|-------------|
| LAB-301 | `ruiluncai/lab301-timestamp-benign-dataset` | Real lab traffic (YouTube, Speedtest, browsing) with timestamps |

### Testing (Attack evaluation)

| Dataset | Kaggle dataset (download the CSVs into `src/datasets/`) | Description |
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
- FastAPI and Uvicorn for the training API server
- pandas, numpy for data processing

See `requirements.txt` for the complete list.

## Installation

```bash
git clone https://github.com/ParrotXray/Mantis-Trainer.git
cd Mantis-Trainer
pip3 install -r requirements.txt
```

PyTorch comes in as a dependency of Lightning.

## Code Layout

| Path | Role |
|------|------|
| `src/pipeline/` | Shared core: stages, stage rules, `section.key=value` parser, CSV pool, runner |
| `src/main.py` | CLI: turns flags into a pipeline request and runs it in-process |
| `src/api/` | HTTP server: turns requests into the same pipeline request and runs it in a worker process |
| `src/components/`, `src/model/` | Training code and configs used by the pipeline |

The CLI and the API accept the same stages and the same extra parameters, and apply the same rules.

## Usage

```bash
cd src
chmod +x main.py
```

Put the CSV files in `src/datasets/` first.

### Run Complete Pipeline

```bash
./main.py -a
```

### Run Individual Stages

```bash
./main.py -dp --args "dp.exclude_verdicts=none"
./main.py -da --args "da.epochs=50 da.batch_size=4096"
./main.py -ft --args "run.model_dir=./old_model"
./main.py -sc --args "da.analysis_threshold=p99.5"
./main.py -fa --args "da.analysis_threshold=p95"
./main.py -ex
```

Flags only pick stages. Every parameter goes in `--args`, written the same way as the API's `args`.

### Command Line Arguments

| Argument | Stage | Description |
|----------|-------|-------------|
| `-a, --all` | `dp` `da` `ex` | Run complete pipeline |
| `-dp, --datapreprocess` | `dp` | Preprocess every CSV in `$DATASET_ROOT` |
| `-da, --deepautoencoder` | `da` | Train LSTM Autoencoder |
| `-ft, --finetune` | `ft` | Finetune the model in `model_dir` (default `./artifacts`, backed up to `artifacts/pretrained_<time>/` first) on the loaded data; cannot run with `-da` |
| `-sc, --score` | `sc` | Score preprocessed benign data with the model in `model_dir` and its saved thresholds, no training; cannot run with `-dp` |
| `-fa, --feature-analysis` | `fa` | Per-feature error analysis of the model in `artifacts/` |
| `-ex, --export` | `ex` | Export model to ONNX |
| `--args` | | Extra parameters, see below |

### Extra Parameters

One line of `section.key=value` pairs separated by spaces (Hydra-style), shared by the CLI (`--args`) and the API (`args`):

```
da.epochs=50 da.batch_size=4096 da.clip_min=-4 dp.exclude_verdicts=none run.resume=autoencoder_temp-v14.ckpt
```

- Every key needs its section prefix: `dp.` (`PreprocessConfig`), `da.` (`DeepAutoencoderConfig`), `ex.` (`ExportConfig`) or `run.` (`resume`, `model_dir`). A key without a prefix is rejected with the prefix to use.
- Values are cast to the field type. Tuple fields take comma-separated values; `none` clears a tuple or an optional field. Quote values that contain spaces.
- `run.resume=<checkpoint>` resumes `da` training. `run.model_dir=<dir>` picks the model for `ft` / `sc` (default `./artifacts`). A bare name without `/` is looked up in `./artifacts` first, so `run.resume=autoencoder_temp-v14.ckpt` or `run.model_dir=pretrained_20261110_0412` is enough.
- Unknown keys, wrong types, duplicates and keys that do not apply to the selected stages are rejected before anything runs.

### Mantis-recorded data

Traffic CSVs recorded by Mantis carry an `ML Verdict` column next to `Label` (which is always `BENIGN`):
`normal`, `anomaly`, `rule_match`, `structural`, `confirmed_attack`, `suppressed` or `uninspected`.
Preprocessing logs the verdict distribution per dataset and drops the rows listed in `dp.exclude_verdicts=`.

| Verdict | Evidence beyond the model |
|---------|---------------------------|
| `rule_match` | Suricata rule match |
| `structural` | Aggregator FLOOD / SCAN / TARGETED verdict |
| `confirmed_attack` | Both |

The default drops `confirmed_attack` and `rule_match`. `structural` rows are kept: the aggregator heuristics
count per source across all destinations, so a busy client's ordinary egress (many HTTPS sites, DNS, push
services) trips them too; add `structural` to the list once those heuristics are trusted. Plain `anomaly` rows
are kept on purpose. The live threshold flags a fixed share of benign traffic by definition, so dropping every anomaly
would trim the benign tail on each retrain and narrow the model every cycle. `suppressed` rows were confirmed
as false positives by an operator and are benign.

`dp.exclude_src_ips=` / `dp.exclude_dst_ips=` (IPs or CIDR blocks, empty by default) drop rows by source or
destination IP, for traffic the model must not learn as normal (a compromised host, a pentest box). They are not for
hosts that alert too often: a host the model never saw scores higher once deployed, so use a Mantis suppression rule
for those. Prefer the source list: excluding by destination also removes flows from the middle of other hosts'
sequences. The log reports what each rule excluded.

Dropped rows keep their original position in the drain order (`flow_seq`), so windows over the remaining flows
follow the order Mantis saw live. Datasets without the column are not affected. When preprocessing data for
stage `sc`, use `dp.exclude_verdicts=none` so the comparison with the live alert rate covers the same flows.

The verdict is also kept in the preprocessed data as `ml_verdict` (analysis only, never a model input), and
stage `sc` reports each verdict's window count, median score and share over the threshold. A window is labelled
with the verdict of its last flow, the one Mantis scored live. To check that a retrain learned the flows an
operator marked as false positives, score the same period with the model before (finetuning backs it up to
`artifacts/pretrained_<time>/`) and after: the `suppressed` share over the threshold should drop, and the
`normal` share should not rise. Flows of the same kind recorded before the suppression rule existed are still
labelled `anomaly`, so the `suppressed` row undercounts them. Data preprocessed before this change has no
`ml_verdict`; rerun `-dp`.

## API Server

Each node runs an independent FastAPI server. Scale out by starting one server per GPU host, each with its own `NODE_ID`; a scheduler can poll `/api/v1/node` and `/api/v1/metrics` and dispatch jobs to idle nodes. A node runs one job at a time and answers `409` when busy. Jobs run in a separate process so the API stays responsive.

```bash
cd src
API_TOKEN=$(openssl rand -hex 32) NODE_ID=node-a ./server.py
```

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `NODE_ID` | `<hostname>-<random>` | Node identity reported by every response |
| `WORKSPACE` | `.` | Directory holding `outputs/`, `artifacts/`, `plots/`, `exports/`, `logs/` |
| `DATASET_ROOT` | `<WORKSPACE>/datasets` | Directory for imported CSV datasets |
| `MAX_UPLOAD_MB` | `10240` | Per-file upload limit |
| `CPU_SAMPLE_INTERVAL` | `0.2` | CPU sampling window in seconds |
| `WS_MIN_INTERVAL` | `0.5` | Minimum WebSocket push interval in seconds |
| `WS_MAX_INTERVAL` | `60` | Maximum WebSocket push interval in seconds |
| `API_TOKEN` | required | At least 16 characters; the server refuses to start without it. Every endpoint except `/health` needs `Authorization: Bearer <token>` (or `?token=<token>` for EventSource, WebSocket and download links) |
| `CORS_ORIGINS` | none | Comma-separated browser origins allowed to call the API |

Interactive docs are served at `/docs`. Job records are kept in `logs/jobs/<id>.job.json` and survive a restart; jobs still running when the server stopped are marked `failed`.

### Endpoints (prefix `/api/v1`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/node` | Node identity and busy state |
| GET | `/metrics` | CPU, RAM and GPU usage |
| GET | `/metrics/cpu` \| `/metrics/memory` \| `/metrics/gpu` | Single resource usage |
| WS | `/metrics/ws?interval=1.0` | Push CPU, RAM and GPU usage every `interval` seconds (JSON per message) |
| WS | `/node/ws?interval=5.0` | Push a `NodeSnapshot` every `interval` seconds: `node` (as `/node`), `metrics` (as `/metrics`) and `jobs` (as `/jobs`, or `null` when unchanged since the previous message on this socket); the web UI uses it instead of polling |
| GET | `/datasets` | List imported CSV files |
| POST | `/datasets?replace=false` | Import CSV files (multipart field `files`, original file names kept; `replace=true` clears the pool first) |
| DELETE | `/datasets/{file}` | Remove one CSV file |
| DELETE | `/datasets` | Remove all CSV files |
| GET | `/jobs/params` | Default values of all `args` keys, grouped by `dp` / `da` / `ex` / `run` |
| POST | `/jobs/check` | Validate a job request without starting it; returns the stages, CSV files and model paths it would use |
| POST | `/jobs` | Start a job |
| GET | `/jobs` | List jobs |
| GET | `/jobs/{id}` | Job status, with `stage_progress` (per-stage status and times) and `current_stage` |
| DELETE | `/jobs/{id}` | Request cancellation (status becomes `cancelled` once the process exits) |
| GET | `/jobs/{id}/logs?lines=200` | Tail job log |
| GET | `/jobs/{id}/logs/stream` | Follow job log (Server-Sent Events) |
| GET | `/logs` | List log files |
| GET | `/logs/{name}?lines=200` | Tail a log file (e.g. `train.log`) |
| GET | `/files/{category}` | List files (`exports`, `artifacts`, `plots`, `outputs`, `logs`) |
| GET | `/files/{category}/download?path=<file>` | Download one file |
| GET | `/files/{category}/archive?path=<dir>` | Download a directory as zip (whole category when `path` is empty) |

The examples below leave the token out for brevity: add `-H "Authorization: Bearer $API_TOKEN"` to each `curl`, and `&token=$API_TOKEN` to the WebSocket URL.

### Import CSV

```bash
curl -F files=@lab301_benign.csv -F files=@attack_test.csv http://localhost:8000/api/v1/datasets
```

### Start a Job

`stages` takes any of `dp`, `da`, `ft`, `sc`, `fa`, `ex` or `all` (`dp` + `da` + `ex`); they always run in pipeline order. `ft` cannot run with `da`, and `sc` cannot run with `dp`. `dp` reads every CSV imported on the node.

`args` is the same `section.key=value` line the CLI takes with `--args` (see [Extra Parameters](#extra-parameters)). Paths in `run.model_dir` and `run.resume` must stay inside the node's workspace.

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"stages": ["all", "fa"], "args": "da.epochs=50 da.batch_size=4096 da.analysis_threshold=p95"}'
```

### Stream Metrics

```bash
websocat "ws://localhost:8000/api/v1/metrics/ws?interval=1"
```

### Download Results

```bash
curl -OJ http://localhost:8000/api/v1/files/exports/archive
```

### Run API with Docker

```bash
docker run --ipc=host --gpus all -p 8000:8000 \
  -v ./src/datasets:/app/src/datasets \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/exports:/app/src/exports \
  -v ./src/logs:/app/src/logs \
  -e API=true -e NODE_ID=node-a -e API_TOKEN=<random string, 16+ characters> \
  ghcr.io/parrotxray/mantis-trainer:main
```

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
docker pull ghcr.io/parrotxray/mantis-trainer:main
```

### Run with Docker

```bash
docker run --ipc=host --gpus all \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/plots:/app/src/plots \
  -v ./src/exports:/app/src/exports \
  -v ./src/logs:/app/src/logs \
  -v ./src/datasets:/app/src/datasets \
  -e ALL=true \
  -e ARGS="da.epochs=50" \
  ghcr.io/parrotxray/mantis-trainer:main
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `API` | Start the API server instead of the CLI pipeline (`true`/`false`) |
| `DATAPATH` | Optional: CSV directory, sets `DATASET_ROOT` (default `src/datasets`) |
| `ARGS` | Extra parameters, same as `--args` |
| `ALL` | Run complete pipeline (`true`/`false`) |
| `DATAPREPROCESS` | Run preprocessing (`true`/`false`) |
| `DEEPAUTOENCODER` | Train autoencoder (`true`/`false`) |
| `EXPORT` | Export to ONNX (`true`/`false`) |
| `FINETUNE` | Run stage `ft` (`true`/`false`); pick the model with `ARGS="run.model_dir=..."` |
| `SCORE` | Run stage `sc` (`true`/`false`); pick the model with `ARGS="run.model_dir=..."` |
| `FEATUREANALYSIS` | Per-feature error analysis of the trained model in `artifacts/`, no retraining (`true`/`false`) |

### Build Docker Image Locally

```bash
docker build -t mantis-trainer .
```

The image is based on `ubuntu:24.04`: the CUDA build of PyTorch from PyPI brings its own CUDA / cuDNN libraries, and the
GPU driver comes from the host through the NVIDIA Container Toolkit. PyTorch comes in as a dependency of Lightning.
Datasets, outputs, logs and every CSV / parquet are left out of the
build context. See the guide's Docker chapter for details.

## Output Directories

| Directory | Contents |
|-----------|----------|
| `outputs/` | Processed CSV files (benign/attack splits, AE scores) |
| `artifacts/` | Trained model files (PyTorch checkpoint, ONNX, config pkl) |
| `plots/` | Training visualizations (ROC, PR curve, t-SNE, score distribution) |
| `exports/` | ONNX model and inference config JSON |
| `logs/` | Process output records (`logs/jobs/<id>.log` per API job) |
| `datasets/` | CSV pool read by stage `dp` |

## Limitations

- Attack evaluation uses CIC-IDS-2017 (2017) and FLNET2023 (2023) benchmark datasets generated in controlled environments. Some attack categories (Exploitation, Reconnaissance) exhibit flow-level statistical features similar to BENIGN traffic, resulting in lower detection rates. This is a known limitation of flow-based NIDS datasets [Dube 2024, Engelen et al. 2021].
- The autoencoder is trained on BENIGN-only traffic. Detection capability for novel attacks depends on how much their flow statistics deviate from normal behaviour.

## Acknowledgments

- Canadian Institute for Cybersecurity for the CIC-IDS datasets
- FLNET2023 dataset authors
- PyTorch and PyTorch Lightning teams

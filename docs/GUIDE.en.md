# Mantis-Trainer Operations Guide

This guide covers every operation: preparing data, running each stage, setting parameters, operating several nodes over the API, collecting results, and fixing common errors. [繁體中文版](GUIDE.md)

- [1. Concepts](#1-concepts)
- [2. Install and Start](#2-install-and-start)
- [3. Prepare Data](#3-prepare-data)
- [4. Stages](#4-stages)
- [5. Parameters](#5-parameters)
- [6. CLI](#6-cli)
- [7. API](#7-api)
- [8. Docker](#8-docker)
- [9. Multiple Nodes](#9-multiple-nodes)
- [10. Common Workflows](#10-common-workflows)
- [11. Output Files](#11-output-files)
- [12. Errors and Troubleshooting](#12-errors-and-troubleshooting)

---

## 1. Concepts

### Three entry points, one core

```
src/pipeline/   Shared core: stages, stage rules, parameter parser, CSV pool, runner
src/main.py     CLI: runs on the local machine
src/api/        API: operated over HTTP; jobs run in a separate worker process
Docker          Starts the CLI or the API from environment variables
```

All three use the same stage names, the same parameter syntax and the same validation rules. Parameters tried on the CLI can be pasted unchanged into the API or Docker.

### Terms

| Term | Meaning |
|---|---|
| Node | A machine running Mantis-Trainer, usually one per GPU host |
| Workspace | The node's working directory, `src/` by default. `datasets/`, `outputs/`, `artifacts/`, `plots/`, `exports/` and `logs/` live here |
| CSV pool | Every CSV the node preprocesses, `src/datasets/` by default |
| Stage | One processing step, such as preprocessing, training or export |
| Job | One API run, which can contain several stages |

---

## 2. Install and Start

### Install

```bash
git clone https://github.com/ParrotXray/Mantis-Trainer.git
```

```bash
cd Mantis-Trainer
```

```bash
pip3 install -r requirements.txt
```

PyTorch comes in as a dependency of Lightning.

Python 3.12, an NVIDIA GPU and 32 GB of RAM or more are recommended. Without a GPU the pipeline falls back to the CPU, which is much slower.

### Start the CLI

Run every CLI command from `src/`:

```bash
cd src
```

```bash
./main.py --help
```

### Start the API

```bash
cd src
```

```bash
API_TOKEN=$(openssl rand -hex 32) NODE_ID=node-a ./server.py
```

Open `http://<host>:8000/docs` in a browser to see and try every endpoint.

API environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `NODE_ID` | `<hostname>-<random>` | Node name, included in every response |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `WORKSPACE` | `.` | Working directory |
| `DATASET_ROOT` | `<WORKSPACE>/datasets` | CSV pool location |
| `MAX_UPLOAD_MB` | `10240` | Upload limit per file, in MB |
| `CPU_SAMPLE_INTERVAL` | `0.2` | CPU sampling window, in seconds |
| `WS_MIN_INTERVAL` / `WS_MAX_INTERVAL` | `0.5` / `60` | Allowed WebSocket push interval, in seconds |
| `API_TOKEN` | required | Token every API call must carry, at least 16 characters (see below) |
| `CORS_ORIGINS` | none | Browser origins allowed to call the API, comma-separated, e.g. `http://console.local:5173,https://ops.example.com` |

### Authentication

`API_TOKEN` is required: without it, or with fewer than 16 characters, the API server refuses to start. Use a random string, for example:

```bash
openssl rand -hex 32
```

```bash
API_TOKEN=<the string above> NODE_ID=node-a ./server.py
```

Every endpoint except `/api/v1/health` needs the token, in one of two ways:

- HTTP header: `Authorization: Bearer <token>`
- Query parameter: `?token=<token>` (for a browser `EventSource`, WebSocket or download link, which cannot set headers)

A missing or wrong token gets 401 over HTTP; a WebSocket is closed with code 1008. Query parameters can end up in proxy logs and browser history, so prefer the header whenever possible.

The `curl` examples in the rest of this guide leave the token out for brevity; add `-H "Authorization: Bearer <token>"` to each command.

### Browser frontends (CORS)

A browser frontend served from a different address needs its address in `CORS_ORIGINS` on every node. Without it, no cross-origin request is allowed. `curl` and server-side callers are not affected by CORS.

---

## 3. Prepare Data

### CSV pool

Preprocessing (`dp`) always reads every CSV in the CSV pool, subdirectories included. Datasets need no names.

- Default location: `src/datasets/`
- Another location: set `DATASET_ROOT`. Both the CLI and the API read it; Docker sets it from `DATAPATH`.

Every CSV needs a `Label` column. Normal traffic is labelled `BENIGN` (or `Normal`); other values are mapped to attack classes such as DoS, DDoS, Brute Force, Reconnaissance and Exploitation. Both common CIC-IDS-2017 / CICFlowMeter column spellings are supported.

CSVs recorded by Mantis may carry an `ML Verdict` column; see `dp.exclude_verdicts` in section 5.3.

### Import option 1: copy files

Put the CSVs in `src/datasets/`.

### Import option 2: upload over the API

```bash
curl -F files=@lab301_benign.csv -F files=@attack_test.csv http://node-a:8000/api/v1/datasets
```

- File names are kept; a file with the same name is overwritten.
- `?replace=true` empties the CSV pool before importing.
- Only `.csv` files are accepted, and one upload cannot repeat a file name.
- Files are written to a staging area (`datasets/.staging`, inside the pool so the final move is a rename even when `datasets` is a Docker volume) and moved into the pool only after every file succeeds. A failed upload leaves the existing files untouched. Folders whose name starts with `.` are never part of the pool.
- A disk or file system error while storing an upload answers `507` with the reason (for example `No space left on device`). Any other unexpected error answers `500` with `{"detail": ...}` and is logged with its traceback, instead of dropping the connection.
- While a job that includes `dp` is running, imports and deletes are refused (409).

### List and delete

```bash
curl http://node-a:8000/api/v1/datasets
```

```bash
curl -X DELETE http://node-a:8000/api/v1/datasets/attack_test.csv
```

```bash
curl -X DELETE http://node-a:8000/api/v1/datasets
```

The first lists every CSV with the total size, the second deletes one file, the third empties the pool.

---

## 4. Stages

| Code | CLI flag | Docker variable | What it does | Needs |
|---|---|---|---|---|
| `dp` | `-dp` | `DATAPREPROCESS=true` | Preprocesses the CSV pool into `outputs/preprocessing_*.parquet` | CSVs in the pool |
| `da` | `-da` | `DEEPAUTOENCODER=true` | Trains the LSTM Autoencoder from scratch into `artifacts/` and `plots/` | `dp` output |
| `ft` | `-ft` | `FINETUNE=true` | Finetunes a finished model on the current data | A model directory, `dp` output |
| `sc` | `-sc` | `SCORE=true` | Scores data with a finished model and its thresholds, no training | A model directory, `dp` output |
| `fa` | `-fa` | `FEATUREANALYSIS=true` | Finds which features and IPs cause false positives of the model in `artifacts/` on benign traffic | Model in `artifacts/`, `dp` output |
| `ex` | `-ex` | `EXPORT=true` | Exports ONNX and the inference config to `exports/` | Model in `artifacts/` |
| `all` | `-a` | `ALL=true` | Same as `dp` + `da` + `ex` | CSVs in the pool |

### Order

Stages always run in the order `dp → da → ft → sc → fa → ex`, whatever order they were picked in. `all` can be combined with other stages: `all` + `fa` runs `dp → da → fa → ex`.

### Combinations that are refused

| Combination | Reason |
|---|---|
| `ft` + `da` | `ft` is itself training and replaces `da` |
| `sc` + `dp` | `sc` reads preprocessed data; run `dp` on its own first. Scored data is usually preprocessed in a separate workspace so it does not replace the training data |

### Finetuning backs up the model

When `ft` uses the model in `./artifacts`, the original model is first copied to `artifacts/pretrained_<time>/`, then the new model is written.

---

## 5. Parameters

### 5.1 Syntax

One line of `section.name=value` pairs separated by spaces:

```
da.epochs=50 da.batch_size=4096 dp.exclude_verdicts=none run.model_dir=./old_model
```

- The CLI takes it in `--args "..."`, the API in the JSON field `args`, Docker in `ARGS`. The syntax is identical.
- Every parameter needs its section prefix:

| Prefix | Config | Stages affected |
|---|---|---|
| `dp.` | PreprocessConfig | `dp` |
| `da.` | DeepAutoencoderConfig | `da`, `ft`, `sc`, `fa` |
| `ex.` | ExportConfig | `ex` |
| `run.` | Run options | See section 5.6 |

- Values are converted to the field's type; a value that cannot be converted is rejected before anything runs.
- Multiple values are comma-separated, e.g. `dp.exclude_verdicts=confirmed_attack,rule_match`.
- `none` means empty: it clears a list (`dp.exclude_verdicts=none`) or turns off an optional setting (`da.max_bound_z=none`).
- Quote values that contain spaces: `ex.version="1.2.0 beta"`.
- A parameter can appear only once.
- Parameters left out keep their defaults. `GET /api/v1/jobs/params` lists every default.

### 5.2 Everything is checked before running

Nothing runs if any of these is wrong:

- Missing prefix, unknown prefix, unknown name
- Wrong type (e.g. `da.epochs=abc`)
- Duplicate parameter
- Conflicting stages
- `run.resume` without `da`, or `run.model_dir` without `ft` / `sc`
- The checkpoint or model directory does not exist
- `dp` is selected but the CSV pool is empty

### 5.3 `dp.` (preprocessing)

| Parameter | Default | Purpose |
|---|---|---|
| `dp.exclude_verdicts` | `confirmed_attack,rule_match` | `ML Verdict` values whose rows are dropped. `none` keeps every row |
| `dp.exclude_src_ips` | empty | Rows whose source IP is in one of these IPs or CIDR blocks are dropped; see below |
| `dp.exclude_dst_ips` | empty | Rows whose destination IP is in one of these IPs or CIDR blocks are dropped; see below |
| `dp.clip_min` | `-1e9` | Lower bound for values |
| `dp.clip_max` | `1e9` | Upper bound for values |
| `dp.fill_value` | `0.0` | Replacement for missing and infinite values |

`ML Verdict` is the verdict Mantis gave each flow live:

| Value | Meaning |
|---|---|
| `normal` | Model judged it normal |
| `anomaly` | Model judged it anomalous, no other evidence |
| `rule_match` | Suricata rule match |
| `structural` | Aggregator FLOOD / SCAN / TARGETED verdict |
| `confirmed_attack` | Both a rule match and an aggregator verdict |
| `suppressed` | Confirmed as a false positive by an operator |
| `uninspected` | Not inspected |

By default only the rule-backed `confirmed_attack` and `rule_match` rows are dropped. `structural` is kept because busy normal clients trigger it too. `anomaly` is kept because dropping it would trim the benign tail on every retrain and narrow the model each cycle. When preprocessing data for `sc` to compare with the live alert rate, use `dp.exclude_verdicts=none`. CSVs without the column are not affected.

**Excluding IPs**: `dp.exclude_src_ips=10.0.0.5,192.168.50.0/24` drops the rows whose source IP is in the list, which removes those hosts' whole sequences. It is for keeping bad data out of what the model learns as normal: a compromised host, a pentest box, a scanner. It is **not** for silencing a host that alerts too often: a host the model never saw scores higher once deployed; use a suppression rule in Mantis for that.

`dp.exclude_dst_ips` excludes by destination IP. It also removes other hosts' flows to those addresses from the middle of their sequences, while Mantis still sees them live, so prefer `exclude_src_ips`. A malformed address is refused before anything runs (422). The preprocessing log lists the rows and addresses each rule excluded and warns when a rule matched nothing.

### 5.4 `da.` (training, finetuning, analysis)

**Training**

| Parameter | Default | Purpose |
|---|---|---|
| `da.epochs` | `1000` | Maximum epochs; training usually stops early |
| `da.batch_size` | `8192` | Training batch size. Lower it if GPU memory runs out |
| `da.inference_batch_size` | `1024` | Inference batch size |
| `da.learning_rate` | `0.001` | Learning rate |
| `da.weight_decay` | `1e-5` | AdamW weight decay |
| `da.clipnorm` | `1.0` | Gradient clipping |
| `da.early_stopping_patience` | `5` | Epochs without validation improvement before stopping |
| `da.early_stopping_min_delta` | `1e-5` | Smallest loss drop that counts as improvement; also the threshold for lowering the learning rate |
| `da.reduce_lr_patience` | `3` | Epochs without improvement before lowering the learning rate |
| `da.reduce_lr_factor` | `0.5` | Factor applied on each reduction |
| `da.min_lr` | `1e-7` | Learning rate floor |
| `da.finetune_lr_scale` | `0.1` | Learning rate multiplier for `ft` |

**Model structure**

| Parameter | Default | Purpose |
|---|---|---|
| `da.window_size` | `15` | Flows per sequence |
| `da.stride` | `1` | Flows the window moves each step |
| `da.hidden_size` | `128` | LSTM hidden size |
| `da.num_layers` | `4` | LSTM layers |
| `da.encoding_dim` | `64` | Bottleneck size |
| `da.dropout` | `0.2` | Dropout rate |
| `da.latent_norm_weight` | `1e-3` | Weight that pulls benign latent vectors toward the origin |

`ft` keeps the original model's structure and preprocessing; structure parameters have no effect on `ft`.

**Splits and augmentation**

| Parameter | Default | Purpose |
|---|---|---|
| `da.validation_split` | `0.10` | Validation share |
| `da.test_split` | `0.15` | Test share |
| `da.split_random_state` | `42` | Split random seed |
| `da.synthetic_augmentation_ratio` | `0.2` | Share of extra synthetic training sequences; `0` turns it off |

**Scaling and clipping**

| Parameter | Default | Purpose |
|---|---|---|
| `da.winsorize_lower` / `da.winsorize_upper` | `0.005` / `0.995` | Quantiles used to cut extreme values |
| `da.max_bound_z` | `4.0` | Maximum distance, in standard deviations, of a cut bound from the mean; `none` turns it off |
| `da.tail_check_z` | `4.0` | Features whose bound is further than this many standard deviations get an extra false-positive check |
| `da.clip_min` / `da.clip_max` | `-5.0` / `5.0` | Clip range after scaling |
| `da.fill_value` | `0.0` | Replacement for missing values |

**Analysis (`fa`, `sc`, and the reports at the end of `da` / `ft`)**

| Parameter | Default | Purpose |
|---|---|---|
| `da.analysis_threshold` | `p99` | Threshold used in reports. One of `p90`–`p99`, `p99.5`, `p99.9`, `mean+1std`, `mean+2std`, `Q3+1.5IQR`, `Q3+3.0IQR` |
| `da.analysis_top_hosts` | `3` | How many IPs with the most false positives are listed |

### 5.5 `ex.` (export)

| Parameter | Default | Purpose |
|---|---|---|
| `ex.opset_version` | `18` | ONNX opset version |
| `ex.post_scaling_clip_min` / `ex.post_scaling_clip_max` | `-6.0` / `6.0` | Post-scaling clip range written to the inference config |
| `ex.verify_atol` | `1e-3` | Allowed difference between ONNX and PyTorch outputs during verification |
| `ex.version` | `1.0.0` | Model version string, written to `version` in `exports/full_config.json` |

### 5.6 `run.` (run options)

| Parameter | Default | Purpose | Only with |
|---|---|---|---|
| `run.resume` | none | Resume training from a checkpoint, e.g. `run.resume=autoencoder_temp-v14.ckpt`; a bare file name is looked up in `./artifacts` first | `da` |
| `run.model_dir` | `./artifacts` | Model directory for `ft` / `sc`; must contain `deep_autoencoder.pt` and `deep_ae_config.pkl`; a bare name is looked up in `./artifacts` first, e.g. `run.model_dir=pretrained_20261110_0412` | `ft`, `sc` |

Over the API, both paths must stay inside the node's workspace, or the request gets 400.

---

## 6. CLI

```
./main.py [-a] [-dp] [-da] [-ft] [-sc] [-fa] [-ex] [--args "section.name=value ..."]
```

Flags only pick stages; every parameter goes in `--args`. Examples:

```bash
./main.py -a
```

```bash
./main.py -a -fa --args "da.epochs=100 da.batch_size=4096"
```

```bash
./main.py -dp --args "dp.exclude_verdicts=none"
```

```bash
./main.py -da --args "run.resume=autoencoder_temp-v14.ckpt"
```

```bash
./main.py -ft --args "run.model_dir=./artifacts da.epochs=20 da.finetune_lr_scale=0.05"
```

```bash
./main.py -sc --args "run.model_dir=./old_model da.analysis_threshold=p99.5"
```

```bash
./main.py -fa --args "da.analysis_threshold=p95 da.analysis_top_hosts=5"
```

```bash
./main.py -ex --args "ex.opset_version=17"
```

```bash
DATASET_ROOT=/data/mantis-csv ./main.py -dp
```

The CLI runs in the foreground and logs to both the terminal and `logs/train.log`. On a parameter error it prints the reason and exits without running anything.

---

## 7. API

Every path starts with `/api/v1`. The examples use `node-a:8000` as the node address.

### 7.1 Node status

```bash
curl http://node-a:8000/api/v1/health
```

```bash
curl http://node-a:8000/api/v1/node
```

`/health` is the liveness check and needs no token. `/node` returns the node name, host name, whether it is busy (`busy`), the current job and the supported stages.

### 7.2 Performance monitoring

One-off query:

```bash
curl http://node-a:8000/api/v1/metrics
```

`/metrics` returns CPU, RAM and GPU together; `/metrics/cpu`, `/metrics/memory` and `/metrics/gpu` return one each.

- CPU: physical / logical cores, overall and per-core usage, frequency, load average
- RAM: total, available, used, usage, swap
- GPU: name, usage, memory, temperature and power for each card. `available` is `false` when there is no NVIDIA GPU

Live push (WebSocket):

```
ws://node-a:8000/api/v1/metrics/ws?interval=1
```

A JSON message in the same format as `GET /metrics` arrives every `interval` seconds. `interval` must be between 0.5 and 60 seconds, or the connection is refused. Add `&token=<token>`.

`/node/ws?interval=5` pushes `node`, `metrics` and `jobs` on one socket (`jobs` is `null` when it did not change since the previous message); the Mantis web UI uses it instead of polling `/node`, `/metrics` and `/jobs`.

### 7.3 Datasets

See [3. Prepare Data](#3-prepare-data).

### 7.4 Start a job

```bash
curl -X POST http://node-a:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"stages": ["all", "fa"], "args": "da.epochs=50 da.analysis_threshold=p95"}'
```

- `stages`: array of stage codes, see [4. Stages](#4-stages)
- `args`: parameter string, see [5. Parameters](#5-parameters); optional

Success returns 202 with the `job_id`, its status, the stages that will run and the CSV files `dp` will read.

A node runs one job at a time and returns 409 while busy. Jobs run in a separate process, so the API keeps responding.

List every parameter default:

```bash
curl http://node-a:8000/api/v1/jobs/params
```

Check a request without starting it (same body as `POST /jobs`):

```bash
curl -X POST http://node-a:8000/api/v1/jobs/check \
  -H "Content-Type: application/json" \
  -d '{"stages": ["ft"], "args": "run.model_dir=./artifacts/pretrained_20261110_0412"}'
```

It applies every rule `POST /jobs` applies (stage combinations, `args` keys and types, CSV files for `dp`, `run.model_dir` / `run.resume` paths) and returns 200 with the stages that would run, the CSV files `dp` would read and the resolved `model_dir` / `resume`, or the same 422 / 400 error a start would return. It does not check whether the node is busy.

### 7.5 Query and cancel jobs

```bash
curl http://node-a:8000/api/v1/jobs
```

```bash
curl http://node-a:8000/api/v1/jobs/<job_id>
```

```bash
curl -X DELETE http://node-a:8000/api/v1/jobs/<job_id>
```

The first lists jobs (up to 200 kept), the second shows one job, the third cancels a job.

Job states:

| State | Meaning |
|---|---|
| `pending` | Created, not started yet |
| `running` | Running |
| `succeeded` | Finished successfully (`exit_code` 0) |
| `failed` | Failed; the reason is in the job log |
| `cancelled` | Cancelled |

Each job also carries `stage_progress`, one entry per stage in run order with `status` and `started_at` / `finished_at` (unix seconds), and `current_stage`, the stage running now (`null` otherwise):

| Stage status | Meaning |
|---|---|
| `pending` | Not reached yet |
| `running` | Running now |
| `succeeded` | Finished |
| `failed` / `cancelled` | Was running when the job failed or was cancelled |
| `skipped` | Never ran, because the job ended earlier |

The worker writes these times to `logs/jobs/<job_id>.stages.json` as each stage starts and finishes.

Cancelling stops the job process and all of its children (such as DataLoader workers). `cancel_requested` becomes `true` at once; the state turns `cancelled` only after the process has exited, and the node stays busy until then. A cancelled job's log often ends with `DataLoader worker ... killed by signal`; that is expected.

Job records are stored in `logs/jobs/<job_id>.job.json` and survive an API restart. Stopping the server stops running jobs; after a restart those jobs are marked `failed` and the node accepts new work again.

### 7.6 Logs

Job log:

```bash
curl "http://node-a:8000/api/v1/jobs/<job_id>/logs?lines=200"
```

```bash
curl -N http://node-a:8000/api/v1/jobs/<job_id>/logs/stream
```

The first returns the last N lines (1–10000); the second follows the log live (Server-Sent Events). The live stream first sends the last 50 lines (change with `?lines=`), then each new line, and sends `event: end` when the job finishes.

All log files:

```bash
curl http://node-a:8000/api/v1/logs
```

```bash
curl "http://node-a:8000/api/v1/logs/train.log?lines=200"
```

`train.log` is the node-wide log (rotated daily, 30 days kept); `jobs/<job_id>.log` is one job's log.

### 7.7 Download results

Categories: `exports`, `artifacts`, `plots`, `outputs`, `logs`.

```bash
curl http://node-a:8000/api/v1/files/exports
```

```bash
curl -OJ "http://node-a:8000/api/v1/files/exports/download?path=inference_config.json"
```

```bash
curl -OJ http://node-a:8000/api/v1/files/exports/archive
```

```bash
curl -OJ "http://node-a:8000/api/v1/files/logs/archive?path=jobs"
```

In order: list files, download one file, download a whole category as zip, download a subdirectory of a category as zip. Zip names carry the node name (e.g. `node-a-exports.zip`), so results from several nodes do not collide.

---

## 8. Docker

### Building the image

```bash
docker build -t mantis-trainer .
```

- The base is `ubuntu:24.04`, not `nvidia/cuda`: the pip-installed CUDA build of PyTorch ships its own CUDA / cuDNN libraries (about 3.3 GiB of wheels), so the old `nvidia/cuda:*-cudnn-runtime` base (2.7 GiB compressed) only duplicated them. The GPU driver still comes from the host through the NVIDIA Container Toolkit; `--gpus all` is unchanged.
- Packages and code are separate layers: a code-only change rebuilds and `docker pull`s only the last one. PyTorch comes in once as a dependency of Lightning; the old image installed the newest torch from PyTorch's index and then `requirements.txt`'s limit made pip download torch and every NVIDIA library again.
- `.dockerignore` leaves out `src/datasets`, `src/outputs`, `src/logs` and every `*.csv` / `*.parquet`, so building on a node that already holds data does not bake it into the image.
- CI pushes to GHCR with zstd compression (`oci-mediatypes`); pulling needs Docker Engine 23 or newer.

### Batch run (CLI mode)

```bash
docker run --ipc=host --gpus all \
  -v ./src/datasets:/app/src/datasets \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/plots:/app/src/plots \
  -v ./src/exports:/app/src/exports \
  -v ./src/logs:/app/src/logs \
  -e ALL=true \
  -e ARGS="da.epochs=100 da.batch_size=4096" \
  ghcr.io/parrotxray/mantis-trainer:main
```

| Variable | Purpose |
|---|---|
| `ALL` / `DATAPREPROCESS` / `DEEPAUTOENCODER` / `FINETUNE` / `SCORE` / `FEATUREANALYSIS` / `EXPORT` | `true` runs the stage |
| `ARGS` | Parameter string, same syntax as `--args` |
| `DATAPATH` | CSV pool location inside the container; sets `DATASET_ROOT` |

Pick the model for `ft` / `sc` with `run.model_dir=...` in `ARGS`.

### Server mode (API)

```bash
docker run -d --ipc=host --gpus all -p 8000:8000 \
  -v ./src/datasets:/app/src/datasets \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/plots:/app/src/plots \
  -v ./src/exports:/app/src/exports \
  -v ./src/logs:/app/src/logs \
  -e API=true -e NODE_ID=node-a -e API_TOKEN=replace-with-a-random-string \
  ghcr.io/parrotxray/mantis-trainer:main
```

With `API=true`, the stage variables and `ARGS` are ignored; jobs come from the API. Other API variables are listed in [Start the API](#start-the-api).

---

## 9. Multiple Nodes

Each GPU host runs its own API node with its own CSV pool, models and outputs; nothing is shared between nodes.

1. Start the API on each host with a different `NODE_ID`, and set `API_TOKEN`.
2. Upload the CSVs to the nodes that will train.
3. Before dispatching, query `GET /node` on each node and pick one whose `busy` is `false`; `GET /metrics` shows GPU usage and memory as well.
4. Submit the job. If the node was taken in the meantime it returns 409; try the next one.
5. Poll `GET /jobs/<id>` or follow `/logs/stream`, then collect results from `/files/exports/archive`.

A browser frontend that talks to several nodes directly needs its address in `CORS_ORIGINS` on every node.

---

## 10. Common Workflows

### First training

1. Put CSVs in the pool (copy to `src/datasets/` or upload over the API).
2. Run the full pipeline plus feature analysis:

```bash
./main.py -a -fa
```

3. Read the `Threshold Analysis [Val Set — BENIGN only]` table in `logs/train.log` and pick a threshold that meets your false-positive tolerance.
4. Deploy `deep_autoencoder.onnx` and `inference_config.json` from `exports/` to Mantis, and set the chosen threshold in Mantis.

### Resume interrupted training

```bash
./main.py -da -ex --args "run.resume=autoencoder_temp.ckpt"
```

Use the name of a `.ckpt` that actually exists in `artifacts/`.

### Finetune on new data

1. Empty the CSV pool and add the new data.
2. Preprocess, finetune and export:

```bash
./main.py -dp
```

```bash
./main.py -ft -ex --args "da.epochs=20"
```

The original model is backed up to `artifacts/pretrained_<time>/`.

### Compare a model with the live alert rate

1. Prepare a separate workspace (another directory) with the CSVs of the period to score, and preprocess them with `dp.exclude_verdicts=none` to keep every row.
2. Score with the model to compare:

```bash
./main.py -sc --args "run.model_dir=/path/to/artifacts da.analysis_threshold=p99"
```

3. The report groups windows by `ML Verdict` and shows each group's share over the threshold. Comparing a model before and after finetuning, the `suppressed` share should drop and the `normal` share should not rise.

### Investigate false positives

```bash
./main.py -fa --args "da.analysis_threshold=p95 da.analysis_top_hosts=5"
```

The report lists the features behind most benign false positives, and for the IPs with the most false positives, their daily false-positive rate and the destinations involved.

### Run the full pipeline over the API and collect results

```bash
curl -F files=@benign.csv -F files=@attack.csv "http://node-a:8000/api/v1/datasets?replace=true"
```

```bash
curl -X POST http://node-a:8000/api/v1/jobs -H "Content-Type: application/json" -d '{"stages": ["all"], "args": "da.epochs=100"}'
```

```bash
curl -N http://node-a:8000/api/v1/jobs/<job_id>/logs/stream
```

```bash
curl -OJ http://node-a:8000/api/v1/files/exports/archive
```

```bash
curl -OJ http://node-a:8000/api/v1/files/plots/archive
```

---

## 11. Output Files

| Location | Contents | Written by |
|---|---|---|
| `datasets/` | CSV pool | You |
| `outputs/preprocessing_benign.*`, `preprocessing_attack.*` | Preprocessed data (CSV and parquet) | `dp` |
| `outputs/deep_ae_scores.csv` | Test set scores | `da`, `ft` |
| `artifacts/deep_autoencoder.pt` | PyTorch model | `da`, `ft` |
| `artifacts/deep_ae_config.pkl` | Preprocessing parameters, thresholds (`ae_thresholds`) and other settings | `da`, `ft` |
| `artifacts/*.ckpt` | Training checkpoints | `da`, `ft` |
| `artifacts/pretrained_<time>/` | Backup of the model before finetuning | `ft` |
| `plots/*.png` | Score distribution, ROC/PR, CDF and t-SNE plots | `da`, `ft` |
| `exports/deep_autoencoder.onnx` | Model for deployment | `ex` |
| `exports/inference_config.json` | Inference config for deployment | `ex` |
| `exports/full_config.json` | Full config, including `version` | `ex` |
| `logs/train.log` | Node-wide log | All |
| `logs/jobs/<job_id>.log` | One job's log | API |
| `logs/jobs/<job_id>.job.json` | Job record (state, stages, parameters, times) | API |
| `logs/jobs/<job_id>.stages.json` | Start and finish time of each stage | API |

---

## 12. Errors and Troubleshooting

### API status codes

| Code | Meaning | Common cause |
|---|---|---|
| 401 | Unauthorized | The request has no token or a wrong one |
| 400 | Invalid name or path | Non-`.csv` upload, repeated file name, path leaving the workspace (`../`) |
| 404 | Not found | Job, file or log does not exist |
| 409 | Conflict | Node is running another job, CSV pool changed while `dp` runs, cancelling a finished job |
| 413 | File too large | Larger than `MAX_UPLOAD_MB` |
| 422 | Invalid parameters or stages | See below |

### Parameter error messages

| Message | Fix |
|---|---|
| `Parameter 'epochs' needs a prefix, use da.epochs` | Add the prefix |
| `Unknown prefix 'zz' ...` | The prefix must be `dp`, `da`, `ex` or `run` |
| `Unknown parameter 'da.xxx'` | Misspelled name, or the parameter belongs to another section |
| `Invalid value for DeepAutoencoderConfig.epochs: 'abc'` | Wrong value type |
| `Duplicate parameter 'da.epochs'` | The same parameter appears twice |
| `Expected section.key=value, got '...'` | `=` is missing |
| `Invalid args: No closing quotation` | Unbalanced quotes |
| `Stage 'ft' replaces 'da' training ...` | Pick either `ft` or `da` |
| `Stage 'sc' reads the preprocessed parquet ...` | Run `dp` on its own first, then `sc` |
| `'run.resume' only applies to stage 'da'` | Remove `run.resume`, or add `da` |
| `'run.model_dir' only applies to stages 'ft' and 'sc'` | Remove `run.model_dir`, or run `ft` / `sc` |
| `Model directory '...' not found` | Check that the model directory exists |
| `Checkpoint '...' not found` | Check the checkpoint path |
| `Stage 'dp' requires CSV files in ...` | Import CSVs first |

### Runtime problems

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Lower `da.batch_size` or `da.inference_batch_size` |
| Job exits with `-9` without a cancel | The system ran out of memory and the OOM killer stopped it (`dmesg \| grep -i oom` confirms). The training windows `da` builds take about `rows × window_size × features × 4 bytes`: raise `da.stride`, lower `da.synthetic_augmentation_ratio`, or train on fewer days at once. Swap is only a safety net; training slows to a crawl once the windows exceed physical memory |
| `da` cannot find `outputs/preprocessing_benign.parquet` | Run `dp` first |
| `ft` / `sc` / `fa` / `ex` cannot find a model | Run `da` first, or point `run.model_dir` at a model (`fa` and `ex` only read `artifacts/`) |
| `sc` reports `Unknown threshold` | `da.analysis_threshold` must be one of the threshold names saved with the model, see the analysis parameters in section 5.4 |
| WebSocket does not connect | Check that `uvicorn[standard]` (which includes `websockets`) is installed, `interval` is in range, and `?token=` is present |
| Browser shows a CORS error | Add the frontend address to `CORS_ORIGINS` on the node and restart the API |
| A job stays `running` | Follow `/jobs/<id>/logs/stream`; cancel with `DELETE /jobs/<id>` if needed |

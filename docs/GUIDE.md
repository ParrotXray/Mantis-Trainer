# Mantis-Trainer 操作指南

這份指南涵蓋所有操作：準備資料、執行各種任務、設定參數、用 API 遠端操作多台節點、取回結果，以及常見錯誤的處理方式。 [English version](GUIDE.en.md)

- [1. 基本概念](#1-基本概念)
- [2. 安裝與啟動](#2-安裝與啟動)
- [3. 準備資料](#3-準備資料)
- [4. 任務](#4-任務)
- [5. 參數](#5-參數)
- [6. 用 CLI 操作](#6-用-cli-操作)
- [7. 用 API 操作](#7-用-api-操作)
- [8. 用 Docker 操作](#8-用-docker-操作)
- [9. 多節點部署](#9-多節點部署)
- [10. 常見流程](#10-常見流程)
- [11. 輸出檔案](#11-輸出檔案)
- [12. 錯誤與排除](#12-錯誤與排除)

---

## 1. 基本概念

### 三個入口，同一個核心

```
src/pipeline/   共用核心：任務、組合規則、參數解析、CSV 池、執行器
src/main.py     CLI：在本機直接執行
src/api/        API：透過 HTTP 操作，任務在獨立子程序執行
Docker          用環境變數啟動 CLI 或 API
```

三個入口用相同的任務代號、相同的參數寫法、相同的檢查規則。在 CLI 試好的參數，可以原封不動貼到 API 或 Docker。

### 名詞

| 名詞 | 意思 |
|---|---|
| 節點 | 一台執行 Mantis-Trainer 的機器，通常一台 GPU 主機一個節點 |
| Workspace | 節點的工作目錄，預設是 `src/`。`datasets/`、`outputs/`、`artifacts/`、`plots/`、`exports/`、`logs/` 都在這裡面 |
| CSV 池 | 節點上所有要拿來前處理的 CSV，預設在 `src/datasets/` |
| 任務（stage） | 一個處理步驟，例如前處理、訓練、匯出 |
| Job | API 送出的一次執行，可以包含多個任務 |

---

## 2. 安裝與啟動

### 安裝

```bash
git clone https://github.com/ParrotXray/Mantis-Trainer.git
```

```bash
cd Mantis-Trainer
```

```bash
pip3 install -r requirements.txt
```

PyTorch 會因為 Lightning 依賴它而自動安裝。

建議 Python 3.12、NVIDIA GPU、32 GB 以上記憶體。沒有 GPU 也能跑，會改用 CPU，速度慢很多。

### 啟動 CLI

所有 CLI 指令都在 `src/` 底下執行：

```bash
cd src
```

```bash
./main.py --help
```

### 啟動 API

```bash
cd src
```

```bash
API_TOKEN=$(openssl rand -hex 32) NODE_ID=node-a ./server.py
```

啟動後開瀏覽器到 `http://<主機>:8000/docs`，可以看到所有 API 並直接試打。

API 的環境變數：

| 變數 | 預設 | 用途 |
|---|---|---|
| `NODE_ID` | `<主機名>-<隨機碼>` | 節點名稱，每個回應都會帶上 |
| `API_HOST` | `0.0.0.0` | 監聽位址 |
| `API_PORT` | `8000` | 監聽埠 |
| `WORKSPACE` | `.` | 工作目錄 |
| `DATASET_ROOT` | `<WORKSPACE>/datasets` | CSV 池位置 |
| `MAX_UPLOAD_MB` | `10240` | 單一檔案上傳上限（MB） |
| `CPU_SAMPLE_INTERVAL` | `0.2` | CPU 取樣時間（秒） |
| `WS_MIN_INTERVAL` / `WS_MAX_INTERVAL` | `0.5` / `60` | WebSocket 推送間隔的允許範圍（秒） |
| `API_TOKEN` | 必填 | 所有 API 都要帶的 token，至少 16 個字元，見下方說明 |
| `CORS_ORIGINS` | 無 | 允許瀏覽器跨網域呼叫的來源，逗號分隔，例如 `http://console.local:5173,https://ops.example.com` |

### 身分驗證

`API_TOKEN` 必填：沒設定或少於 16 個字元時，API 伺服器會直接拒絕啟動。用隨機字串，例如：

```bash
openssl rand -hex 32
```

```bash
API_TOKEN=<上面產生的字串> NODE_ID=node-a ./server.py
```

除了 `/api/v1/health` 以外的所有 API 都要帶 token，兩種帶法擇一：

- HTTP header：`Authorization: Bearer <token>`
- 網址參數：`?token=<token>`（給瀏覽器的 `EventSource`、WebSocket、下載連結這類不能自訂 header 的情況）

沒帶或帶錯時，HTTP 回 401，WebSocket 以代碼 1008 關閉連線。網址參數可能被記錄在代理伺服器或瀏覽器紀錄中，能用 header 時請優先用 header。

本指南後面的 `curl` 範例為了簡潔都省略了 token，實際使用時請在每個指令加上 `-H "Authorization: Bearer <token>"`。

### 瀏覽器前端（CORS）

瀏覽器上的前端如果不是從同一個網址提供，需要在每個節點設定 `CORS_ORIGINS`，列出前端的網址。沒設定時不允許任何跨網域請求；`curl` 和程式呼叫不受 CORS 影響。

---

## 3. 準備資料

### CSV 池

前處理（`dp`）一律讀取 CSV 池裡的全部 CSV，包含子資料夾。不需要替資料集取名。

- 預設位置：`src/datasets/`
- 改位置：設定環境變數 `DATASET_ROOT`。CLI 和 API 都看這個變數；Docker 用 `DATAPATH` 設定。

CSV 必須有 `Label` 欄位。正常流量標成 `BENIGN`（或 `Normal`），其他值會依對照表歸成 DoS、DDoS、Brute Force、Reconnaissance、Exploitation 等攻擊類別。欄位名稱支援 CIC-IDS-2017 / CICFlowMeter 兩種常見寫法。

Mantis 錄下的 CSV 可能有 `ML Verdict` 欄位，用法見 5.3 節的 `dp.exclude_verdicts`。

### 匯入方式一：直接複製

把 CSV 放進 `src/datasets/` 即可。

### 匯入方式二：透過 API 上傳

```bash
curl -F files=@lab301_benign.csv -F files=@attack_test.csv http://node-a:8000/api/v1/datasets
```

- 保留原檔名，同名檔案會覆蓋。
- 加 `?replace=true` 會先清空 CSV 池再匯入。
- 只接受 `.csv`；同一次上傳不能有重複檔名。
- 檔案先寫到暫存區（`datasets/.staging`，放在 CSV 池裡面，所以就算 `datasets` 是 Docker volume，最後一步也只是改名），全部成功才搬進 CSV 池。中途失敗時，原本的檔案不受影響。名稱以 `.` 開頭的資料夾不算在 CSV 池裡。
- 存檔時遇到磁碟或檔案系統錯誤會回 `507` 並附上原因（例如 `No space left on device`）。其他未預期的錯誤回 `500` 和 `{"detail": ...}`，並在 log 記下完整 traceback，不會直接斷線。
- 有包含 `dp` 的 job 正在跑時不能匯入或刪除（409）。

### 查看與刪除

```bash
curl http://node-a:8000/api/v1/datasets
```

```bash
curl -X DELETE http://node-a:8000/api/v1/datasets/attack_test.csv
```

```bash
curl -X DELETE http://node-a:8000/api/v1/datasets
```

第一行列出所有 CSV 和總大小，第二行刪除單一檔案，第三行清空整個 CSV 池。

---

## 4. 任務

| 代號 | CLI 旗標 | Docker 變數 | 做什麼 | 需要先有 |
|---|---|---|---|---|
| `dp` | `-dp` | `DATAPREPROCESS=true` | 前處理 CSV 池，輸出 `outputs/preprocessing_*.parquet` | CSV 池裡有檔案 |
| `da` | `-da` | `DEEPAUTOENCODER=true` | 從頭訓練 LSTM Autoencoder，輸出 `artifacts/` 和 `plots/` | `dp` 的輸出 |
| `ft` | `-ft` | `FINETUNE=true` | 用目前資料 finetune 一個已訓練好的模型 | 模型資料夾、`dp` 的輸出 |
| `sc` | `-sc` | `SCORE=true` | 用已訓練好的模型和它的門檻替資料評分，不訓練 | 模型資料夾、`dp` 的輸出 |
| `fa` | `-fa` | `FEATUREANALYSIS=true` | 分析 `artifacts/` 模型在正常流量上的誤報來自哪些特徵、哪些 IP | `artifacts/` 的模型、`dp` 的輸出 |
| `ex` | `-ex` | `EXPORT=true` | 匯出 ONNX 與推論設定到 `exports/` | `artifacts/` 的模型 |
| `all` | `-a` | `ALL=true` | 等於 `dp` + `da` + `ex` | CSV 池裡有檔案 |

### 執行順序

不管怎麼選，一律照 `dp → da → ft → sc → fa → ex` 的順序執行。`all` 可以再加其他任務，例如 `all` + `fa` 會跑 `dp → da → fa → ex`。

### 不能一起選的組合

| 組合 | 原因 |
|---|---|
| `ft` + `da` | `ft` 本身就是訓練，取代 `da` |
| `sc` + `dp` | `sc` 讀的是前處理結果；要先單獨跑 `dp`。通常會把要評分的資料前處理到另一個 workspace，避免蓋掉訓練資料 |

### finetune 會自動備份

`ft` 使用 `./artifacts` 的模型時，會先把原模型備份到 `artifacts/pretrained_<時間>/`，再寫入新模型。

---

## 5. 參數

### 5.1 寫法

一行 `區段.名稱=值`，用空白分隔：

```
da.epochs=50 da.batch_size=4096 dp.exclude_verdicts=none run.model_dir=./old_model
```

- CLI 放在 `--args "..."`，API 放在 JSON 的 `args`，Docker 放在 `ARGS`。三者寫法完全相同。
- 每個參數都必須加區段前綴：

| 前綴 | 對應 | 影響的任務 |
|---|---|---|
| `dp.` | PreprocessConfig | `dp` |
| `da.` | DeepAutoencoderConfig | `da`、`ft`、`sc`、`fa` |
| `ex.` | ExportConfig | `ex` |
| `run.` | 執行選項 | 見 5.6 節 |

- 值會自動轉成欄位的型別；轉不過去會在執行前就報錯。
- 多個值用逗號分隔，例如 `dp.exclude_verdicts=confirmed_attack,rule_match`。
- `none` 代表空值：清空列表（`dp.exclude_verdicts=none`）或關閉選用設定（`da.max_bound_z=none`）。
- 值裡有空白就加引號：`ex.version="1.2.0 beta"`。
- 同一個參數只能出現一次。
- 沒寫的參數使用預設值。完整預設值可以用 `GET /api/v1/jobs/params` 查詢。

### 5.2 參數會在執行前全部檢查

以下情況都會在開始執行前被擋下，什麼都不會跑：

- 沒加前綴、前綴錯誤、名稱不存在
- 型別不對（例如 `da.epochs=abc`）
- 參數重複
- 任務組合衝突
- `run.resume` 用在沒有 `da` 的 job、`run.model_dir` 用在沒有 `ft` / `sc` 的 job
- 指定的 checkpoint 或模型資料夾不存在
- `dp` 需要的 CSV 池是空的

### 5.3 `dp.`（前處理）

| 參數 | 預設 | 用途 |
|---|---|---|
| `dp.exclude_verdicts` | `confirmed_attack,rule_match` | 要丟掉哪些 `ML Verdict` 的資料列。`none` 表示全部保留 |
| `dp.exclude_src_ips` | 空 | 來源 IP 落在這些 IP 或 CIDR 內的資料列都丟掉，見下方說明 |
| `dp.exclude_dst_ips` | 空 | 目的 IP 落在這些 IP 或 CIDR 內的資料列都丟掉，見下方說明 |
| `dp.clip_min` | `-1e9` | 數值下限 |
| `dp.clip_max` | `1e9` | 數值上限 |
| `dp.fill_value` | `0.0` | 缺值與無限大的填補值 |

`ML Verdict` 是 Mantis 線上偵測時對每筆 flow 的判定：

| 值 | 意思 |
|---|---|
| `normal` | 模型判定正常 |
| `anomaly` | 模型判定異常，沒有其他證據 |
| `rule_match` | 命中 Suricata 規則 |
| `structural` | 聚合器判定為 FLOOD / SCAN / TARGETED |
| `confirmed_attack` | 規則和聚合器都命中 |
| `suppressed` | 操作員確認為誤報 |
| `whitelisted` | 在 Mantis 白名單內：有評分，但一律不告警 |
| `uninspected` | 未經檢查 |

預設只丟有規則佐證的 `confirmed_attack` 和 `rule_match`。`structural` 保留，因為流量大的正常使用者也會觸發；`anomaly` 保留，否則每次重訓都會把正常流量的尾端修掉，模型越訓越窄。要拿資料跑 `sc` 和線上告警率比較時，前處理請用 `dp.exclude_verdicts=none`。沒有 `ML Verdict` 欄位的 CSV 不受影響。

**排除特定 IP**：`dp.exclude_src_ips=10.0.0.5,192.168.50.0/24` 會丟掉來源 IP 在清單內的資料列，等於把這些主機的整段序列拿掉。用途是「不讓模型把壞資料學成正常」，例如中毒主機、滲透測試機、掃描器。它**不是**用來讓吵雜的主機不再告警：模型沒看過的主機，上線後反而更常被判異常；那種情況請在 Mantis 用 Suppression rule。

`dp.exclude_dst_ips` 依目的 IP 排除，會連帶拿掉其他主機連到這些位址的流量，讓它們的序列中間缺幾筆，但 Mantis 線上推論時看到的是完整序列，所以優先用 `exclude_src_ips`。格式錯誤的位址在啟動前就會被拒絕（422）。前處理日誌會列出每條規則排除了幾列、幾個位址，一列都沒排到時會警告。

**先看報告再決定**：設定 `dp.exclude_verdicts` 或 `dp.exclude_src_ips` 之前，可以先跑 `./main.py --verdict-report`（或在 Web UI 的 Trainer 頁面 Datasets 分頁按 Analyze，API 是 `GET /api/v1/datasets/verdicts`）。它只讀每個 CSV 的 `ML Verdict`、`Source IP`、`Timestamp` 三欄，不改任何東西，也不需要訓練用的套件。報告會列出：

- 各 verdict 的數量和比例（不認得的值會標出來，沒有 `ML Verdict` 欄位的檔案會列出並略過）
- 觸發 verdict（預設 `rule_match,confirmed_attack`，跟 `dp.exclude_verdicts` 的預設一樣）總共幾列
- 每個至少有一筆觸發列的來源 IP：觸發列數、總列數、各 verdict 分佈、是否為私有位址、第一次和最後一次觸發的時間，以及排除它會連帶刪掉多少列（整份資料，以及只算每筆觸發列前後 `--window-secs` 秒內）。`suppressed` 的列不算，因為那是操作員確認過的正常流量。

公有位址、列數少的來源，多半是攻擊者，排除的代價很小；私有位址、佔比大的來源是你自己的主機，整台排除會連它的正常行為一起拿掉。參數：`--trigger-verdicts`、`--window-secs`（預設 600）、`--top`（預設 50）、`--json`。

### 5.4 `da.`（訓練、finetune、分析）

**訓練**

| 參數 | 預設 | 用途 |
|---|---|---|
| `da.epochs` | `1000` | 最多訓練幾輪；通常會提早停止 |
| `da.batch_size` | `8192` | 訓練批次大小。GPU 記憶體不足時調小 |
| `da.inference_batch_size` | `1024` | 推論批次大小 |
| `da.learning_rate` | `0.001` | 學習率 |
| `da.weight_decay` | `1e-5` | AdamW 權重衰減 |
| `da.clipnorm` | `1.0` | 梯度裁剪 |
| `da.early_stopping_patience` | `5` | 驗證損失幾輪沒進步就停止 |
| `da.early_stopping_min_delta` | `1e-5` | 算「有進步」的最小下降量；也是降低學習率的判斷門檻 |
| `da.reduce_lr_patience` | `3` | 幾輪沒進步就降低學習率 |
| `da.reduce_lr_factor` | `0.5` | 每次降低的倍數 |
| `da.min_lr` | `1e-7` | 學習率下限 |
| `da.finetune_lr_scale` | `0.1` | `ft` 時學習率乘上的倍數 |

**模型結構**

| 參數 | 預設 | 用途 |
|---|---|---|
| `da.window_size` | `15` | 幾筆 flow 組成一個序列 |
| `da.stride` | `1` | 視窗每次移動幾筆 |
| `da.hidden_size` | `128` | LSTM 隱藏層大小 |
| `da.num_layers` | `4` | LSTM 層數 |
| `da.encoding_dim` | `64` | 瓶頸層維度 |
| `da.dropout` | `0.2` | Dropout 比例 |
| `da.latent_norm_weight` | `1e-3` | 把正常流量的潛在向量往原點壓縮的權重 |

`ft` 會沿用原模型的結構和前處理設定，改結構參數對 `ft` 無效。

**資料切分與增強**

| 參數 | 預設 | 用途 |
|---|---|---|
| `da.validation_split` | `0.10` | 驗證集比例 |
| `da.test_split` | `0.15` | 測試集比例 |
| `da.split_random_state` | `42` | 切分亂數種子 |
| `da.synthetic_augmentation_ratio` | `0.2` | 額外合成多少比例的訓練序列；`0` 關閉 |

**縮放與裁剪**

| 參數 | 預設 | 用途 |
|---|---|---|
| `da.winsorize_lower` / `da.winsorize_upper` | `0.005` / `0.995` | 依分位數截斷極端值 |
| `da.max_bound_z` | `4.0` | 截斷邊界距離平均值最多幾個標準差；`none` 關閉 |
| `da.tail_check_z` | `4.0` | 邊界超過幾個標準差的特徵要額外檢查誤報 |
| `da.clip_min` / `da.clip_max` | `-5.0` / `5.0` | 縮放後的裁剪範圍 |
| `da.fill_value` | `0.0` | 缺值填補 |

**分析（`fa`、`sc`，以及 `da` / `ft` 結束時的報告）**

| 參數 | 預設 | 用途 |
|---|---|---|
| `da.analysis_threshold` | `p99` | 報告使用的門檻。可用值：`p90`～`p99`、`p99.5`、`p99.9`、`mean+1std`、`mean+2std`、`Q3+1.5IQR`、`Q3+3.0IQR` |
| `da.analysis_top_hosts` | `3` | 列出誤報最多的前幾個 IP |

### 5.5 `ex.`（匯出）

| 參數 | 預設 | 用途 |
|---|---|---|
| `ex.opset_version` | `18` | ONNX opset 版本 |
| `ex.post_scaling_clip_min` / `ex.post_scaling_clip_max` | `-6.0` / `6.0` | 推論設定中的縮放後裁剪範圍 |
| `ex.verify_atol` | `1e-3` | 驗證 ONNX 與 PyTorch 輸出差異的容許值 |
| `ex.version` | `1.0.0` | 模型版本字串，寫入 `exports/full_config.json` 的 `version` 欄位 |

### 5.6 `run.`（執行選項）

| 參數 | 預設 | 用途 | 只能搭配 |
|---|---|---|---|
| `run.resume` | 無 | 從 checkpoint 接續訓練，例如 `run.resume=autoencoder_temp-v14.ckpt`；只寫檔名時會先到 `./artifacts` 找 | `da` |
| `run.model_dir` | `./artifacts` | `ft` / `sc` 使用的模型資料夾，裡面要有 `deep_autoencoder.pt` 和 `deep_ae_config.pkl`；只寫名稱時會先到 `./artifacts` 找，例如 `run.model_dir=pretrained_20261110_0412` | `ft`、`sc` |

透過 API 時，這兩個路徑必須位於節點的 workspace 內，否則回 400。

---

## 6. 用 CLI 操作

```
./main.py [-a] [-dp] [-da] [-ft] [-sc] [-fa] [-ex] [--args "區段.名稱=值 ..."]
```

旗標只用來選任務，參數一律放在 `--args`。範例：

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

CLI 在前景執行，日誌同時印在終端機和 `logs/train.log`。參數有錯時會印出原因並結束，不會開始執行。

---

## 7. 用 API 操作

所有路徑的前綴都是 `/api/v1`。以下用 `node-a:8000` 代表節點位址。

### 7.1 節點狀態

```bash
curl http://node-a:8000/api/v1/health
```

```bash
curl http://node-a:8000/api/v1/node
```

`/health` 用於存活檢查，不需要 token。`/node` 回傳節點名稱、主機名、是否忙碌（`busy`）、目前的 job，以及支援的任務。

### 7.2 效能監控

一次性查詢：

```bash
curl http://node-a:8000/api/v1/metrics
```

`/metrics` 回傳 CPU、RAM、GPU 全部；也可以只查 `/metrics/cpu`、`/metrics/memory`、`/metrics/gpu`。

- CPU：實體 / 邏輯核心數、整體與每核心使用率、頻率、load average
- RAM：總量、可用、已用、使用率、swap
- GPU：每張卡的名稱、使用率、顯示記憶體、溫度、功耗。沒有 NVIDIA GPU 時 `available` 為 `false`

即時推送（WebSocket）：

```
ws://node-a:8000/api/v1/metrics/ws?interval=1
```

每 `interval` 秒推送一則 JSON，格式與 `GET /metrics` 相同。`interval` 必須在 0.5～60 秒之間，超出範圍會拒絕連線。網址要加上 `&token=<token>`。

`/node/ws?interval=5` 用一條連線推送 `node`、`metrics` 和 `jobs`（`jobs` 和上一則相同時為 `null`）；Mantis 網頁用它取代輪詢 `/node`、`/metrics`、`/jobs`。

### 7.3 資料集

見 [3. 準備資料](#3-準備資料)。

### 7.4 啟動 job

```bash
curl -X POST http://node-a:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"stages": ["all", "fa"], "args": "da.epochs=50 da.analysis_threshold=p95"}'
```

- `stages`：任務代號陣列，見 [4. 任務](#4-任務)
- `args`：參數字串，見 [5. 參數](#5-參數)，可以省略

成功回 202，內容包含 `job_id`、狀態、實際執行的任務，以及 `dp` 讀取的 CSV 清單。

一個節點同時只跑一個 job；忙碌時回 409。job 在獨立子程序執行，執行中 API 仍可正常回應。

查詢所有參數的預設值：

```bash
curl http://node-a:8000/api/v1/jobs/params
```

只檢查請求、不啟動 job（內容和 `POST /jobs` 相同）：

```bash
curl -X POST http://node-a:8000/api/v1/jobs/check \
  -H "Content-Type: application/json" \
  -d '{"stages": ["ft"], "args": "run.model_dir=./artifacts/pretrained_20261110_0412"}'
```

套用和 `POST /jobs` 完全相同的規則（任務組合、`args` 的參數與型別、`dp` 需要的 CSV、`run.model_dir` / `run.resume` 路徑）。通過時回 200，內容是會執行的任務、`dp` 會讀取的 CSV，以及解析後的 `model_dir` / `resume`；不通過時回和啟動時相同的 422 / 400 錯誤。不檢查節點是否忙碌。

### 7.5 查詢與取消 job

```bash
curl http://node-a:8000/api/v1/jobs
```

```bash
curl http://node-a:8000/api/v1/jobs/<job_id>
```

```bash
curl -X DELETE http://node-a:8000/api/v1/jobs/<job_id>
```

第一行列出 job 列表（最多保留 200 筆），第二行查單一 job，第三行取消 job。

job 狀態：

| 狀態 | 意思 |
|---|---|
| `pending` | 已建立，尚未啟動 |
| `running` | 執行中 |
| `succeeded` | 成功（`exit_code` 為 0） |
| `failed` | 失敗，原因在 job 日誌 |
| `cancelled` | 已取消 |

每個 job 也帶有 `stage_progress`：依執行順序，每個任務一筆，含 `status` 與 `started_at` / `finished_at`（unix 秒）；`current_stage` 是目前執行中的任務（沒有時為 `null`）：

| 任務狀態 | 意思 |
|---|---|
| `pending` | 還沒輪到 |
| `running` | 執行中 |
| `succeeded` | 已完成 |
| `failed` / `cancelled` | job 失敗或被取消時正在執行的任務 |
| `skipped` | job 提早結束，沒有執行 |

worker 在每個任務開始與結束時，把時間寫到 `logs/jobs/<job_id>.stages.json`。

取消時會停止 job 程序和它的所有子程序（例如 DataLoader worker）。取消後 `cancel_requested` 立即變成 `true`，等程序真正結束後狀態才會變成 `cancelled`，在那之前節點仍是忙碌狀態。被取消的 job 日誌最後常出現 `DataLoader worker ... killed by signal`，這是正常現象。

job 紀錄存在 `logs/jobs/<job_id>.job.json`，API 伺服器重啟後仍可查詢。伺服器關閉時會停止執行中的 job；重啟後，這些 job 會標成 `failed`，節點恢復成可接新工作的狀態。

### 7.6 日誌

job 日誌：

```bash
curl "http://node-a:8000/api/v1/jobs/<job_id>/logs?lines=200"
```

```bash
curl -N http://node-a:8000/api/v1/jobs/<job_id>/logs/stream
```

第一行取最後 N 行（1～10000），第二行即時追蹤（Server-Sent Events）。即時追蹤先送出最後 50 行（可用 `?lines=` 調整），之後有新日誌就推送，job 結束時送出 `event: end`。

所有日誌檔：

```bash
curl http://node-a:8000/api/v1/logs
```

```bash
curl "http://node-a:8000/api/v1/logs/train.log?lines=200"
```

`train.log` 是整台節點的總日誌（每天輪替，保留 30 天），`jobs/<job_id>.log` 是單一 job 的日誌。

### 7.7 下載結果

類別：`exports`、`artifacts`、`plots`、`outputs`、`logs`。

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

依序是：列出檔案、下載單一檔案、把整個類別打包成 zip、把類別內的某個子資料夾打包成 zip。zip 檔名會帶節點名稱（例如 `node-a-exports.zip`），多台節點的結果不會撞名。

---

## 8. 用 Docker 操作

### 建置 image

```bash
docker build -t mantis-trainer .
```

- 基底是 `ubuntu:24.04`，不是 `nvidia/cuda`：pip 裝的 CUDA 版 PyTorch 自帶 CUDA / cuDNN 函式庫（約 3.3 GiB），舊的 `nvidia/cuda:*-cudnn-runtime` 基底（壓縮後 2.7 GiB）又放了一份，等於重複。GPU 驅動一樣由主機的 NVIDIA Container Toolkit 提供，`--gpus all` 用法不變。
- 套件和程式碼分層：只改程式碼時，重新 build 和 `docker pull` 都只傳最後一層。PyTorch 由 Lightning 依賴自動安裝，只裝一次；以前先從 PyTorch 的 index 裝最新版、`requirements.txt` 又限制版本，torch 和整套 NVIDIA 函式庫會被下載兩次。
- `.dockerignore` 排除 `src/datasets`、`src/outputs`、`src/logs`、所有 `*.csv` / `*.parquet`：在已經有資料的節點上 build，資料不會被包進 image。
- CI 推到 GHCR 時用 zstd 壓縮（`oci-mediatypes`），拉取需要 Docker Engine 23 以上。

### 批次執行（CLI 模式）

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

| 變數 | 用途 |
|---|---|
| `ALL` / `DATAPREPROCESS` / `DEEPAUTOENCODER` / `FINETUNE` / `SCORE` / `FEATUREANALYSIS` / `EXPORT` | 設成 `true` 就執行對應任務 |
| `ARGS` | 參數字串，寫法同 `--args` |
| `DATAPATH` | CSV 池位置（容器內路徑），會設定 `DATASET_ROOT` |

`ft` / `sc` 要用哪個模型，請在 `ARGS` 裡寫 `run.model_dir=...`。

### 伺服器模式（API）

```bash
docker run -d --ipc=host --gpus all -p 8000:8000 \
  -v ./src/datasets:/app/src/datasets \
  -v ./src/outputs:/app/src/outputs \
  -v ./src/artifacts:/app/src/artifacts \
  -v ./src/plots:/app/src/plots \
  -v ./src/exports:/app/src/exports \
  -v ./src/logs:/app/src/logs \
  -e API=true -e NODE_ID=node-a -e API_TOKEN=請換成隨機字串 \
  ghcr.io/parrotxray/mantis-trainer:main
```

`API=true` 時會忽略任務變數和 `ARGS`，改由 API 送出 job。其他 API 環境變數見 [2. 安裝與啟動](#啟動-api)。

---

## 9. 多節點部署

每台 GPU 主機各跑一個 API 節點，各自有獨立的 CSV 池、模型和輸出，彼此不共享。

1. 每台主機用不同的 `NODE_ID` 啟動 API。
2. 把要訓練的 CSV 上傳到要用的節點。
3. 派工前查詢各節點的 `GET /node`，挑 `busy` 為 `false` 的節點；可同時參考 `GET /metrics` 的 GPU 使用率與記憶體。
4. 送出 job。如果節點剛好被佔用，會回 409，換下一台即可。
5. 用 `GET /jobs/<id>` 輪詢或用 `/logs/stream` 追蹤，完成後從 `/files/exports/archive` 取回結果。

每個節點都必須設定 `API_TOKEN`，可以相同也可以各自不同。如果有瀏覽器前端要直接連多台節點，每個節點都要在 `CORS_ORIGINS` 加上前端的網址。

---

## 10. 常見流程

### 第一次訓練

1. 把 CSV 放進 CSV 池（複製到 `src/datasets/` 或用 API 上傳）。
2. 執行完整流程，順便做特徵分析：

```bash
./main.py -a -fa
```

3. 看 `logs/train.log` 裡的 `Threshold Analysis [Val Set — BENIGN only]` 表格，依可接受的誤報率選定門檻。
4. 從 `exports/` 取出 `deep_autoencoder.onnx` 和 `inference_config.json` 部署到 Mantis，並在 Mantis 的設定中填入選定的門檻。

### 中斷後接續訓練

```bash
./main.py -da -ex --args "run.resume=autoencoder_temp.ckpt"
```

checkpoint 檔名請看 `artifacts/` 裡實際存在的 `.ckpt`。

### 用新資料 finetune

1. 清空 CSV 池，放入新資料。
2. 依序執行前處理、finetune、匯出：

```bash
./main.py -dp
```

```bash
./main.py -ft -ex --args "da.epochs=20"
```

原模型會備份到 `artifacts/pretrained_<時間>/`。

### 比較模型與線上告警率

1. 準備一個獨立的 workspace（例如另一個資料夾），放入要評分那段期間的 CSV，並用 `dp.exclude_verdicts=none` 前處理，保留所有資料列。
2. 用要比較的模型評分：

```bash
./main.py -sc --args "run.model_dir=/path/to/artifacts da.analysis_threshold=p99"
```

3. 報告會依 `ML Verdict` 分組，列出各組超過門檻的比例。比較 finetune 前後的模型時，`suppressed` 的比例應該下降，`normal` 的比例不應上升。

### 調查誤報

```bash
./main.py -fa --args "da.analysis_threshold=p95 da.analysis_top_hosts=5"
```

報告會列出正常流量中誤報最多的特徵，以及誤報最多的幾個 IP 每天的誤報率和相關目的地。

### 用 API 跑完整流程並取回結果

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

## 11. 輸出檔案

| 位置 | 內容 | 產生者 |
|---|---|---|
| `datasets/` | CSV 池 | 使用者 |
| `outputs/preprocessing_benign.*`、`preprocessing_attack.*` | 前處理結果（CSV 與 parquet） | `dp` |
| `outputs/deep_ae_scores.csv` | 測試集分數 | `da`、`ft` |
| `artifacts/deep_autoencoder.pt` | PyTorch 模型 | `da`、`ft` |
| `artifacts/deep_ae_config.pkl` | 前處理參數、門檻（`ae_thresholds`）等設定 | `da`、`ft` |
| `artifacts/*.ckpt` | 訓練 checkpoint | `da`、`ft` |
| `artifacts/pretrained_<時間>/` | finetune 前的原模型備份 | `ft` |
| `plots/*.png` | 分數分佈、ROC/PR、CDF、t-SNE 圖 | `da`、`ft` |
| `exports/deep_autoencoder.onnx` | 部署用模型 | `ex` |
| `exports/inference_config.json` | 部署用推論設定 | `ex` |
| `exports/full_config.json` | 完整設定，含 `version` | `ex` |
| `logs/train.log` | 總日誌 | 全部 |
| `logs/jobs/<job_id>.log` | 單一 job 日誌 | API |
| `logs/jobs/<job_id>.job.json` | job 紀錄（狀態、任務、參數、時間） | API |
| `logs/jobs/<job_id>.stages.json` | 每個任務的開始與結束時間 | API |

---

## 12. 錯誤與排除

### API 狀態碼

| 狀態碼 | 意思 | 常見原因 |
|---|---|---|
| 401 | 未授權 | 請求沒帶 token 或 token 錯誤 |
| 400 | 名稱或路徑不合法 | 上傳非 `.csv`、檔名重複、路徑跑出 workspace（`../`） |
| 404 | 找不到 | job、檔案、日誌不存在 |
| 409 | 衝突 | 節點正在跑其他 job、`dp` 執行中修改 CSV 池、取消已結束的 job |
| 413 | 檔案太大 | 超過 `MAX_UPLOAD_MB` |
| 422 | 參數或任務組合錯誤 | 見下表 |

### 參數錯誤訊息

| 訊息 | 處理方式 |
|---|---|
| `Parameter 'epochs' needs a prefix, use da.epochs` | 加上前綴 |
| `Unknown prefix 'zz' ...` | 前綴只能是 `dp`、`da`、`ex`、`run` |
| `Unknown parameter 'da.xxx'` | 名稱拼錯，或該參數不屬於這個區段 |
| `Invalid value for DeepAutoencoderConfig.epochs: 'abc'` | 值的型別不對 |
| `Duplicate parameter 'da.epochs'` | 同一個參數寫了兩次 |
| `Expected section.key=value, got '...'` | 少了 `=` |
| `Invalid args: No closing quotation` | 引號沒有成對 |
| `Stage 'ft' replaces 'da' training ...` | `ft` 和 `da` 擇一 |
| `Stage 'sc' reads the preprocessed parquet ...` | 先單獨跑 `dp`，再跑 `sc` |
| `'run.resume' only applies to stage 'da'` | 拿掉 `run.resume`，或加上 `da` |
| `'run.model_dir' only applies to stages 'ft' and 'sc'` | 拿掉 `run.model_dir`，或改跑 `ft` / `sc` |
| `Model directory '...' not found` | 確認模型資料夾存在 |
| `Checkpoint '...' not found` | 確認 checkpoint 路徑 |
| `Stage 'dp' requires CSV files in ...` | 先匯入 CSV |

### 執行時的問題

| 現象 | 處理方式 |
|---|---|
| `CUDA out of memory` | 調小 `da.batch_size` 或 `da.inference_batch_size` |
| job 結束碼 `-9`（沒有按取消） | 系統記憶體不足，被 OOM killer 砍掉（`dmesg \| grep -i oom` 可確認）。`da` 建的訓練 window 約是 `資料筆數 × window_size × 特徵數 × 4 bytes`：調大 `da.stride`、調小 `da.synthetic_augmentation_ratio`，或一次少訓練幾天的資料。swap 只能當保險，訓練資料超出實體記憶體時會非常慢 |
| `da` 找不到 `outputs/preprocessing_benign.parquet` | 先跑 `dp` |
| `ft` / `sc` / `fa` / `ex` 找不到模型 | 先跑 `da`，或用 `run.model_dir` 指定模型（`fa`、`ex` 只讀 `artifacts/`） |
| `sc` 報 `Unknown threshold` | `da.analysis_threshold` 必須是模型存下的門檻名稱之一，見 5.4 節的分析參數 |
| WebSocket 連不上 | 確認有安裝 `uvicorn[standard]`（內含 `websockets`）、`interval` 在允許範圍內，網址有帶 `?token=` |
| 瀏覽器顯示 CORS 錯誤 | 在節點的 `CORS_ORIGINS` 加上前端網址，並重啟 API |
| job 一直是 `running` | 用 `/jobs/<id>/logs/stream` 看進度；必要時 `DELETE /jobs/<id>` 取消 |

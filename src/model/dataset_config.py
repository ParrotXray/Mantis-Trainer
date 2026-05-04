from dataclasses import dataclass
from typing import Dict, Final, List

import kagglehub
import pandas as pd

from utils import Logger

logger = Logger(__name__)


# ---------------------------------------------------------------------------
# Unified feature schema
# ---------------------------------------------------------------------------

SEQUENCE_META_COLUMNS: Final[List[str]] = ["timestamp", "src_ip"]

UNIFIED_FEATURE_NAMES: Final[List[str]] = [
    # ── Original 27 features ──────────────────────────────────────────────
    "flow_duration",
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_win_bytes",
    "bwd_win_bytes",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "flow_iat_mean",
    "pkt_len_mean",
    "dst_port",
    "protocol",
    "psh_flag_cnt",
    "ack_flag_cnt",
    "syn_flag_cnt",
    "fin_flag_cnt",
    "rst_flag_cnt",
    "pkt_len_std",
    "fwd_pkt_len_std",
    "bwd_pkt_len_std",
    "fwd_seg_size_min",
    "fwd_act_data_pkts",
    # ── Added features (27 → 38) ──────────────────────────────────────────
    # Flow IAT spread — slow/low-and-slow attack discrimination
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    # Active / Idle timing — behavioural rhythm, high SHAP for CIC-IDS2017 AE
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
    # Down/Up ratio — scan/recon pattern discrimination
    "down_up_ratio",
]


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    name: str
    label_column: str
    benign_labels: List[str]
    label_mapping: Dict[str, str]
    column_mapping: Dict[str, str]
    kaggle_dataset_id: str = ""
    csv_glob: str = "*.csv"
    header_file: str = ""
    header_name_column: str = ""

    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        rename_map = {}
        for src_col, unified_name in self.column_mapping.items():
            if src_col in df.columns:
                rename_map[src_col] = unified_name
        df = df.rename(columns=rename_map)
        return df

    def compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def extract_labels(self, df: pd.DataFrame) -> pd.Series:
        if self.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found. "
                f"Available: {list(df.columns)}"
            )
        labels = df[self.label_column].astype(str).str.strip()
        labels = labels.str.replace("\ufffd", "-", regex=False)
        labels = labels.replace(self.label_mapping)
        return labels


# ---------------------------------------------------------------------------
# Shared extended column mapping (covers both 2017 and 2018 naming)
# ---------------------------------------------------------------------------

_EXTENDED_COLUMN_MAPPING: Final[Dict[str, str]] = {
    # ── Original features (2017 naming) ───────────────────────────────────
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_pkts_per_sec",
    "Init_Win_bytes_forward": "fwd_win_bytes",
    "Init_Win_bytes_backward": "bwd_win_bytes",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Flow IAT Mean": "flow_iat_mean",
    "Packet Length Mean": "pkt_len_mean",
    "Destination Port": "dst_port",
    "Protocol": "protocol",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "FIN Flag Count": "fin_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "Packet Length Std": "pkt_len_std",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "min_seg_size_forward": "fwd_seg_size_min",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    # ── New features ──────────────────────────────────────────────────────
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
    "Down/Up Ratio": "down_up_ratio",
    # ── 2018 fallback naming ───────────────────────────────────────────────
    "Dst Port": "dst_port",
    "Tot Fwd Pkts": "fwd_packets",
    "Tot Bwd Pkts": "bwd_packets",
    "TotLen Fwd Pkts": "fwd_bytes",
    "TotLen Bwd Pkts": "bwd_bytes",
    "Flow Byts/s": "flow_bytes_per_sec",
    "Init Fwd Win Byts": "fwd_win_bytes",
    "Init Bwd Win Byts": "bwd_win_bytes",
    "Fwd Pkt Len Mean": "fwd_pkt_len_mean",
    "Bwd Pkt Len Mean": "bwd_pkt_len_mean",
    "Pkt Len Mean": "pkt_len_mean",
    "PSH Flag Cnt": "psh_flag_cnt",
    "ACK Flag Cnt": "ack_flag_cnt",
    "SYN Flag Cnt": "syn_flag_cnt",
    "FIN Flag Cnt": "fin_flag_cnt",
    "RST Flag Cnt": "rst_flag_cnt",
    "Pkt Len Std": "pkt_len_std",
    "Fwd Pkt Len Std": "fwd_pkt_len_std",
    "Bwd Pkt Len Std": "bwd_pkt_len_std",
    "Fwd Seg Size Min": "fwd_seg_size_min",
    "Fwd Act Data Pkts": "fwd_act_data_pkts",
    # ── Sequence metadata ─────────────────────────────────────────────────
    "Timestamp": "timestamp",
    "Src IP": "src_ip",
    "Source IP": "src_ip",
}


# ---------------------------------------------------------------------------
# CIC-UNSW-NB15
# ---------------------------------------------------------------------------

CIC_UNSW_NB15_CONFIG: Final[DatasetConfig] = DatasetConfig(
    name="unsw",
    kaggle_dataset_id="yasserhessein/cic-unsw-nb15-augmented-dataset/versions/1",
    csv_glob="CICFlowMeter.csv",
    label_column="Label",
    benign_labels=["Normal"],
    label_mapping={
        "Benign": "Normal",
        "BENIGN": "Normal",
        "Normal": "Normal",
        "": "Normal",
        "nan": "Normal",
        "DoS": "DoS",
        "Dos": "DoS",
        "DOS": "DoS",
        "Exploits": "Exploitation",
        "Shellcode": "Exploitation",
        "Worms": "Exploitation",
        "Backdoor": "Exploitation",
        "Backdoors": "Exploitation",
        "Reconnaissance": "Reconnaissance",
        "Generic": "Reconnaissance",
        "Fuzzers": "Reconnaissance",
        "Analysis": "Reconnaissance",
    },
    column_mapping=_EXTENDED_COLUMN_MAPPING,
)

# ---------------------------------------------------------------------------
# LAB-301
# ---------------------------------------------------------------------------

LAB_301_CONFIG: Final[DatasetConfig] = DatasetConfig(
    name="lab301",
    kaggle_dataset_id="ruiluncai/lab301-timestamp-benign-dataset/versions/1",
    label_column="Label",
    benign_labels=["Normal"],
    label_mapping={
        "BENIGN": "Normal",
        "Benign": "Normal",
        "DoS GoldenEye": "DoS",
        "DoS Hulk": "DoS",
        "DoS Slowhttptest": "DoS",
        "DoS slowloris": "DoS",
        "DDoS": "DDoS",
        "FTP-Patator": "Brute Force",
        "SSH-Patator": "Brute Force",
        "Web Attack \u2013 Brute Force": "Brute Force",
        "Web Attack - Brute Force": "Brute Force",
        "Web Attack \u2013 Sql Injection": "Reconnaissance",
        "Web Attack \u2013 XSS": "Reconnaissance",
        "Web Attack - Sql Injection": "Reconnaissance",
        "Web Attack - XSS": "Reconnaissance",
        "Infiltration": "Exploitation",
        "Heartbleed": "Exploitation",
        "PortScan": "Reconnaissance",
        "Bot": "Exploitation",
    },
    column_mapping=_EXTENDED_COLUMN_MAPPING,
)

# ---------------------------------------------------------------------------
# CIC-IDS-test
# ---------------------------------------------------------------------------

CIC_TEST_CONFIG: Final[DatasetConfig] = DatasetConfig(
    name="test",
    kaggle_dataset_id="ruiluncai/attack-test-dataset/versions/1",
    label_column="Label",
    benign_labels=["Normal"],
    label_mapping={
        "BENIGN": "Normal",
        "Benign": "Normal",
        "DoS GoldenEye": "DoS",
        "DoS Hulk": "DoS",
        "DoS Slowhttptest": "DoS",
        "DoS slowloris": "DoS",
        "DoS-slowhttp": "DoS",
        "DoS-hulk": "DoS",
        "DDoS": "DDoS",
        "DDoS-bot": "DDoS",
        "DDoS-stomp": "DDoS",
        "DDoS-dyn": "DDoS",
        "DDoS-tcp": "DDoS",
        "FTP-Patator": "Brute Force",
        "SSH-Patator": "Brute Force",
        "Web Attack \u2013 Brute Force": "Brute Force",
        "Web Attack - Brute Force": "Brute Force",
        "Web Attack \u2013 Sql Injection": "Reconnaissance",
        "Web Attack \u2013 XSS": "Reconnaissance",
        "Web Attack - Sql Injection": "Reconnaissance",
        "Web Attack - XSS": "Reconnaissance",
        "Web-sql-injection": "Reconnaissance",
        "Web-xss": "Reconnaissance",
        "Web-command-injection": "Reconnaissance",
        "Infiltration": "Exploitation",
        "Heartbleed": "Exploitation",
        "Infiltration-mitm": "Exploitation",
        "PortScan": "Reconnaissance",
        "Bot": "Exploitation",
        "slowheaders": "DoS",
        "ddossim": "DDoS",
        "slowread": "DoS",
        "slowloris": "DoS",
        "hulk": "DoS",
        "slowbody2": "DoS",
        "rudy": "DoS",
        "goldeneye": "DoS",
    },
    column_mapping=_EXTENDED_COLUMN_MAPPING,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: Final[Dict[str, DatasetConfig]] = {
    "unsw": CIC_UNSW_NB15_CONFIG,
    "lab301": LAB_301_CONFIG,
    "test": CIC_TEST_CONFIG,
}


def get_dataset_config(name: str) -> DatasetConfig:
    key = name.strip().lower()
    if key not in _DATASET_REGISTRY:
        available = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATASET_REGISTRY[key]


def list_available_datasets() -> List[str]:
    return list(_DATASET_REGISTRY.keys())


def download_dataset(config: DatasetConfig) -> str:
    if not config.kaggle_dataset_id:
        raise ValueError(
            f"Dataset '{config.name}' has no kaggle_dataset_id configured. "
            f"Provide a --path manually."
        )
    logger.info(f"Downloading '{config.kaggle_dataset_id}' via kagglehub...")
    path = kagglehub.dataset_download(config.kaggle_dataset_id)
    logger.info(f"Dataset ready: {path}")
    return path

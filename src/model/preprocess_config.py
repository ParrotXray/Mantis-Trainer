from dataclasses import dataclass
from typing import Tuple


@dataclass
class PreprocessConfig:
    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0
    # Rows whose Mantis "ML Verdict" is one of these are dropped at load time.
    # By default only anomalies backed by a Suricata rule match:
    # confirmed_attack (rule match plus an aggregator FLOOD / SCAN / TARGETED
    # verdict) and rule_match (rule match alone). "structural" (aggregator
    # verdict alone) is kept: those heuristics also flag a busy client's
    # ordinary egress. Dropping plain "anomaly" would trim the benign tail on
    # every retrain, since the live threshold flags a fixed share of benign
    # traffic by definition.
    exclude_verdicts: Tuple[str, ...] = ("confirmed_attack", "rule_match")

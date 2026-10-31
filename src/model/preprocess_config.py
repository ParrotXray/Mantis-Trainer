from dataclasses import dataclass
from typing import Tuple


@dataclass
class PreprocessConfig:
    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0
    # Rows whose Mantis "ML Verdict" is one of these are dropped at load time.
    # Only confirmed_attack by default: it has evidence beyond the model (a
    # Suricata match or a FLOOD / SCAN / TARGETED verdict). Dropping plain
    # "anomaly" too would trim the benign tail on every retrain, since the
    # live threshold flags a fixed share of benign traffic by definition.
    exclude_verdicts: Tuple[str, ...] = ("confirmed_attack",)

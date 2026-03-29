from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    # Data Cleaning Parameters
    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0

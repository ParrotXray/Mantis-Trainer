from dataclasses import dataclass


@dataclass
class ExportConfig:
    version: str = "1.0.0"
    opset_version: int = 18
    post_scaling_clip_min: float = -5.0
    post_scaling_clip_max: float = 5.0
    verify_atol: float = 1e-3

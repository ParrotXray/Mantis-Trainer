from dataclasses import dataclass
from typing import List, Optional

from model import DeepAutoencoderConfig, ExportConfig, PreprocessConfig

from .stage import Stage


@dataclass(frozen=True)
class PipelineOptions:
    stages: List[Stage]
    data_root: str
    data_files: List[str]
    model_dir: str
    resume: Optional[str]
    preprocess: PreprocessConfig
    autoencoder: DeepAutoencoderConfig
    export: ExportConfig

from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepAutoencoderConfig:
    # Data Preprocessing Parameters
    clip_min: float = -5.0
    clip_max: float = 5.0
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
    fill_value: float = 0.0

    # Autoencoder Architecture Parameters
    # Input: 27 unified features → bottleneck: 16
    encoding_dim: int = 16
    layer_sizes: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    dropout_rates: List[float] = field(default_factory=lambda: [0.3, 0.25, 0.2, 0.1])
    l2_reg: float = 0.0001

    # Autoencoder Training Parameters
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    batch_size: int = 4096
    epochs: int = 350
    validation_split: float = 0.15
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    split_random_state: int = 42

    test_split: float = 0.20

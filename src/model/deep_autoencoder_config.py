from dataclasses import dataclass, field


@dataclass
class DeepAutoencoderConfig:
    # Data Preprocessing Parameters
    clip_min: float = -5.0
    clip_max: float = 5.0
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
    fill_value: float = 0.0

    # Sequence Parameters
    window_size: int = 10   # number of consecutive flows per sequence
    stride: int = 1         # sliding-window step (training)

    # LSTM Architecture Parameters
    hidden_size: int = 128
    num_layers: int = 2
    encoding_dim: int = 32  # bottleneck dimension
    dropout: float = 0.2

    # Training Parameters
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    batch_size: int = 512
    epochs: int = 100
    validation_split: float = 0.15
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    split_random_state: int = 42

    test_split: float = 0.20

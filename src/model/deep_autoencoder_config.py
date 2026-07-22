from dataclasses import dataclass


@dataclass
class DeepAutoencoderConfig:
    clip_min: float = -5.0
    clip_max: float = 5.0
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
    fill_value: float = 0.0

    window_size: int = 15
    stride: int = 1

    hidden_size: int = 128
    num_layers: int = 4
    encoding_dim: int = 64
    dropout: float = 0.2

    learning_rate: float = 0.001
    clipnorm: float = 1.0
    batch_size: int = 8192
    inference_batch_size: int = 1024
    epochs: int = 1000
    validation_split: float = 0.10
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    split_random_state: int = 42
    test_split: float = 0.15

    # Adds ||z||^2 to loss, compressing BENIGN latent vectors toward origin to widen the gap with attack flows.
    latent_norm_weight: float = 1e-3

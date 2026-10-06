from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DeepAutoencoderConfig:
    clip_min: float = -5.0
    clip_max: float = 5.0
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995

    # Features whose winsorized train values sit at the lower bound at least
    # this often (zero-inflated, e.g. rst/psh_flag_cnt) are min-max scaled to
    # [0, 1] instead of z-scored: z-scoring puts their rare non-zero value at
    # sqrt((1-p)/p) std, letting it dominate the reconstruction MSE.
    # None disables (all features z-scored).
    sparse_min_mass: Optional[float] = 0.9

    # Features checked after prediction for benign false positives on flows
    # whose window contains a non-zero value (raw > 0) of that feature.
    sparse_check_features: Tuple[str, ...] = ("rst_flag_cnt", "psh_flag_cnt")

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

    # Fraction of extra training sequences to synthesize by drawing window_size
    # flows at random from the whole scaled benign pool (src_ip/time ignored),
    # widening the combinations of interleaved flow types seen during
    # training beyond what naturally co-occurred in the captured order.
    # 0 disables it. Train-only — val/test stay untouched for honest eval.
    synthetic_augmentation_ratio: float = 0.2

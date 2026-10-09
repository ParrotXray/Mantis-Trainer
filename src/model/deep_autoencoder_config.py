from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepAutoencoderConfig:
    clip_min: float = -5.0
    clip_max: float = 5.0
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995

    # Cap on how many std a feature's winsorize bound may sit from its mean
    # after scaling: scale_ = max(std, |bound - mean| / max_bound_z). Features
    # whose tail is far out (e.g. zero-inflated rst_flag_cnt at z~11) would
    # otherwise dominate the reconstruction MSE and saturate the post-scaling
    # clip. Most flow features sit at z <= ~4. None disables.
    max_bound_z: Optional[float] = 4.0

    # Features whose winsorize bound sits beyond this many std (before any
    # max_bound_z cap) get a post-prediction check of benign FPR on windows
    # holding a tail value (past the train median) vs not. Kept separate from
    # max_bound_z so a max_bound_z=None baseline checks the same features.
    tail_check_z: float = 4.0

    # Val threshold (key of ae_thresholds) used by analyze_feature_errors to
    # split benign test windows into false positives vs true negatives.
    analysis_threshold: str = "p99"

    fill_value: float = 0.0

    window_size: int = 15
    stride: int = 1

    hidden_size: int = 128
    num_layers: int = 4
    encoding_dim: int = 64
    dropout: float = 0.2

    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    clipnorm: float = 1.0
    batch_size: int = 8192
    inference_batch_size: int = 1024
    epochs: int = 1000
    validation_split: float = 0.10
    early_stopping_patience: int = 5
    # Minimum val_loss drop that counts as an improvement for early stopping.
    # Must be meaningful against the loss scale (~5e-3): at 1e-6, ~1e-5-sized
    # gains at lr ~1e-5 kept resetting patience for 400+ epochs.
    early_stopping_min_delta: float = 1e-5
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

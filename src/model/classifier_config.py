from dataclasses import dataclass, field
from typing import List


@dataclass
class ClassifierConfig:
    # Data Split
    test_size: float = 0.2
    val_size: float = 0.15
    random_state: int = 42
    fill_value: float = 0.0

    # ResNet MLP Architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3

    # Training
    batch_size: int = 2048
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Class Weights
    use_class_weights: bool = True

    # Learning Rate Scheduler (ReduceLROnPlateau)
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 8
    min_lr: float = 1e-7

    # Early Stopping
    early_stopping_patience: int = 15

    # Gradient Clipping
    gradient_clip_val: float = 1.0

    # SMOTE+ENN Oversampling
    smote_ratio: float = 0.5
    smote_k_neighbors: int = 5
    smote_max_multiplier: int = 20
    enn_k_neighbors: int = 3

    # Hyperparameter Tuning
    enable_tuning: bool = False
    n_trials: int = 20
    tuning_metric: str = "f1_weighted"
    tuning_subsample: float = 0.2

    # Preprocessing
    clip_min: float = -5.0
    clip_max: float = 5.0

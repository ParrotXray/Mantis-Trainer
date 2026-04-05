import os
import warnings
from pathlib import Path
from typing import List, Optional

import joblib
import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from model import ClassifierConfig
from utils import Logger

# ---------------------------------------------------------------------------
# ResNet MLP building blocks
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Two-layer MLP block with BatchNorm, ReLU, Dropout and skip connection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.skip(x))


class ResNetMLP(nn.Module):
    """Stack of ResidualBlocks followed by a linear classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        n_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.dropout = dropout

        blocks = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(prev_dim, h_dim, dropout))
            prev_dim = h_dim
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.head(x)  # logits


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class ClassifierLightningModule(L.LightningModule):
    def __init__(
        self,
        model: ResNetMLP,
        class_weights: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 8,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr

        self._class_weights = class_weights
        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = (
            self._class_weights.to(logits.device)
            if self._class_weights is not None
            else None
        )
        return nn.functional.cross_entropy(logits, y, weight=weight)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Callback to record per-epoch losses for visualisation
# ---------------------------------------------------------------------------


class LossHistoryCallback(L.Callback):
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(float(train_loss))

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(float(val_loss))


# ---------------------------------------------------------------------------
# Classifier orchestrator (public API unchanged)
# ---------------------------------------------------------------------------


class Classifier:
    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        self.data: Optional[pd.DataFrame] = None

        self.label_encoder: Optional[LabelEncoder] = None

        self.features: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
        self.labels_encoded: Optional[np.ndarray] = None

        self.train_features: Optional[np.ndarray] = None
        self.val_features: Optional[np.ndarray] = None
        self.test_features: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.val_labels: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None

        self.train_features_balanced: Optional[np.ndarray] = None
        self.train_labels_balanced: Optional[np.ndarray] = None

        self.model: Optional[ResNetMLP] = None
        self.lightning_module: Optional[ClassifierLightningModule] = None
        self.loss_history: Optional[LossHistoryCallback] = None

        self.test_accuracy: Optional[float] = None
        self.predictions: Optional[np.ndarray] = None
        self.prediction_probs: Optional[np.ndarray] = None

        self.config: ClassifierConfig = config or ClassifierConfig()
        self.log: Logger = Logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for attr in [
            "train_features",
            "val_features",
            "test_features",
            "train_labels",
            "val_labels",
            "test_labels",
            "train_features_balanced",
            "train_labels_balanced",
            "prediction_probs",
            "_device",
        ]:
            setattr(self, attr, None)

        if self.lightning_module is not None:
            self.lightning_module.cpu()
            self.lightning_module = None
        if self.model is not None:
            self.model.cpu()
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        plt.close("all")
        return False

    # ------------------------------------------------------------------
    # Data loading & feature preparation
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        self.log.info("Loading data from outputs/deep_ae_scores.csv...")
        self.data = pd.read_csv("./outputs/deep_ae_scores.csv")
        self.data.columns = self.data.columns.str.strip()
        self.log.info(f"Total samples: {len(self.data):,}")

    def prepare_features(self) -> None:
        self.log.info("Preparing features...")

        feature_cols = [c for c in self.data.columns if c != "Label"]
        self.log.info(f"Using {len(feature_cols)} features")

        self.features = self.data[feature_cols].copy()
        self.labels = self.data["Label"].copy()

        self.features = self.features.replace([np.inf, -np.inf], np.nan)

        for col in self.features.columns:
            self.features[col] = pd.to_numeric(self.features[col], errors="coerce")

        self.features = self.features.fillna(0.0)

        # Remove rare classes
        min_samples = 50
        label_counts = self.labels.value_counts()
        rare_labels = label_counts[label_counts < min_samples].index.tolist()

        if rare_labels:
            self.log.warning(
                f"Removing {len(rare_labels)} classes with < {min_samples} samples:"
            )
            for label in rare_labels:
                self.log.warning(f"  - {label}: {label_counts[label]} samples")

            mask = ~self.labels.isin(rare_labels)
            self.features = self.features[mask].reset_index(drop=True)
            self.labels = self.labels[mask].reset_index(drop=True)

        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

        self.log.info(f"Feature dimensions: {self.features.shape}")
        self.log.info(f"Number of classes: {len(self.label_encoder.classes_)}")

        lines = ["\nClass distribution:", "=" * 60]
        for idx, label in enumerate(self.label_encoder.classes_):
            count = (self.labels_encoded == idx).sum()
            lines.append(f"  {idx:2d}. {label:<35} {count:>10,}")
        lines.append("=" * 60)
        self.log.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------

    def split_data(self) -> None:
        """3-way split: train / val / test."""
        self.log.info("Splitting data (train / val / test)...")

        features_array = self.features.values.astype(np.float32)

        trainval_features, self.test_features, trainval_labels, self.test_labels = (
            train_test_split(
                features_array,
                self.labels_encoded,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=self.labels_encoded,
            )
        )

        val_ratio = self.config.val_size
        self.train_features, self.val_features, self.train_labels, self.val_labels = (
            train_test_split(
                trainval_features,
                trainval_labels,
                test_size=val_ratio,
                random_state=self.config.random_state,
                stratify=trainval_labels,
            )
        )

        self.log.info(f"Training set:   {self.train_features.shape[0]:,}")
        self.log.info(f"Validation set: {self.val_features.shape[0]:,}")
        self.log.info(f"Test set:       {self.test_features.shape[0]:,}")

    # ------------------------------------------------------------------
    # Optuna hyperparameter tuning
    # ------------------------------------------------------------------

    def tune_hyperparameters(self) -> None:
        if not self.config.enable_tuning:
            self.log.info("Hyperparameter tuning disabled, using default parameters")
            return

        self.log.info(
            f"Starting Optuna hyperparameter tuning "
            f"({self.config.n_trials} trials)..."
        )

        tune_features, tune_labels = self._subsample_for_tuning()

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            n_layers = trial.suggest_int("n_layers", 2, 4)
            hidden_dims = []
            dim = trial.suggest_categorical("first_dim", [128, 256, 512])
            for i in range(n_layers):
                hidden_dims.append(dim)
                dim = max(32, dim // 2)

            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])

            # Split tuning data into train/val
            t_feat, v_feat, t_lab, v_lab = train_test_split(
                tune_features,
                tune_labels,
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=tune_labels,
            )

            n_classes = len(np.unique(tune_labels))
            input_dim = t_feat.shape[1]

            net = ResNetMLP(input_dim, hidden_dims, n_classes, dropout)

            # Class weights
            counts = np.bincount(t_lab, minlength=n_classes).astype(np.float32)
            weights = torch.tensor(1.0 / np.maximum(counts, 1.0), dtype=torch.float32)
            weights = weights / weights.sum() * n_classes

            module = ClassifierLightningModule(
                model=net,
                class_weights=weights,
                learning_rate=lr,
                weight_decay=weight_decay,
            )

            train_ds = TensorDataset(
                torch.tensor(t_feat, dtype=torch.float32),
                torch.tensor(t_lab, dtype=torch.long),
            )
            val_ds = TensorDataset(
                torch.tensor(v_feat, dtype=torch.float32),
                torch.tensor(v_lab, dtype=torch.long),
            )
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            trainer = L.Trainer(
                max_epochs=30,
                accelerator="auto",
                devices=1,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=5, mode="min"),
                ],
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )

            trainer.fit(module, train_dl, val_dl)

            # Evaluate F1 on validation set
            module.eval()
            all_preds = []
            with torch.no_grad():
                for batch in val_dl:
                    x = batch[0]
                    logits = module(x)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.append(preds)

            all_preds = np.concatenate(all_preds)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                score = f1_score(v_lab, all_preds, average="weighted")

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        best = study.best_params

        # Reconstruct hidden_dims from best params
        n_layers = best["n_layers"]
        dim = best["first_dim"]
        hidden_dims = []
        for _ in range(n_layers):
            hidden_dims.append(dim)
            dim = max(32, dim // 2)

        self.config.hidden_dims = hidden_dims
        self.config.dropout_rate = best["dropout"]
        self.config.learning_rate = best["lr"]
        self.config.weight_decay = best["weight_decay"]
        self.config.batch_size = best["batch_size"]

        self.log.info(
            f"Best score ({self.config.tuning_metric}): {study.best_value:.4f}"
        )
        self.log.info(f"Completed: {len(study.trials)} trials")
        self.log.info("Best parameters:")
        for key, value in best.items():
            self.log.info(f"  {key}: {value}")

    def _subsample_for_tuning(self) -> tuple:
        n_total = len(self.train_features)
        fraction = self.config.tuning_subsample

        if fraction >= 1.0:
            self.log.info(f"Using full training set for tuning ({n_total:,})")
            return self.train_features, self.train_labels

        rng = np.random.RandomState(self.config.random_state)
        indices = []

        for cls in np.unique(self.train_labels):
            cls_idx = np.where(self.train_labels == cls)[0]
            n_cls = max(2, int(len(cls_idx) * fraction))
            sampled = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
            indices.extend(sampled)

        indices = np.array(indices)
        rng.shuffle(indices)

        self.log.info(f"Subsampled {len(indices):,} / {n_total:,} for tuning")
        return self.train_features[indices], self.train_labels[indices]

    # ------------------------------------------------------------------
    # SMOTE (model-agnostic, operates on numpy)
    # ------------------------------------------------------------------

    def apply_smote(self) -> None:
        """SMOTE on train set only. val/test are never modified."""
        self.log.info("SMOTE data augmentation...")

        train_for_smote = np.nan_to_num(self.train_features, nan=0.0)

        unique, counts = np.unique(self.train_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))

        min_samples_for_smote = 100
        removed_classes = [
            cls for cls, count in class_counts.items() if count < min_samples_for_smote
        ]

        if removed_classes:
            lines = [
                f"Removing {len(removed_classes)} classes with "
                f"< {min_samples_for_smote} samples before SMOTE:"
            ]
            for cls in removed_classes:
                label = self.label_encoder.classes_[cls]
                lines.append(f"  - {label}: {class_counts[cls]} samples")
            self.log.warning("\n".join(lines))

            train_mask = ~np.isin(self.train_labels, removed_classes)
            val_mask = ~np.isin(self.val_labels, removed_classes)
            test_mask = ~np.isin(self.test_labels, removed_classes)

            train_for_smote = train_for_smote[train_mask]
            self.train_features = self.train_features[train_mask]
            self.train_labels = self.train_labels[train_mask]
            self.val_features = self.val_features[val_mask]
            self.val_labels = self.val_labels[val_mask]
            self.test_features = self.test_features[test_mask]
            self.test_labels = self.test_labels[test_mask]

            remaining = [
                self.label_encoder.classes_[i]
                for i in range(len(self.label_encoder.classes_))
                if i not in removed_classes
            ]
            old_classes = self.label_encoder.classes_
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(remaining)

            self.train_labels = self.label_encoder.transform(
                [old_classes[i] for i in self.train_labels]
            )
            self.val_labels = self.label_encoder.transform(
                [old_classes[i] for i in self.val_labels]
            )
            self.test_labels = self.label_encoder.transform(
                [old_classes[i] for i in self.test_labels]
            )

            unique, counts = np.unique(self.train_labels, return_counts=True)
            class_counts = dict(zip(unique, counts))

        max_count = max(counts)
        smote_strategy = {}
        for cls, count in class_counts.items():
            global_target = int(max_count * self.config.smote_ratio)
            local_cap = count * self.config.smote_max_multiplier
            target = min(global_target, local_cap)
            if count < target:
                smote_strategy[cls] = target

        if not smote_strategy:
            self.log.warning("No classes need SMOTE augmentation")
            self.train_features_balanced = self.train_features
            self.train_labels_balanced = self.train_labels
            return

        lines = ["\nSMOTE strategy:"]
        for cls in smote_strategy:
            label = self.label_encoder.classes_[cls]
            original = class_counts[cls]
            target = smote_strategy[cls]
            lines.append(f"  {label:<35} {original:>6,} -> {target:>6,}")
        self.log.info("\n".join(lines))

        min_k = min(class_counts[c] for c in smote_strategy)
        k_neighbors = min(self.config.smote_k_neighbors, min_k - 1)

        smote = SMOTE(
            sampling_strategy=smote_strategy,
            k_neighbors=NearestNeighbors(n_neighbors=k_neighbors, n_jobs=-1),
            random_state=self.config.random_state,
        )

        self.log.info(
            f"SMOTE initialized: sampling_strategy={smote_strategy}, k_neighbors={k_neighbors}"
        )

        balanced_features, self.train_labels_balanced = smote.fit_resample(
            train_for_smote, self.train_labels
        )

        n_original = len(self.train_features)
        self.train_features_balanced = balanced_features.copy()
        self.train_features_balanced[:n_original] = self.train_features

        self.log.info(
            f"Training set: {len(self.train_features):,} -> "
            f"{len(self.train_features_balanced):,}"
        )

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        self.log.info("Building ResNet MLP classifier...")

        n_classes = len(self.label_encoder.classes_)
        input_dim = self.train_features_balanced.shape[1]

        self.model = ResNetMLP(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            n_classes=n_classes,
            dropout=self.config.dropout_rate,
        )

        # Compute inverse-frequency class weights
        class_weights = None
        if self.config.use_class_weights:
            counts = np.bincount(
                self.train_labels_balanced, minlength=n_classes
            ).astype(np.float32)
            weights = 1.0 / np.maximum(counts, 1.0)
            weights = weights / weights.sum() * n_classes
            class_weights = torch.tensor(weights, dtype=torch.float32)
            self.log.info("Using inverse-frequency class weights")

        self.lightning_module = ClassifierLightningModule(
            model=self.model,
            class_weights=class_weights,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            reduce_lr_factor=self.config.reduce_lr_factor,
            reduce_lr_patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
        )

        arch = " -> ".join(
            [str(input_dim)]
            + [str(d) for d in self.config.hidden_dims]
            + [str(n_classes)]
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        self.log.info(f"Architecture: {arch}")
        self.log.info(f"Total parameters: {total_params:,}")
        self.log.info(f"Classes: {n_classes}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(self) -> None:
        self.log.info("Training ResNet MLP...")

        train_ds = TensorDataset(
            torch.tensor(self.train_features_balanced, dtype=torch.float32),
            torch.tensor(self.train_labels_balanced, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(self.val_features, dtype=torch.float32),
            torch.tensor(self.val_labels, dtype=torch.long),
        )

        num_workers = 4 if os.name != "nt" else 0

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

        os.makedirs("./artifacts", exist_ok=True)

        self.loss_history = LossHistoryCallback()

        callbacks = [
            EarlyStopping(
                monitor="val_acc",
                patience=self.config.early_stopping_patience,
                min_delta=1e-4,
                mode="max",
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="./artifacts",
                filename="classifier_temp",
                monitor="val_acc",
                save_top_k=1,
                mode="max",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            self.loss_history,
        ]

        trainer = L.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            gradient_clip_val=self.config.gradient_clip_val,
            log_every_n_steps=50,
            logger=True,
        )

        trainer.fit(self.lightning_module, train_loader, val_loader)

        # Restore best checkpoint
        best_path = callbacks[1].best_model_path
        if best_path:
            self.lightning_module = ClassifierLightningModule.load_from_checkpoint(
                best_path,
                model=self.model,
            )
            self.log.info(f"Loaded best model from {best_path}")

        epochs_trained = (
            trainer.current_epoch + 1
            if trainer.current_epoch is not None
            else self.config.max_epochs
        )
        self.log.info(f"Training completed: {epochs_trained} epochs")
        self.log.info(f"Best validation accuracy: {callbacks[1].best_model_score:.4f}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self) -> None:
        self.log.info("Evaluating model...")

        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        test_tensor = torch.tensor(self.test_features, dtype=torch.float32)
        test_ds = TensorDataset(test_tensor)
        test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False
        )

        all_logits = []
        with torch.no_grad():
            for (batch_x,) in test_loader:
                batch_x = batch_x.to(self.device)
                logits = self.lightning_module(batch_x)
                all_logits.append(logits.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(all_logits, dim=1).numpy()
        self.predictions = all_logits.argmax(dim=1).numpy()
        self.prediction_probs = probs

        self.test_accuracy = (self.predictions == self.test_labels).mean()
        self.log.info(f"Test accuracy: {self.test_accuracy:.4f}")

        report = classification_report(
            self.test_labels,
            self.predictions,
            target_names=self.label_encoder.classes_,
            digits=4,
        )
        self.log.info(f"\nClassification Report:\n  {report}")

        # Per-class FPR (within attack-only test set)
        cm = confusion_matrix(self.test_labels, self.predictions)
        lines = ["\nPer-class FPR (within attack test set):"]
        for i, cls in enumerate(self.label_encoder.classes_):
            fp = cm[:, i].sum() - cm[i, i]
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            lines.append(f"  {cls:<35} {fpr*100:.4f}%")
        self.log.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self) -> None:
        self.log.info("Saving results...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./artifacts", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        output_df = pd.DataFrame(
            {
                "Label": self.label_encoder.inverse_transform(self.test_labels),
                "predicted_label": self.label_encoder.inverse_transform(
                    self.predictions
                ),
                "correct": self.test_labels == self.predictions,
            }
        )
        for idx, class_name in enumerate(self.label_encoder.classes_):
            output_df[f"prob_{class_name}"] = self.prediction_probs[:, idx]

        csv_path = Path("outputs") / "classifier.csv"
        output_df.to_csv(csv_path, index=False)
        self.log.info(f"Saved: {csv_path}")

        # Save model as .pt (state_dict + architecture params)
        model_path = Path("artifacts") / "classifier.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.model.input_dim,
                "hidden_dims": self.model.hidden_dims,
                "n_classes": self.model.n_classes,
                "dropout": self.model.dropout,
            },
            model_path,
        )
        self.log.info(f"Saved: {model_path}")

        encoder_path = Path("artifacts") / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        self.log.info(f"Saved: {encoder_path}")

        config_data = {
            "encoder": self.label_encoder,
            "feature_names": list(self.features.columns),
            "test_accuracy": float(self.test_accuracy),
            "n_classes": len(self.label_encoder.classes_),
            "hidden_dims": self.model.hidden_dims,
            "input_dim": self.model.input_dim,
        }
        config_path = Path("artifacts") / "classifier_config.pkl"
        joblib.dump(config_data, config_path)
        self.log.info(f"Saved: {config_path}")

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def generate_visualizations(self) -> None:
        self.log.info("Generating visualizations...")

        os.makedirs("./plots", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Confusion matrix
        ax = axes[0, 0]
        cm = confusion_matrix(self.test_labels, self.predictions)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_normalized,
            annot=False,
            cmap="Blues",
            ax=ax,
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        ax.set_title("Confusion Matrix (Normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # 2. Training / Validation loss curves
        ax = axes[0, 1]
        if self.loss_history and self.loss_history.train_losses:
            epochs = range(1, len(self.loss_history.train_losses) + 1)
            ax.plot(
                epochs, self.loss_history.train_losses, label="Train Loss", color="blue"
            )
            if self.loss_history.val_losses:
                val_epochs = range(1, len(self.loss_history.val_losses) + 1)
                ax.plot(
                    val_epochs,
                    self.loss_history.val_losses,
                    label="Val Loss",
                    color="orange",
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training / Validation Loss")
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No loss history",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Training / Validation Loss")

        # 3. Per-class accuracy
        ax = axes[1, 0]
        accuracies = []
        labels_list = []
        for idx, label in enumerate(self.label_encoder.classes_):
            mask = self.test_labels == idx
            if mask.sum() > 0:
                acc = (self.predictions[mask] == idx).sum() / mask.sum()
                accuracies.append(acc)
                labels_list.append(label[:20])

        y_pos = np.arange(len(labels_list))
        colors = [
            "red" if acc < 0.5 else "orange" if acc < 0.8 else "green"
            for acc in accuracies
        ]
        ax.barh(y_pos, accuracies, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_list, fontsize=8)
        ax.set_xlabel("Accuracy")
        ax.set_title("Per-Class Accuracy")
        ax.grid(alpha=0.3, axis="x")

        # 4. Class distribution
        ax = axes[1, 1]
        train_dist = np.bincount(
            self.train_labels_balanced,
            minlength=len(self.label_encoder.classes_),
        )
        test_dist = np.bincount(
            self.test_labels, minlength=len(self.label_encoder.classes_)
        )
        x = np.arange(len(self.label_encoder.classes_))
        width = 0.35
        ax.bar(x - width / 2, train_dist, width, label="Train (SMOTE)", alpha=0.8)
        ax.bar(x + width / 2, test_dist, width, label="Test", alpha=0.8)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = Path("plots") / "classifier_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()

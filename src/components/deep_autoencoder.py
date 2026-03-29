import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import UNIFIED_FEATURE_NAMES, DeepAutoencoderConfig
from utils import Logger


class AutoencoderModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int],
        encoding_dim: int,
        dropout_rates: List[float],
        l2_reg: float = 0.0001,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        encoder_layers = []
        prev_size = input_dim
        for size, dropout in zip(layer_sizes, dropout_rates):
            encoder_layers.append(nn.Linear(prev_size, size))
            encoder_layers.append(nn.BatchNorm1d(size))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_size = size

        encoder_layers.append(nn.Linear(prev_size, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_size = encoding_dim
        for size, dropout in zip(reversed(layer_sizes), reversed(dropout_rates)):
            decoder_layers.append(nn.Linear(prev_size, size))
            decoder_layers.append(nn.BatchNorm1d(size))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_size = size

        decoder_layers.append(nn.Linear(prev_size, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.l2_reg = l2_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoencoderLightningModule(L.LightningModule):
    def __init__(
        self,
        model: AutoencoderModel,
        learning_rate: float = 0.001,
        clipnorm: float = 1.0,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 8,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr
        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = torch.mean(torch.abs(x_hat - x))

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = torch.mean(torch.abs(x_hat - x))

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.model.l2_reg,
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


class DeepAutoencoder:
    def __init__(self, config: Optional[DeepAutoencoderConfig] = None) -> None:
        self.benign_data: Optional[pd.DataFrame] = None
        self.attack_data: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.features: Optional[pd.DataFrame] = None
        self.binary_labels: Optional[pd.Series] = None

        self.benign_features: Optional[pd.DataFrame] = None
        self.benign_train: Optional[pd.DataFrame] = None
        self.benign_val: Optional[pd.DataFrame] = None

        self.test_features: Optional[pd.DataFrame] = None
        self.test_labels: Optional[pd.Series] = None  # binary
        self.test_labels_orig: Optional[pd.Series] = None

        self.scaler: Optional[StandardScaler] = None
        self.clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.benign_train_scaled: Optional[np.ndarray] = None
        self.benign_val_scaled: Optional[np.ndarray] = None
        self.test_features_scaled: Optional[np.ndarray] = None

        self.autoencoder_model: Optional[AutoencoderModel] = None
        self.lightning_module: Optional[AutoencoderLightningModule] = None

        self.ae_mse_scores: Optional[np.ndarray] = None  # test set (benign + attack)

        self.config: DeepAutoencoderConfig = config or DeepAutoencoderConfig()
        self.log: Logger = Logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_names: List[str] = UNIFIED_FEATURE_NAMES

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.lightning_module is not None:
            self.lightning_module.cpu()
            self.lightning_module = None
        if self.autoencoder_model is not None:
            self.autoencoder_model.cpu()
            self.autoencoder_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        plt.close("all")
        return False

    def check_environment(self) -> None:
        self.log.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")
            self.log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.log.info(f"CUDA: {torch.version.cuda}")
        else:
            self.log.info("GPU: No GPU detected, using CPU")

    def load_data(self) -> None:
        self.log.info("Loading data from outputs/preprocessing_benign.csv...")
        self.benign_data = pd.read_csv("./outputs/preprocessing_benign.csv")
        self.benign_data.columns = self.benign_data.columns.str.strip()

        self.log.info("Loading data from outputs/preprocessing_attack.csv...")
        self.attack_data = pd.read_csv("./outputs/preprocessing_attack.csv")
        self.attack_data.columns = self.attack_data.columns.str.strip()

        self.log.info(f"Normal samples: {len(self.benign_data):,}")
        self.log.info(f"Attack samples: {len(self.attack_data):,}")

    def prepare_data(self) -> None:
        self.log.info("Preparing data...")

        available = [f for f in self.feature_names if f in self.benign_data.columns]
        self.log.info(
            f"Using {len(available)}/{len(self.feature_names)} features for AE"
        )

        self.benign_features = self.benign_data[available].copy()
        attack_features = self.attack_data[available].copy()

        benign_trainval, benign_test = train_test_split(
            self.benign_features,
            test_size=self.config.test_split,
            random_state=self.config.split_random_state,
        )
        val_ratio = self.config.validation_split / (1.0 - self.config.test_split)
        self.benign_train, self.benign_val = train_test_split(
            benign_trainval,
            test_size=val_ratio,
            random_state=self.config.split_random_state,
        )

        test_features = pd.concat([benign_test, attack_features], ignore_index=True)
        test_labels_orig = pd.concat(
            [
                self.benign_data.loc[benign_test.index, "Label"],
                self.attack_data["Label"],
            ],
            ignore_index=True,
        )
        test_binary = (~test_labels_orig.isin(["Normal"])).astype(int)

        self.test_features = test_features
        self.test_labels = test_binary
        self.test_labels_orig = test_labels_orig

        self.labels = pd.concat(
            [self.benign_data["Label"], self.attack_data["Label"]], ignore_index=True
        )
        self.features = pd.concat(
            [self.benign_features, attack_features], ignore_index=True
        )
        self.binary_labels = (~self.labels.isin(["Normal"])).astype(int)

        self.log.info(
            f"Benign split:  train={len(self.benign_train):,} "
            f"/ val={len(self.benign_val):,} "
            f"/ test={len(benign_test):,}"
        )
        self.log.info(f"Attack samples (test only): {len(attack_features):,}")
        self.log.info(
            f"Test set total: {len(self.test_features):,} "
            f"(benign={int((self.test_labels == 0).sum()):,}, "
            f"attack={int((self.test_labels == 1).sum()):,})"
        )
        self.log.info(f"Number of features: {self.features.shape[1]}")

    def preprocess_data(self) -> None:
        self.log.info("Preprocessing data...")

        def _clean(df: pd.DataFrame) -> pd.DataFrame:
            return df.replace([np.inf, -np.inf], np.nan).fillna(self.config.fill_value)

        self.benign_train = _clean(self.benign_train)
        self.benign_val = _clean(self.benign_val)
        self.test_features = _clean(self.test_features)

        self.clip_params = {}
        for col in self.benign_train.columns:
            lower = self.benign_train[col].quantile(self.config.winsorize_lower)
            upper = self.benign_train[col].quantile(self.config.winsorize_upper)
            self.benign_train[col] = np.clip(self.benign_train[col], lower, upper)
            self.benign_val[col] = np.clip(self.benign_val[col], lower, upper)
            self.test_features[col] = np.clip(self.test_features[col], lower, upper)
            self.clip_params[col] = {"lower": float(lower), "upper": float(upper)}

        self.scaler = StandardScaler()
        self.benign_train_scaled = self.scaler.fit_transform(self.benign_train)
        self.benign_val_scaled = self.scaler.transform(self.benign_val)
        self.test_features_scaled = self.scaler.transform(self.test_features)

        def _clip_scaled(arr: np.ndarray) -> np.ndarray:
            return np.clip(arr, self.config.clip_min, self.config.clip_max)

        self.benign_train_scaled = _clip_scaled(self.benign_train_scaled)
        self.benign_val_scaled = _clip_scaled(self.benign_val_scaled)
        self.test_features_scaled = _clip_scaled(self.test_features_scaled)

        self.log.info("Preprocessing completed")

    def build_autoencoder(self) -> None:
        self.log.info("Building Deep Autoencoder...")

        input_dim = self.benign_train_scaled.shape[1]

        layer_info = " -> ".join(
            [str(input_dim)]
            + [str(s) for s in self.config.layer_sizes]
            + [str(self.config.encoding_dim)]
        )
        self.log.info(f"Architecture: {layer_info}")

        self.autoencoder_model = AutoencoderModel(
            input_dim=input_dim,
            layer_sizes=self.config.layer_sizes,
            encoding_dim=self.config.encoding_dim,
            dropout_rates=self.config.dropout_rates,
            l2_reg=self.config.l2_reg,
        )

        self.lightning_module = AutoencoderLightningModule(
            model=self.autoencoder_model,
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.clipnorm,
            reduce_lr_factor=self.config.reduce_lr_factor,
            reduce_lr_patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
        )

        total_params = sum(p.numel() for p in self.autoencoder_model.parameters())
        self.log.info(f"Total parameters: {total_params:,}")

    def train_autoencoder(self) -> None:
        """
        AE is trained on benign_train; benign_val is used as the Lightning
        validation set for early-stopping / LR scheduling.
        Neither set contains attack samples → no leakage.
        """
        self.log.info("Training Deep Autoencoder with PyTorch Lightning...")

        train_dataset = TensorDataset(torch.FloatTensor(self.benign_train_scaled))
        val_dataset = TensorDataset(torch.FloatTensor(self.benign_val_scaled))

        num_workers = 4 if os.name != "nt" else 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

        os.makedirs("./artifacts", exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                min_delta=1e-6,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="./artifacts",
                filename="autoencoder_temp",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = L.Trainer(
            max_epochs=self.config.epochs,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            gradient_clip_val=self.config.clipnorm,
            log_every_n_steps=50,
            logger=True,
        )

        trainer.fit(self.lightning_module, train_loader, val_loader)

        best_model_path = callbacks[1].best_model_path
        if best_model_path:
            self.lightning_module = AutoencoderLightningModule.load_from_checkpoint(
                best_model_path,
                model=self.autoencoder_model,
            )
            self.log.info(f"Loaded best model from {best_model_path}")

        epochs_trained = (
            trainer.current_epoch + 1
            if trainer.current_epoch is not None
            else self.config.epochs
        )

        self.log.info(f"Training completed: {epochs_trained} epochs")
        self.log.info(f"Best validation loss: {callbacks[1].best_model_score:.6f}")

    def _ae_predict_mse(self, features_scaled: np.ndarray) -> np.ndarray:
        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        batch_size = 2048
        n_samples = len(features_scaled)
        mse_scores = np.zeros(n_samples, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = torch.FloatTensor(features_scaled[start:end]).to(self.device)
                recon = self.lightning_module(batch).cpu().numpy()
                mse_scores[start:end] = np.mean(
                    np.square(features_scaled[start:end] - recon), axis=1
                )

        return mse_scores

    def predict_autoencoder(self) -> None:
        self.log.info("Calculating Deep AE anomaly scores on test set...")

        self.ae_mse_scores = self._ae_predict_mse(self.test_features_scaled)

        ae_mse_benign = self.ae_mse_scores[self.test_labels == 0]
        ae_mse_attack = self.ae_mse_scores[self.test_labels == 1]

        separation = (
            ae_mse_attack.mean() / ae_mse_benign.mean()
            if ae_mse_benign.mean() > 0
            else 0
        )

        self.log.info(
            "\n".join(
                [
                    "\nAE MSE statistics (test set):",
                    f"Normal: Mean={ae_mse_benign.mean():.6f}, Median={np.median(ae_mse_benign):.6f}",
                    f"Attack: Mean={ae_mse_attack.mean():.6f}, Median={np.median(ae_mse_attack):.6f}",
                    f"Separation: {separation:.2f}x",
                ]
            )
        )

    def save_results(self) -> None:
        self.log.info("Saving results...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./artifacts", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        attack_mask = (self.test_labels == 1).values

        output = pd.DataFrame(
            self.test_features_scaled[attack_mask],
            columns=list(self.benign_train.columns),
        )
        output["ae_anomaly_score"] = self.ae_mse_scores[attack_mask]
        output["Label"] = self.test_labels_orig[attack_mask].values

        output_path = Path("outputs") / "deep_ae_scores.csv"
        output.to_csv(output_path, index=False)
        self.log.info(
            f"Saved: {output_path} ({len(output):,} rows, {output.shape[1]} columns)"
        )

        normal_scores = self.ae_mse_scores[self.test_labels == 0]
        ae_threshold = float(normal_scores.mean() + 2 * normal_scores.std())
        self.log.info(
            f"\nAE threshold (mean+2std): {ae_threshold:.6f}"
            f"\n  benign test mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}"
        )

        model_ae_path = Path("artifacts") / "deep_autoencoder.pt"
        torch.save(
            {
                "model_state_dict": self.autoencoder_model.state_dict(),
                "input_dim": self.autoencoder_model.input_dim,
                "encoding_dim": self.autoencoder_model.encoding_dim,
                "layer_sizes": self.config.layer_sizes,
                "dropout_rates": self.config.dropout_rates,
                "l2_reg": self.config.l2_reg,
            },
            model_ae_path,
        )
        self.log.info(f"Saved: {model_ae_path}")

        config_data = {
            "scaler": self.scaler,
            "clip_params": self.clip_params,
            "encoding_dim": self.config.encoding_dim,
            "feature_names": list(self.benign_train.columns),
            "ae_threshold": ae_threshold,
        }
        config_path = Path("artifacts") / "deep_ae_config.pkl"
        joblib.dump(config_data, config_path)
        self.log.info(f"Saved: {config_path}")

    def generate_visualizations(self) -> None:
        self.log.info("Generating visualizations...")

        os.makedirs("./plots", exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        bins = 50
        ax.hist(
            self.ae_mse_scores[self.test_labels == 0],
            bins=bins,
            alpha=0.7,
            label="Normal",
            color="green",
            density=True,
        )
        ax.hist(
            self.ae_mse_scores[self.test_labels == 1],
            bins=bins,
            alpha=0.7,
            label="Attack",
            color="red",
            density=True,
        )
        ax.set_xlabel("AE MSE Score")
        ax.set_title("Anomaly Score Distribution (Test Set)")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.boxplot(
            [
                self.ae_mse_scores[self.test_labels == 0],
                self.ae_mse_scores[self.test_labels == 1],
            ],
            labels=["Normal", "Attack"],
            showfliers=False,
        )
        ax.set_ylabel("AE MSE Score")
        ax.set_title("Score Distribution by Class (Test Set)")
        ax.grid(alpha=0.3)

        ax = axes[2]
        attack_labels = self.test_labels_orig[self.test_labels == 1]
        attack_scores_all = self.ae_mse_scores[self.test_labels == 1]

        type_means = {}
        for label in sorted(attack_labels.unique()):
            mask = attack_labels.values == label
            type_means[label[:20]] = attack_scores_all[mask].mean()

        if type_means:
            sorted_types = sorted(type_means.items(), key=lambda x: x[1], reverse=True)
            names = [t[0] for t in sorted_types[:10]]
            values = [t[1] for t in sorted_types[:10]]
            ax.barh(names, values, color="coral")
            ax.set_xlabel("Mean AE MSE Score")
            ax.set_title("Top 10 Attack Types by AE Score")
            ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plot_path = Path("plots") / "deep_ae_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()

        self._plot_roc_pr()
        self._plot_score_cdf()
        self._plot_per_feature_reconstruction()
        self._plot_latent_tsne()

    def _plot_roc_pr(self) -> None:
        """ROC Curve + Precision-Recall Curve（AE standalone）"""
        from sklearn.metrics import (
            auc,
            average_precision_score,
            precision_recall_curve,
            roc_curve,
        )

        fpr, tpr, thresholds = roc_curve(self.test_labels, self.ae_mse_scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(
            self.test_labels, self.ae_mse_scores
        )
        ap = average_precision_score(self.test_labels, self.ae_mse_scores)

        # 最佳 threshold（Youden's J）
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_threshold = thresholds[best_idx]
        best_fpr = fpr[best_idx]
        best_tpr = tpr[best_idx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
        ax.scatter(
            best_fpr,
            best_tpr,
            color="red",
            zorder=5,
            label=f"Best threshold = {best_threshold:.4f}",
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (AE Standalone)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(recall, precision, color="steelblue", lw=2, label=f"AP = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (AE Standalone)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = Path("plots") / "deep_ae_roc_pr.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path} (AUC={roc_auc:.4f}, AP={ap:.4f})")
        plt.close()

    def _plot_score_cdf(self) -> None:
        normal_scores = self.ae_mse_scores[self.test_labels == 0]
        attack_scores = self.ae_mse_scores[self.test_labels == 1]

        fig, ax = plt.subplots(figsize=(8, 5))
        for scores, label, color in [
            (normal_scores, "Normal", "green"),
            (attack_scores, "Attack", "red"),
        ]:
            sorted_s = np.sort(scores)
            cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
            ax.plot(sorted_s, cdf, label=label, color=color, lw=2)

        # mean+2std threshold 標記
        threshold = float(normal_scores.mean() + 2 * normal_scores.std())
        ax.axvline(
            threshold,
            color="navy",
            linestyle="--",
            lw=1.5,
            label=f"Threshold (mean+2σ) = {threshold:.4f}",
        )

        ax.set_xlabel("AE MSE Score")
        ax.set_ylabel("CDF")
        ax.set_title("Cumulative Distribution of Anomaly Scores")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = Path("plots") / "deep_ae_cdf.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()

    def _plot_per_feature_reconstruction(self) -> None:
        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        normal_mask = self.test_labels.values == 0
        attack_mask = self.test_labels.values == 1

        def _get_per_feature_mse(scaled: np.ndarray) -> np.ndarray:
            results = []
            batch_size = 2048
            with torch.no_grad():
                for start in range(0, len(scaled), batch_size):
                    batch = torch.FloatTensor(scaled[start : start + batch_size]).to(
                        self.device
                    )
                    recon = self.lightning_module(batch).cpu().numpy()
                    results.append(
                        np.square(scaled[start : start + batch_size] - recon)
                    )
            return np.mean(np.concatenate(results, axis=0), axis=0)

        normal_mse = _get_per_feature_mse(self.test_features_scaled[normal_mask])
        attack_mse = _get_per_feature_mse(self.test_features_scaled[attack_mask])

        feature_names = list(self.benign_train.columns)
        diff = attack_mse - normal_mse
        sorted_idx = np.argsort(diff)[::-1][:20]

        x = np.arange(len(sorted_idx))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(
            x + width / 2,
            normal_mse[sorted_idx[::-1]],
            width,
            label="Normal",
            color="green",
            alpha=0.7,
        )
        ax.barh(
            x - width / 2,
            attack_mse[sorted_idx[::-1]],
            width,
            label="Attack",
            color="red",
            alpha=0.7,
        )
        ax.set_yticks(x)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]], fontsize=9)
        ax.set_xlabel("Mean Reconstruction MSE")
        ax.set_title("Top 20 Features by Reconstruction Error (Normal vs Attack)")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plot_path = Path("plots") / "deep_ae_feature_reconstruction.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()

    def _plot_latent_tsne(self, n_samples: int = 5000) -> None:
        """t-SNE 潛在空間視覺化（依攻擊類別上色）"""
        from sklearn.manifold import TSNE

        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        total = len(self.test_features_scaled)
        n_samples = min(n_samples, total)
        rng = np.random.default_rng(42)
        idx = rng.choice(total, n_samples, replace=False)

        x_sample = torch.FloatTensor(self.test_features_scaled[idx]).to(self.device)
        labels_sample = self.test_labels_orig.iloc[idx].values

        with torch.no_grad():
            latent = self.lightning_module.model.encode(x_sample).cpu().numpy()

        self.log.info(f"Running t-SNE on {n_samples} samples (latent dim={latent.shape[1]})...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
        latent_2d = tsne.fit_transform(latent)

        label_set = sorted(set(labels_sample))
        cmap = plt.cm.get_cmap("tab10", len(label_set))

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, label in enumerate(label_set):
            mask = labels_sample == label
            ax.scatter(
                latent_2d[mask, 0],
                latent_2d[mask, 1],
                c=[cmap(i)],
                label=label,
                alpha=0.5,
                s=8,
            )
        ax.legend(markerscale=2, fontsize=9)
        ax.set_title(f"t-SNE of AE Latent Space (n={n_samples})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        plot_path = Path("plots") / "deep_ae_tsne.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()
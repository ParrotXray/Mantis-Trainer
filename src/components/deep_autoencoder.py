import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightning as L
import matplotlib
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
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import (
    SEQUENCE_META_COLUMNS,
    UNIFIED_FEATURE_NAMES,
    DeepAutoencoderConfig,
    TrainingError,
)
from utils import Logger

matplotlib.use("Agg")


class PlainProgressCallback(L.Callback):
    """Print-based progress for Docker / non-TTY environments."""

    def __init__(self, logger: Logger, print_every_n_batches: int = 50):
        super().__init__()
        self.logger = logger
        self.print_every_n_batches = print_every_n_batches
        self._train_loss_sum: float = 0.0
        self._batch_count: int = 0
        self._epoch_start: float = time.time()

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._epoch_start = time.time()
        self._train_loss_sum = 0.0
        self._batch_count = 0

        lr = trainer.optimizers[0].param_groups[0]["lr"]
        self.logger.info(
            f"[Epoch {trainer.current_epoch}/{trainer.max_epochs}] "
            f"Start — {trainer.num_training_batches} batches  lr={lr:.2e}"
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        loss = outputs["loss"].item() if isinstance(outputs, dict) else float(outputs)
        self._train_loss_sum += loss
        self._batch_count += 1

        if (batch_idx + 1) % self.print_every_n_batches == 0:
            avg = self._train_loss_sum / self._batch_count
            total = trainer.num_training_batches
            pct = (batch_idx + 1) / total * 100
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            self.logger.info(
                f"Batch {batch_idx + 1:>5}/{total} ({pct:5.1f}%) train_loss={avg:.6f} lr={lr:.2e}"
            )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # Skip sanity check (batch_count == 0 means no training happened yet)
        if self._batch_count == 0:
            return

        elapsed = time.time() - self._epoch_start
        m = trainer.callback_metrics
        lr = trainer.optimizers[0].param_groups[0]["lr"]

        self.logger.info(
            f"[Epoch {trainer.current_epoch}/{trainer.max_epochs}] "
            f"Done {timedelta(seconds=elapsed)} | "
            f"train_loss={float(m.get('train_loss', float('nan'))):.6f}  "
            f"val_loss={float(m.get('val_loss', float('nan'))):.6f}  "
            f"val_mae={float(m.get('val_mae', float('nan'))):.6f}  "
            f"lr={lr:.2e}"
        )


class LSTMAutoencoderModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        encoding_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoding_dim = encoding_dim

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.encoder_fc = nn.Linear(hidden_size, encoding_dim)

        self.decoder_fc = nn.Linear(encoding_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=encoding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output_fc = nn.Linear(hidden_size, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder_lstm(x)
        return self.encoder_fc(h_n[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        _, (h_n, _) = self.encoder_lstm(x)
        encoded = self.encoder_fc(h_n[-1])

        decoder_input = encoded.unsqueeze(1).expand(
            batch_size, seq_len, self.encoding_dim
        )

        h_0 = (
            self.decoder_fc(encoded)
            .unsqueeze(0)
            .expand(self.num_layers, batch_size, self.hidden_size)
            .contiguous()
        )
        c_0 = torch.zeros_like(h_0)

        decoder_out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))
        return self.output_fc(decoder_out)


class LSTMAutoencoderLightningModule(L.LightningModule):
    def __init__(
        self,
        model: LSTMAutoencoderModel,
        learning_rate: float = 0.001,
        clipnorm: float = 1.0,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 5,
        min_lr: float = 1e-7,
        latent_norm_weight: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr
        self.latent_norm_weight = latent_norm_weight
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch[0]
        z = self.model.encode(x)
        x_hat = self(x)
        recon_loss = self.loss_fn(x_hat, x)
        latent_norm_loss = torch.mean(z.pow(2))
        loss = recon_loss + self.latent_norm_weight * latent_norm_loss
        mae = torch.mean(torch.abs(x_hat - x))
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_recon_loss", recon_loss, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log("train_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch[0]
        z = self.model.encode(x)
        x_hat = self(x)
        recon_loss = self.loss_fn(x_hat, x)
        latent_norm_loss = torch.mean(z.pow(2))
        loss = recon_loss + self.latent_norm_weight * latent_norm_loss
        mae = torch.mean(torch.abs(x_hat - x))
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_recon_loss", recon_loss, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log("val_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
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


def _make_train_sequences(
    df: pd.DataFrame,
    scaled: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    n_features = scaled.shape[1]
    has_meta = all(c in df.columns for c in SEQUENCE_META_COLUMNS)

    sequences: List[np.ndarray] = []

    if has_meta:
        tmp = df[["src_ip", "timestamp"]].copy()
        tmp["_pos"] = range(len(df))
        for _, group in tmp.groupby("src_ip", sort=False):
            group_sorted = group.sort_values("timestamp")
            positions = group_sorted["_pos"].values
            n = len(positions)
            for start in range(0, n - window_size + 1, stride):
                sequences.append(scaled[positions[start : start + window_size]])
    else:
        n = len(scaled)
        for start in range(0, n - window_size + 1, stride):
            sequences.append(scaled[start : start + window_size])

    if not sequences:
        return np.empty((0, window_size, n_features), dtype=np.float32)
    return np.stack(sequences).astype(np.float32)


def _make_per_flow_sequences(
    df: pd.DataFrame,
    scaled: np.ndarray,
    window_size: int,
) -> np.ndarray:
    n_flows = len(scaled)
    n_features = scaled.shape[1]
    sequences = np.zeros((n_flows, window_size, n_features), dtype=np.float32)

    has_meta = all(c in df.columns for c in SEQUENCE_META_COLUMNS)

    if has_meta:
        tmp = df[["src_ip", "timestamp"]].copy()
        tmp["_pos"] = range(n_flows)
        for _, group in tmp.groupby("src_ip", sort=False):
            group_sorted = group.sort_values("timestamp")
            positions = group_sorted["_pos"].values
            for k, pos in enumerate(positions):
                start_in_group = max(0, k + 1 - window_size)
                window_pos = positions[start_in_group : k + 1]
                pad_len = window_size - len(window_pos)
                sequences[pos, pad_len:] = scaled[window_pos]
    else:
        for i in range(n_flows):
            start = max(0, i + 1 - window_size)
            pad_len = window_size - (i + 1 - start)
            sequences[i, pad_len:] = scaled[start : i + 1]

    return sequences


class DeepAutoencoder:
    def __init__(self, config: Optional[DeepAutoencoderConfig] = None) -> None:
        self.benign_data: Optional[pd.DataFrame] = None
        self.attack_data: Optional[pd.DataFrame] = None

        self._feature_cols: Optional[List[str]] = None

        self.benign_train: Optional[pd.DataFrame] = None
        self.benign_val: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.test_labels: Optional[pd.Series] = None
        self.test_labels_orig: Optional[pd.Series] = None

        self.benign_train_scaled: Optional[np.ndarray] = None
        self.benign_val_scaled: Optional[np.ndarray] = None
        self.test_features_scaled: Optional[np.ndarray] = None

        self.train_sequences: Optional[np.ndarray] = None
        self.val_sequences: Optional[np.ndarray] = None

        self.ae_mse_scores: Optional[np.ndarray] = None
        self.ae_threshold: Optional[Dict[str, float]] = None

        self.iso_forest: Optional[IsolationForest] = None
        self.iso_scores: Optional[np.ndarray] = None
        self.iso_threshold: Optional[float] = None
        self.combined_anomaly_mask: Optional[np.ndarray] = None

        self.scaler: Optional[StandardScaler] = None
        self.clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.log_transform_features: Optional[List[str]] = None

        self.autoencoder_model: Optional[LSTMAutoencoderModel] = None
        self.lightning_module: Optional[LSTMAutoencoderLightningModule] = None

        self.config: DeepAutoencoderConfig = config or DeepAutoencoderConfig()
        self.log: Logger = Logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        self.log.info("Loading data from outputs/preprocessing_benign.parquet...")
        self.benign_data = pd.read_parquet("./outputs/preprocessing_benign.parquet")
        self.benign_data.columns = self.benign_data.columns.str.strip()

        self.log.info("Loading data from outputs/preprocessing_attack.parquet...")
        self.attack_data = pd.read_parquet("./outputs/preprocessing_attack.parquet")
        self.attack_data.columns = self.attack_data.columns.str.strip()

        self.log.info(f"Benign samples : {len(self.benign_data):,}")
        self.log.info(f"Attack samples : {len(self.attack_data):,}")

        has_ts = "timestamp" in self.benign_data.columns
        has_ip = "src_ip" in self.benign_data.columns
        self.log.info(f"Sequence metadata — timestamp: {has_ts}, src_ip: {has_ip}")

    def prepare_data(self) -> None:
        self.log.info("Preparing data (time-based split)...")

        available_features = [
            f for f in self.feature_names if f in self.benign_data.columns
        ]
        self.log.info(
            f"Using {len(available_features)}/{len(self.feature_names)} "
            f"flow features for LSTM AE"
        )

        meta_cols = [c for c in SEQUENCE_META_COLUMNS if c in self.benign_data.columns]
        if "timestamp" not in meta_cols:
            raise TrainingError(
                "Time-based splits require a 'timestamp' field, but this is not present in benign_data."
            )

        all_cols = available_features + meta_cols + ["Label"]
        benign_all = self.benign_data[all_cols].copy()

        benign_sorted = benign_all.sort_values("timestamp").reset_index(drop=True)
        n = len(benign_sorted)

        test_frac = self.config.test_split
        val_frac = self.config.validation_split
        train_end = int(n * (1 - test_frac - val_frac))
        val_end = int(n * (1 - test_frac))

        self.benign_train = benign_sorted.iloc[:train_end].reset_index(drop=True)
        self.benign_val = benign_sorted.iloc[train_end:val_end].reset_index(drop=True)
        benign_test = benign_sorted.iloc[val_end:].reset_index(drop=True)

        self.log.info(
            f"Time-based split boundaries — "
            f"train ends: {datetime.fromtimestamp(benign_sorted['timestamp'].iloc[train_end - 1] / 1000).strftime('%Y-%m-%d %H:%M:%S')}, "
            f"val ends: {datetime.fromtimestamp(benign_sorted['timestamp'].iloc[val_end - 1] / 1000).strftime('%Y-%m-%d %H:%M:%S')}, "
            f"test ends: {datetime.fromtimestamp(benign_sorted['timestamp'].iloc[-1] / 1000).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        atk_meta_cols = [
            c for c in SEQUENCE_META_COLUMNS if c in self.attack_data.columns
        ]
        atk_cols = available_features + atk_meta_cols + ["Label"]
        attack_all = self.attack_data[
            [c for c in atk_cols if c in self.attack_data.columns]
        ].copy()

        test_labels_orig = pd.concat(
            [benign_test["Label"], attack_all["Label"]],
            ignore_index=True,
        )
        self.test_labels = (~test_labels_orig.isin(["Normal"])).astype(int)
        self.test_labels_orig = test_labels_orig

        self.test_df = pd.concat(
            [benign_test.reset_index(drop=True), attack_all.reset_index(drop=True)],
            ignore_index=True,
        )

        self._feature_cols = available_features

        self.log.info(
            f"Benign split: train={len(self.benign_train):,} "
            f"/ val={len(self.benign_val):,} "
            f"/ test={len(benign_test):,}"
        )
        self.log.info(f"Attack samples (test only): {len(attack_all):,}")
        self.log.info(
            f"Test set total : {len(self.test_df):,} "
            f"(benign={int((self.test_labels == 0).sum()):,}, "
            f"attack={int((self.test_labels == 1).sum()):,})"
        )
        self.log.info(f"Flow features  : {len(self._feature_cols)}")

    def preprocess_data(self) -> None:
        self.log.info("Preprocessing data (scale flow features only)...")

        def _feat(df: pd.DataFrame) -> pd.DataFrame:
            return df[self._feature_cols].copy()

        def _clean(df: pd.DataFrame) -> pd.DataFrame:
            return df.replace([np.inf, -np.inf], np.nan).fillna(self.config.fill_value)

        train_feat = _clean(_feat(self.benign_train))
        val_feat = _clean(_feat(self.benign_val))
        test_feat = _clean(_feat(self.test_df))

        # log1p heavy-tailed features (|skew| > 1): raw z-scoring these lets the
        # winsorize upper bound land several std past the post-scaling clip,
        # saturating any legitimate value near that bound to the clip ceiling
        # (confirmed for active_*/idle_*/*_iat_*/*_bytes/*_flag_cnt etc.).
        # `protocol` is a categorical code (6/17), never log-transformed.
        skewness = train_feat.skew()
        self.log_transform_features = [
            col
            for col in train_feat.columns
            if col != "protocol" and abs(skewness[col]) > 1.0
        ]
        self.log.info(
            f"Log1p transform ({len(self.log_transform_features)} features, "
            f"|skew| > 1.0): {self.log_transform_features}"
        )
        for col in self.log_transform_features:
            train_feat[col] = np.log1p(train_feat[col].clip(lower=0))
            val_feat[col] = np.log1p(val_feat[col].clip(lower=0))
            test_feat[col] = np.log1p(test_feat[col].clip(lower=0))

        self.clip_params = {}
        for col in train_feat.columns:
            lower = train_feat[col].quantile(self.config.winsorize_lower)
            upper = train_feat[col].quantile(self.config.winsorize_upper)
            train_feat[col] = np.clip(train_feat[col], lower, upper)
            val_feat[col] = np.clip(val_feat[col], lower, upper)
            test_feat[col] = np.clip(test_feat[col], lower, upper)
            self.clip_params[col] = {"lower": float(lower), "upper": float(upper)}

        self.scaler = StandardScaler()
        self.benign_train_scaled = self.scaler.fit_transform(train_feat)
        self.benign_val_scaled = self.scaler.transform(val_feat)
        self.test_features_scaled = self.scaler.transform(test_feat)

        def _clip_scaled(arr: np.ndarray) -> np.ndarray:
            return np.clip(arr, self.config.clip_min, self.config.clip_max)

        self.benign_train_scaled = _clip_scaled(self.benign_train_scaled)
        self.benign_val_scaled = _clip_scaled(self.benign_val_scaled)
        self.test_features_scaled = _clip_scaled(self.test_features_scaled)

        self.log.info("Preprocessing completed")

    def build_sequences(self) -> None:
        W = self.config.window_size
        S = self.config.stride
        self.log.info(f"Building sequences — window_size={W}, stride={S} (training)")

        self.train_sequences = _make_train_sequences(
            self.benign_train, self.benign_train_scaled, W, S
        )
        self.val_sequences = _make_train_sequences(
            self.benign_val, self.benign_val_scaled, W, S
        )

        self.log.info(
            f"Training sequences : {len(self.train_sequences):,} "
            f"shape={self.train_sequences.shape}"
        )
        self.log.info(
            f"Validation sequences: {len(self.val_sequences):,} "
            f"shape={self.val_sequences.shape}"
        )

        if len(self.train_sequences) == 0:
            raise TrainingError(
                "No training sequences were generated. "
                "Increase data size or decrease window_size."
            )

    def build_autoencoder(self) -> None:
        self.log.info("Building LSTM Deep Autoencoder...")

        input_dim = len(self._feature_cols)

        self.log.info(
            f"Architecture: input_dim={input_dim} "
            f"→ LSTM(hidden={self.config.hidden_size}, "
            f"layers={self.config.num_layers}) "
            f"→ bottleneck={self.config.encoding_dim} "
            f"→ LSTM decoder → {input_dim}"
        )

        self.autoencoder_model = LSTMAutoencoderModel(
            input_dim=input_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            encoding_dim=self.config.encoding_dim,
            dropout=self.config.dropout,
        )

        self.lightning_module = LSTMAutoencoderLightningModule(
            model=self.autoencoder_model,
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.clipnorm,
            reduce_lr_factor=self.config.reduce_lr_factor,
            reduce_lr_patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
            latent_norm_weight=self.config.latent_norm_weight,
        )

        total_params = sum(p.numel() for p in self.autoencoder_model.parameters())
        self.log.info(f"Total parameters: {total_params:,}")

    def train_autoencoder(self, resume_ckpt: Optional[Path] = None) -> None:
        self.log.info("Training LSTM Deep Autoencoder with PyTorch Lightning...")

        train_dataset = TensorDataset(torch.FloatTensor(self.train_sequences))
        val_dataset = TensorDataset(torch.FloatTensor(self.val_sequences))

        num_workers = min(4, os.cpu_count() or 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        os.makedirs("./artifacts", exist_ok=True)

        callbacks = [
            PlainProgressCallback(
                logger=self.log,
                print_every_n_batches=50,
            ),
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
            enable_progress_bar=False,
            gradient_clip_val=self.config.clipnorm,
            log_every_n_steps=50,
            logger=True,
        )

        resume_path = (
            resume_ckpt if resume_ckpt and os.path.exists(resume_ckpt) else None
        )
        if resume_path:
            self.log.info(f"Resuming from checkpoint: {resume_path}")
        else:
            self.log.info("No checkpoint provided, starting training from scratch...")

        trainer.fit(
            self.lightning_module,
            train_loader,
            val_loader,
            ckpt_path=resume_path or None,
        )

        best_model_path = callbacks[2].best_model_path
        if best_model_path:
            self.lightning_module = LSTMAutoencoderLightningModule.load_from_checkpoint(
                best_model_path,
                model=self.autoencoder_model,
            )
            self.log.info(f"Loaded best model from {best_model_path}")

        self.log.info(f"Training completed: {trainer.current_epoch + 1} epochs")
        self.log.info(f"Best validation loss: {callbacks[2].best_model_score:.6f}")

    def predict_autoencoder(self) -> None:
        self.log.info(
            "Calculating LSTM AE anomaly scores on test set "
            f"({len(self.test_df):,} flows)..."
        )

        val_seqs = _make_per_flow_sequences(
            self.benign_val,
            self.benign_val_scaled,
            self.config.window_size,
        )
        val_scores = self._ae_predict_mse(val_seqs)
        self.log.info(
            f"Validation scores — mean={val_scores.mean():.6f}, std={val_scores.std():.6f}"
        )

        per_flow_seqs = _make_per_flow_sequences(
            self.test_df,
            self.test_features_scaled,
            self.config.window_size,
        )

        self.ae_mse_scores = self._ae_predict_mse(per_flow_seqs)

        ae_mse_benign = self.ae_mse_scores[self.test_labels == 0]
        ae_mse_attack = self.ae_mse_scores[self.test_labels == 1]

        separation = (
            ae_mse_attack.mean() / ae_mse_benign.mean()
            if ae_mse_benign.mean() > 0
            else 0
        )

        candidate_names = [f"p{p}" for p in range(90, 100)] + [
            "mean+2std",
            "mean+1std",
            "Q3+1.5IQR",
            "Q3+3.0IQR",
        ]

        q1_val = float(np.percentile(val_scores, 25))
        q3_val = float(np.percentile(val_scores, 75))
        iqr_val = q3_val - q1_val

        val_candidate_values = np.array(
            [float(np.percentile(val_scores, p)) for p in range(90, 100)]
            + [
                float(val_scores.mean() + 2 * val_scores.std()),
                float(val_scores.mean() + 1 * val_scores.std()),
                q3_val + 1.5 * iqr_val,
                q3_val + 3.0 * iqr_val,
            ]
        )

        q1_test = float(np.percentile(ae_mse_benign, 25))
        q3_test = float(np.percentile(ae_mse_benign, 75))
        iqr_test = q3_test - q1_test

        test_candidate_values = np.array(
            [float(np.percentile(ae_mse_benign, p)) for p in range(90, 100)]
            + [
                float(ae_mse_benign.mean() + 2 * ae_mse_benign.std()),
                float(ae_mse_benign.mean() + 1 * ae_mse_benign.std()),
                q3_test + 1.5 * iqr_test,
                q3_test + 3.0 * iqr_test,
            ]
        )

        # Val FPR — computed from val BENIGN scores only (no leakage)
        val_fpr = np.array([(val_scores > t).mean() for t in val_candidate_values])

        test_fpr = np.array([(ae_mse_benign > t).mean() for t in test_candidate_values])
        test_tpr = np.array([(ae_mse_attack > t).mean() for t in test_candidate_values])
        test_youden = test_tpr - test_fpr

        # Store all val thresholds as dict — user selects manually after inspecting table
        self.ae_threshold = dict(zip(candidate_names, val_candidate_values.tolist()))

        lines = ["\nThreshold Analysis [Val Set — BENIGN only]:"]
        lines.append(f"{'Name':<14} {'Threshold':<14} {'Val FPR':<12}")
        lines.append("-" * 42)
        for name, thresh, fpr_p in zip(candidate_names, val_candidate_values, val_fpr):
            lines.append(f"{name:<14} {thresh:<14.6f} {fpr_p*100:<12.2f}%")
        self.log.info("\n".join(lines))

        lines = ["\nThreshold Analysis [Test Set — evaluation only]:"]
        lines.append(
            f"{'Name':<14} {'Threshold':<14} {'FPR':<10} {'TPR':<10} {'Youden':<10}"
        )
        lines.append("-" * 58)
        for name, thresh, fpr_p, tpr_p, youden in zip(
            candidate_names, test_candidate_values, test_fpr, test_tpr, test_youden
        ):
            lines.append(
                f"{name:<14} {thresh:<14.4f} {fpr_p*100:<10.2f}% {tpr_p*100:<10.2f}% {youden:<10.4f}"
            )
        self.log.info("\n".join(lines))

        attack_mask = self.test_labels == 1
        attack_scores = self.ae_mse_scores[attack_mask]
        attack_labels = self.test_labels_orig[attack_mask].reset_index(drop=True)

        for name, threshold in self.ae_threshold.items():
            lines = [
                f"\nPer-class TPR @ val threshold={threshold:.4f} ({name}):",
                "=" * 55,
            ]
            for label in sorted(attack_labels.unique()):
                mask = attack_labels.values == label
                tpr = (attack_scores[mask] > threshold).mean()
                lines.append(f"  {label:<35} TPR={tpr:.4f} ({mask.sum():>6,} samples)")
            lines.append("=" * 55)
            self.log.info("\n".join(lines))

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

    def build_isolation_forest(self) -> None:
        """
        Trains an Isolation Forest on the AE latent space of benign_train only.
        Must run after train_autoencoder() (encoder weights fixed from here on).
        """
        self.log.info("Building Isolation Forest on LSTM AE latent space...")

        train_seqs = _make_per_flow_sequences(
            self.benign_train, self.benign_train_scaled, self.config.window_size
        )
        latent_train = self._encode_latent(train_seqs)

        self.log.info(
            f"Latent vectors for training: {latent_train.shape[0]:,} samples, "
            f"dim={latent_train.shape[1]}"
        )

        self.iso_forest = IsolationForest(
            n_estimators=self.config.iso_n_estimators,
            random_state=self.config.iso_random_state,
            n_jobs=-1,
        )
        self.iso_forest.fit(latent_train)

        self.log.info("Isolation Forest training completed")

    def predict_isolation_forest(self) -> None:
        """
        Calibrates the IF threshold on benign_val (never on test, matching the AE
        threshold calibration in predict_autoencoder), scores the test set, and
        OR-combines with the AE decision using the same val-calibrated AE threshold
        predict_autoencoder() already produced — no threshold is recomputed from
        test-set statistics, to avoid leaking test labels into either threshold.
        """
        if self.iso_forest is None:
            raise TrainingError("Need to run build_isolation_forest() first")
        if (
            self.ae_mse_scores is None
            or self.ae_threshold is None
            or self.test_labels is None
        ):
            raise TrainingError("Need to run predict_autoencoder() first")

        ref_name = self.config.ae_reference_threshold_name
        if ref_name not in self.ae_threshold:
            raise TrainingError(
                f"ae_reference_threshold_name={ref_name!r} not among calibrated "
                f"AE thresholds: {list(self.ae_threshold.keys())}"
            )
        ae_threshold_ref = self.ae_threshold[ref_name]

        val_seqs = _make_per_flow_sequences(
            self.benign_val, self.benign_val_scaled, self.config.window_size
        )
        latent_val = self._encode_latent(val_seqs)
        iso_scores_val = self.iso_forest.score_samples(
            latent_val
        )  # lower = more anomalous
        iso_threshold = float(
            np.percentile(iso_scores_val, self.config.iso_threshold_percentile)
        )

        self.log.info(
            f"Isolation Forest threshold (val set, "
            f"p{self.config.iso_threshold_percentile}): {iso_threshold:.6f}"
        )

        test_seqs = _make_per_flow_sequences(
            self.test_df, self.test_features_scaled, self.config.window_size
        )
        latent_test = self._encode_latent(test_seqs)
        self.iso_scores = self.iso_forest.score_samples(latent_test)

        is_anomaly_iso = self.iso_scores < iso_threshold
        is_anomaly_ae = self.ae_mse_scores > ae_threshold_ref

        combined = is_anomaly_ae | is_anomaly_iso
        y_true = self.test_labels.values.astype(bool)

        ae_missed = y_true & (~is_anomaly_ae)
        iso_catches_ae_missed = int((ae_missed & is_anomaly_iso).sum())

        iso_missed = y_true & (~is_anomaly_iso)
        ae_catches_iso_missed = int((iso_missed & is_anomaly_ae).sum())

        def _rates(pred: np.ndarray) -> Tuple[float, float]:
            tp = (pred & y_true).sum()
            fp = (pred & ~y_true).sum()
            tpr = tp / y_true.sum() if y_true.sum() > 0 else float("nan")
            fpr = fp / (~y_true).sum() if (~y_true).sum() > 0 else float("nan")
            return tpr, fpr

        tpr_ae, fpr_ae = _rates(is_anomaly_ae)
        tpr_iso, fpr_iso = _rates(is_anomaly_iso)
        tpr_combined, fpr_combined = _rates(combined)

        self.log.info(
            "\n".join(
                [
                    f"\nAE ({ref_name}, val-calibrated) vs Isolation Forest vs Combined:",
                    f"{'':<12} {'TPR':<10} {'FPR':<10}",
                    f"{'AE only':<12} {tpr_ae:<10.4f} {fpr_ae:<10.4f}",
                    f"{'IF only':<12} {tpr_iso:<10.4f} {fpr_iso:<10.4f}",
                    f"{'Combined':<12} {tpr_combined:<10.4f} {fpr_combined:<10.4f}",
                    "",
                    f"The number of attack samples missed by AE but captured by IF: {iso_catches_ae_missed:,} "
                    f"(AE missed attacks: {int(ae_missed.sum()):,})",
                    f"IF missed, but AE caught attacks: {ae_catches_iso_missed:,} "
                    f"(IF missed attacks: {int(iso_missed.sum()):,})",
                ]
            )
        )

        self.iso_threshold = iso_threshold
        self.combined_anomaly_mask = combined

    def bootstrap_metrics(
        self,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        random_state: int = 42,
    ) -> None:
        """
        Bootstrap resampling for self.ae_mse_scores / self.test_labels.
        Does not retrain the model, only performs repeated sampling on the existing test set predictions.
        """
        if self.ae_mse_scores is None or self.test_labels is None:
            raise TrainingError(
                "Need to run predict_autoencoder() first to have scores for bootstrap"
            )

        y_true = self.test_labels.values
        y_score = self.ae_mse_scores
        n = len(y_true)

        rng = np.random.default_rng(random_state)
        alpha = (1 - ci) / 2

        auc_samples = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            y_t, y_s = y_true[idx], y_score[idx]

            if len(np.unique(y_t)) < 2:
                continue

            auc_samples.append(roc_auc_score(y_t, y_s))

        if len(auc_samples) == 0:
            self.log.warning(
                "Bootstrap resampling failed: no valid resamples with both classes present."
            )
            return

        auc_samples = np.array(auc_samples)
        point_auc = roc_auc_score(y_true, y_score)
        lower = float(np.percentile(auc_samples, 100 * alpha))
        upper = float(np.percentile(auc_samples, 100 * (1 - alpha)))

        self.log.info(
            f"\nBootstrap ({len(auc_samples)}/{n_bootstrap} valid resamples, "
            f"{ci*100:.0f}% CI):\n"
            f"  AUC: {point_auc:.4f}  [{lower:.4f}, {upper:.4f}]"
        )

    def _ae_predict_mse(self, sequences: np.ndarray) -> np.ndarray:
        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        n = len(sequences)
        mse_scores = np.zeros(n, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n, self.config.inference_batch_size):
                end = min(start + self.config.inference_batch_size, n)
                batch = torch.FloatTensor(sequences[start:end]).to(self.device)
                recon = self.lightning_module(batch).cpu().numpy()
                mse_scores[start:end] = np.mean(
                    np.square(sequences[start:end] - recon), axis=(1, 2)
                )

        return mse_scores

    def _encode_latent(self, sequences: np.ndarray) -> np.ndarray:
        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        n = len(sequences)
        latents = []
        with torch.no_grad():
            for start in range(0, n, self.config.inference_batch_size):
                end = min(start + self.config.inference_batch_size, n)
                batch = torch.FloatTensor(sequences[start:end]).to(self.device)
                z = self.lightning_module.model.encode(batch).cpu().numpy()
                latents.append(z)
        return np.concatenate(latents, axis=0)

    def save_results(self) -> None:
        self.log.info("Saving results...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./artifacts", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        attack_mask = self.test_labels.values == 1

        output = pd.DataFrame(
            self.test_df[self._feature_cols].values[attack_mask],
            columns=self._feature_cols,
        )
        output["ae_anomaly_score"] = self.ae_mse_scores[attack_mask]
        if self.iso_scores is not None:
            output["iso_anomaly_score"] = self.iso_scores[attack_mask]
        if self.combined_anomaly_mask is not None:
            output["combined_is_anomaly"] = self.combined_anomaly_mask[attack_mask]
        output["Label"] = self.test_labels_orig[attack_mask].values

        output_path = Path("outputs") / "deep_ae_scores.csv"
        output.to_csv(output_path, index=False)
        self.log.info(
            f"Saved: {output_path} ({len(output):,} rows, {output.shape[1]} columns)"
        )

        self.log.info(
            f"\nAE val thresholds saved to pkl: {list(self.ae_threshold.keys())}"
        )

        model_ae_path = Path("artifacts") / "deep_autoencoder.pt"
        torch.save(
            {
                "model_state_dict": self.autoencoder_model.state_dict(),
                "input_dim": self.autoencoder_model.input_dim,
                "hidden_size": self.autoencoder_model.hidden_size,
                "num_layers": self.autoencoder_model.num_layers,
                "encoding_dim": self.autoencoder_model.encoding_dim,
                "dropout": self.config.dropout,
                "window_size": self.config.window_size,
                "inference_batch_size": self.config.inference_batch_size,
            },
            model_ae_path,
        )
        self.log.info(f"Saved: {model_ae_path}")

        if self.iso_forest is not None:
            iso_forest_path = Path("artifacts") / "isolation_forest.joblib"
            joblib.dump(self.iso_forest, iso_forest_path)
            self.log.info(f"Saved: {iso_forest_path}")

        config_data = {
            "scaler": self.scaler,
            "clip_params": self.clip_params,
            "encoding_dim": self.config.encoding_dim,
            "window_size": self.config.window_size,
            "inference_batch_size": self.config.inference_batch_size,
            "feature_names": self._feature_cols,
            "ae_thresholds": self.ae_threshold,
            "log_transform_features": self.log_transform_features,
            "ae_reference_threshold_name": self.config.ae_reference_threshold_name,
            "iso_threshold": self.iso_threshold,
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
        ax.set_xlabel("LSTM AE MSE Score")
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
        ax.set_ylabel("LSTM AE MSE Score")
        ax.set_title("Score Distribution by Class (Test Set)")
        ax.grid(alpha=0.3)

        ax = axes[2]
        attack_labels = self.test_labels_orig[self.test_labels == 1]
        attack_scores = self.ae_mse_scores[self.test_labels == 1]

        type_means: dict = {}
        for label in sorted(attack_labels.unique()):
            mask = attack_labels.values == label
            type_means[label[:20]] = attack_scores[mask].mean()

        if type_means:
            sorted_types = sorted(type_means.items(), key=lambda x: x[1], reverse=True)
            names = [t[0] for t in sorted_types[:10]]
            values = [t[1] for t in sorted_types[:10]]
            ax.barh(names, values, color="coral")
            ax.set_xlabel("Mean LSTM AE MSE Score")
            ax.set_title("Top 10 Attack Types by AE Score")
            ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plot_path = Path("plots") / f"deep_ae_analysis-{self.datestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()

        self._plot_roc_pr()
        self._plot_score_cdf()
        self._plot_latent_tsne()

    def _plot_roc_pr(self) -> None:
        fpr, tpr, thresholds = roc_curve(self.test_labels, self.ae_mse_scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(
            self.test_labels, self.ae_mse_scores
        )
        ap = average_precision_score(self.test_labels, self.ae_mse_scores)

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
        ax.set_title("ROC Curve (LSTM AE)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(recall, precision, color="steelblue", lw=2, label=f"AP = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (LSTM AE)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = Path("plots") / f"deep_ae_roc_pr-{self.datestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path} (AUC={roc_auc:.4f}, AP={ap:.4f})")
        plt.close()

    def _plot_score_cdf(self) -> None:
        normal_scores = self.ae_mse_scores[self.test_labels == 0]
        attack_scores = self.ae_mse_scores[self.test_labels == 1]

        fig, ax = plt.subplots(figsize=(12, 6))
        for scores, label, color in [
            (normal_scores, "Normal", "green"),
            (attack_scores, "Attack", "red"),
        ]:
            sorted_s = np.sort(scores)
            cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
            ax.plot(sorted_s, cdf, label=label, color=color, lw=2)

        thresholds: Dict[str, float] = self.ae_threshold or {}
        if thresholds:
            cmap = plt.cm.get_cmap("tab20", len(thresholds))
            for i, (name, thresh) in enumerate(thresholds.items()):
                ax.axvline(
                    thresh,
                    color=cmap(i),
                    linestyle="--",
                    lw=1.2,
                    alpha=0.85,
                    label=f"{name} = {thresh:.4f}",
                )

        ax.set_xlabel("LSTM AE MSE Score")
        ax.set_ylabel("CDF")
        ax.set_title("Cumulative Distribution of Anomaly Scores")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = Path("plots") / f"deep_ae_cdf-{self.datestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()

    def _plot_latent_tsne(self, n_samples: int = 5000) -> None:
        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        # Use per-flow sequences so each sample has a latent vector
        per_flow_seqs = _make_per_flow_sequences(
            self.test_df,
            self.test_features_scaled,
            self.config.window_size,
        )

        total = len(per_flow_seqs)
        n_samples = min(n_samples, total)
        rng = np.random.default_rng(42)
        idx = rng.choice(total, n_samples, replace=False)

        x_sample = torch.FloatTensor(per_flow_seqs[idx]).to(self.device)
        labels_sample = self.test_labels_orig.iloc[idx].values

        with torch.no_grad():
            latent = self.lightning_module.model.encode(x_sample).cpu().numpy()

        self.log.info(
            f"Running t-SNE on {n_samples} samples "
            f"(latent dim={latent.shape[1]})..."
        )
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
        ax.set_title(f"t-SNE of LSTM AE Latent Space (n={n_samples})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        plot_path = Path("plots") / f"deep_ae_tsne-{self.datestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()

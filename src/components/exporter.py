import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import ujson

from model import UNIFIED_FEATURE_NAMES, ExportConfig, ExportError, LoadingConfigFailed
from utils import Logger

from .deep_autoencoder import LSTMAutoencoderModel


class Exporter:
    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.deep_ae_model: Optional[LSTMAutoencoderModel] = None

        self.ae_scaler: Optional[Any] = None
        self.ae_clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.ae_thresholds: Optional[Dict[str, float]] = None
        self.feature_names: Optional[List[str]] = None
        self.log_transform_features: Optional[List[str]] = None
        self.encoding_dim: Optional[int] = None
        self.window_size: Optional[int] = None
        self.inference_batch_size: Optional[int] = None

        self.deep_ae_onnx_path: Optional[Path] = None

        self.full_config: Optional[Dict[str, Any]] = None
        self.inference_config: Optional[Dict[str, Any]] = None

        self.config = config or ExportConfig()
        self.log = Logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.deep_ae_model is not None:
            self.deep_ae_model.cpu()
            self.deep_ae_model = None
        self.ae_scaler = None
        self.full_config = None
        self.inference_config = None
        return False

    def load_models(self) -> None:
        self.log.info("Loading LSTM AE model and configuration...")

        os.makedirs("./exports", exist_ok=True)
        model_path = Path("artifacts")

        try:
            ae_path = model_path / "deep_autoencoder.pt"
            checkpoint = torch.load(
                ae_path, map_location=self.device, weights_only=False
            )

            self.deep_ae_model = LSTMAutoencoderModel(
                input_dim=checkpoint["input_dim"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                encoding_dim=checkpoint["encoding_dim"],
                dropout=checkpoint.get("dropout", 0.0),
            )
            self.deep_ae_model.load_state_dict(checkpoint["model_state_dict"])
            self.deep_ae_model.eval()
            self.window_size = checkpoint.get("window_size", 10)
            self.log.info(
                f"LSTM Deep Autoencoder loaded "
                f"(input_dim={checkpoint['input_dim']}, "
                f"window_size={self.window_size})"
            )
        except Exception as e:
            self.log.error(f"Failed to load LSTM Deep Autoencoder: {e}")
            raise LoadingConfigFailed(f"Failed to load LSTM Deep Autoencoder: {e}")

        try:
            ae_config_path = model_path / "deep_ae_config.pkl"
            ae_config = joblib.load(ae_config_path)
            self.ae_scaler = ae_config["scaler"]
            self.ae_clip_params = ae_config["clip_params"]
            self.encoding_dim = ae_config.get("encoding_dim", 32)
            self.feature_names = ae_config.get("feature_names", UNIFIED_FEATURE_NAMES)
            self.window_size = ae_config.get("window_size", self.window_size or 10)
            self.inference_batch_size = ae_config.get("inference_batch_size", 1024)
            self.ae_thresholds = ae_config.get("ae_thresholds", {})
            self.log_transform_features = ae_config.get("log_transform_features", [])

            self.log.info(
                f"AE config loaded ({len(self.feature_names)} features, "
                f"thresholds={list(self.ae_thresholds.keys())}, "
                f"log1p_features={len(self.log_transform_features)})"
            )
        except Exception as e:
            self.log.error(f"Failed to load AE config: {e}")
            raise LoadingConfigFailed(f"Failed to load AE config: {e}")

    def export_deep_ae_onnx(self) -> None:
        self.log.info("Converting LSTM Deep Autoencoder to ONNX...")

        self.deep_ae_model.eval()
        input_dim = self.deep_ae_model.input_dim

        dummy_input = torch.randn(1, self.window_size, input_dim)

        self.deep_ae_onnx_path = Path("exports") / "deep_autoencoder.onnx"

        torch.onnx.export(
            self.deep_ae_model,
            (dummy_input,),
            self.deep_ae_onnx_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=False,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
            external_data=False,
        )

        self.log.info(f"Saved: {self.deep_ae_onnx_path}")

        onnx_model = onnx.load(self.deep_ae_onnx_path)
        onnx.checker.check_model(onnx_model)
        self.log.info("ONNX validation passed")

    def build_config_json(self) -> None:
        self.log.info("Building configuration JSON...")

        ae_scaler_params = {
            "mean": self.ae_scaler.mean_.tolist(),
            "std": self.ae_scaler.scale_.tolist(),
            "feature_names": self.feature_names,
        }

        ae_clip_params_json = {
            col: {"lower": float(p["lower"]), "upper": float(p["upper"])}
            for col, p in self.ae_clip_params.items()
        }

        self.full_config = {
            "created_at": pd.Timestamp.now().isoformat(),
            "framework": "PyTorch",
            "model": {
                "lstm_deep_autoencoder": {
                    "file": "deep_autoencoder.onnx",
                    "type": "LSTM Autoencoder",
                    "input_dim": int(self.deep_ae_model.input_dim),
                    "hidden_size": int(self.deep_ae_model.hidden_size),
                    "num_layers": int(self.deep_ae_model.num_layers),
                    "encoding_dim": int(self.encoding_dim),
                    "window_size": int(self.window_size),
                    "ae_feature_names": self.feature_names,
                    "ae_thresholds": self.ae_thresholds,
                },
            },
            "preprocessing": {
                # Applied first, in this order, before winsorize/scaler below.
                "ae_log_transform_features": self.log_transform_features,
                "ae_clip_params": ae_clip_params_json,
                "ae_scaler": ae_scaler_params,
                "post_scaling_clip": {
                    "min": self.config.post_scaling_clip_min,
                    "max": self.config.post_scaling_clip_max,
                },
            },
        }

        self.inference_config = {
            "ae_feature_names": self.feature_names,
            "ae_log_transform_features": self.log_transform_features,
            "ae_clip_params": ae_clip_params_json,
            "ae_scaler_mean": ae_scaler_params["mean"],
            "ae_scaler_std": ae_scaler_params["std"],
            "ae_post_clip_min": self.config.post_scaling_clip_min,
            "ae_post_clip_max": self.config.post_scaling_clip_max,
            "ae_thresholds": self.ae_thresholds,
            "window_size": int(self.window_size),
            "inference_batch_size": self.inference_batch_size,
        }

    def save_config_json(self) -> None:
        self.log.info("Saving configuration JSON...")

        os.makedirs("./exports", exist_ok=True)

        full_path = Path("exports") / "full_config.json"
        with open(full_path, "w", encoding="utf-8") as f:
            ujson.dump(self.full_config, f, indent=2, ensure_ascii=False)
        self.log.info(f"Saved: {full_path}")

        inference_path = Path("exports") / "inference_config.json"
        with open(inference_path, "w", encoding="utf-8") as f:
            ujson.dump(self.inference_config, f, indent=2, ensure_ascii=False)
        self.log.info(f"Saved: {inference_path}")

    def verify_onnx_models(self) -> None:
        self.log.info("Verifying LSTM AE ONNX model with end-to-end inference test...")

        input_dim = self.deep_ae_model.input_dim
        test_input = np.random.randn(1, self.window_size, input_dim).astype(np.float32)

        session_ae = ort.InferenceSession(str(self.deep_ae_onnx_path))
        ae_reconstructed = session_ae.run(None, {"input": test_input})[0]
        ae_mse = float(np.mean((test_input - ae_reconstructed) ** 2))

        self.log.info(
            f"LSTM Deep Autoencoder: "
            f"input={test_input.shape}, "
            f"output={ae_reconstructed.shape}, "
            f"MSE={ae_mse:.6f}"
        )

    def print_summary(self) -> None:
        self.log.info("Export Summary...")

        threshold_lines = [
            f"    {name}: {val:.6f}" for name, val in (self.ae_thresholds or {}).items()
        ]

        lines = (
            [
                "\nModel Information:",
                f"  Framework   : PyTorch (LSTM Autoencoder)",
                (
                    f"  LSTM AE     : input_dim={self.deep_ae_model.input_dim}, "
                    f"hidden={self.deep_ae_model.hidden_size}, "
                    f"layers={self.deep_ae_model.num_layers}, "
                    f"bottleneck={self.encoding_dim}, "
                    f"window={self.window_size}"
                ),
                f"  Features    : {len(self.feature_names)} flow features",
                f"  AE thresholds (val set):",
            ]
            + threshold_lines
            + [
                "",
                "Exported files:",
                f"  {self.deep_ae_onnx_path}",
                f"  exports/full_config.json",
                f"  exports/inference_config.json",
            ]
        )
        self.log.info("\n".join(lines))

    def verify_onnx_export(self) -> None:
        self.log.info("Verifying ONNX export correctness...")

        num_features = self.deep_ae_model.input_dim
        window_size = self.window_size

        sess = ort.InferenceSession(str(self.deep_ae_onnx_path))
        input_name = sess.get_inputs()[0].name
        self.deep_ae_model.eval()

        ort_outputs = []
        max_diffs = []

        for i in range(4):
            x = torch.randn(1, window_size, num_features)
            x_np = x.numpy().astype(np.float32)

            with torch.no_grad():
                pt_out = self.deep_ae_model(x).numpy()

            ort_out = sess.run(None, {input_name: x_np})[0]
            ort_outputs.append(ort_out)

            max_diff = float(np.abs(pt_out - ort_out).max())
            max_diffs.append(max_diff)

            if max_diff > self.config.verify_atol:
                raise ExportError(
                    f"[input {i}] PyTorch vs ONNX max diff {max_diff:.6f} exceeds atol {self.config.verify_atol:.6f}"
                )

        variation = max(
            np.abs(ort_outputs[0] - ort_outputs[j]).max()
            for j in range(1, len(ort_outputs))
        )
        if variation < 1e-6:
            raise ExportError(
                "ONNX outputs are identical across different inputs — "
                "decoder h_0 was likely constant-folded. Re-export with do_constant_folding=False."
            )

        self.log.info(
            f"ONNX verification passed | "
            f"inputs tested: {len(ort_outputs)} | "
            f"max PyTorch/ONNX diff: {max(max_diffs):.2e} | "
            f"output variation across inputs: {variation:.2e}"
        )
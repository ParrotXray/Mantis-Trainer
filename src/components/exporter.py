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

from model import UNIFIED_FEATURE_NAMES, ExportConfig
from utils import Logger

from .deep_autoencoder import LSTMAutoencoderModel


class Exporter:
    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.deep_ae_model: Optional[LSTMAutoencoderModel] = None

        self.ae_scaler: Optional[Any] = None
        self.ae_clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.ae_threshold: Optional[float] = None
        self.ae_threshold_method: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        self.encoding_dim: Optional[int] = None
        self.window_size: Optional[int] = None

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

        # ---- Load LSTM Deep Autoencoder ----
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
            raise

        # ---- Load AE config ----
        try:
            ae_config_path = model_path / "deep_ae_config.pkl"
            ae_config = joblib.load(ae_config_path)
            self.ae_scaler = ae_config["scaler"]
            self.ae_clip_params = ae_config["clip_params"]
            self.encoding_dim = ae_config.get("encoding_dim", 32)
            self.feature_names = ae_config.get("feature_names", UNIFIED_FEATURE_NAMES)
            self.ae_threshold = ae_config.get("ae_threshold", 0.08)
            self.ae_threshold_method = ae_config.get("ae_threshold_method", "mean+1std")
            self.window_size = ae_config.get("window_size", self.window_size or 10)
            self.log.info(
                f"AE config loaded ({len(self.feature_names)} features, "
                f"threshold={self.ae_threshold:.6f})"
            )
        except Exception as e:
            self.log.error(f"Failed to load AE config: {e}")
            raise

    def export_deep_ae_onnx(self) -> None:
        self.log.info("Converting LSTM Deep Autoencoder to ONNX...")

        self.deep_ae_model.eval()
        input_dim = self.deep_ae_model.input_dim

        # Dummy input: (batch=1, seq_len=window_size, input_dim)
        dummy_input = torch.randn(1, self.window_size, input_dim)

        self.deep_ae_onnx_path = Path("exports") / "deep_autoencoder.onnx"

        torch.onnx.export(
            self.deep_ae_model,
            (dummy_input,),
            self.deep_ae_onnx_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
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
                    "ae_threshold": self.ae_threshold,
                    "ae_threshold_method": self.ae_threshold_method,
                },
            },
            "preprocessing": {
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
            "ae_clip_params": ae_clip_params_json,
            "ae_scaler_mean": ae_scaler_params["mean"],
            "ae_scaler_std": ae_scaler_params["std"],
            "ae_post_clip_min": self.config.post_scaling_clip_min,
            "ae_post_clip_max": self.config.post_scaling_clip_max,
            "ae_threshold": self.ae_threshold,
            "ae_threshold_method": self.ae_threshold_method,
            "window_size": int(self.window_size),
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
        # Test with dynamic sequence length (window_size)
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

        lines = [
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
            f"  AE threshold: {self.ae_threshold:.6f}",
            "",
            "Exported files:",
            f"  {self.deep_ae_onnx_path}",
            f"  exports/full_config.json",
            f"  exports/inference_config.json",
        ]
        self.log.info("\n".join(lines))

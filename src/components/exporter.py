import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import ujson

from model import UNIFIED_FEATURE_NAMES, ExportConfig
from utils import Logger

from .classifier import ResNetMLP
from .deep_autoencoder import AutoencoderModel


class Exporter:
    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.deep_ae_model: Optional[AutoencoderModel] = None
        self.classifier_model: Optional[ResNetMLP] = None
        self.label_encoder: Optional[Any] = None

        self.ae_scaler: Optional[Any] = None
        self.ae_clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.ae_threshold: Optional[float] = None
        self.feature_names: Optional[List[str]] = None
        self.encoding_dim: Optional[int] = None

        self.deep_ae_onnx_path: Optional[Path] = None
        self.classifier_onnx_path: Optional[Path] = None

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

        if self.classifier_model is not None:
            self.classifier_model.cpu()
            self.classifier_model = None

        self.ae_scaler = None
        self.full_config = None
        self.inference_config = None
        return False

    def load_models(self) -> None:
        self.log.info("Loading models and configurations...")

        os.makedirs("./exports", exist_ok=True)
        model_path = Path("artifacts")

        # Load Deep Autoencoder
        try:
            ae_path = model_path / "deep_autoencoder.pt"
            checkpoint = torch.load(
                ae_path, map_location=self.device, weights_only=False
            )

            self.deep_ae_model = AutoencoderModel(
                input_dim=checkpoint["input_dim"],
                layer_sizes=checkpoint["layer_sizes"],
                encoding_dim=checkpoint["encoding_dim"],
                dropout_rates=checkpoint["dropout_rates"],
                l2_reg=checkpoint["l2_reg"],
            )
            self.deep_ae_model.load_state_dict(checkpoint["model_state_dict"])
            self.deep_ae_model.eval()
            self.log.info("Deep Autoencoder loaded")
        except Exception as e:
            self.log.error(f"Failed to load Deep Autoencoder: {e}")
            raise

        # Load AE config
        try:
            ae_config_path = model_path / "deep_ae_config.pkl"
            ae_config = joblib.load(ae_config_path)
            self.ae_scaler = ae_config["scaler"]
            self.ae_clip_params = ae_config["clip_params"]
            self.encoding_dim = ae_config.get("encoding_dim", 16)
            self.feature_names = ae_config.get("feature_names", UNIFIED_FEATURE_NAMES)
            self.ae_threshold = ae_config.get("ae_threshold", 0.08)
            self.log.info(f"AE config loaded ({len(self.feature_names)} features)")
        except Exception as e:
            self.log.error(f"Failed to load AE config: {e}")
            raise

        # Load ResNet MLP classifier
        try:
            cls_path = model_path / "classifier.pt"
            checkpoint = torch.load(
                cls_path, map_location=self.device, weights_only=False
            )

            self.classifier_model = ResNetMLP(
                input_dim=checkpoint["input_dim"],
                hidden_dims=checkpoint["hidden_dims"],
                n_classes=checkpoint["n_classes"],
                dropout=checkpoint["dropout"],
            )
            self.classifier_model.load_state_dict(checkpoint["model_state_dict"])
            self.classifier_model.eval()
            self.log.info("ResNet MLP classifier loaded")
        except Exception as e:
            self.log.error(f"Failed to load classifier: {e}")
            raise

        # Load classifier config
        try:
            cls_config_path = model_path / "classifier_config.pkl"
            cls_config = joblib.load(cls_config_path)
            self.label_encoder = cls_config["encoder"]
            self.log.info(
                f"Classifier config loaded "
                f"({len(self.label_encoder.classes_)} classes)"
            )
        except Exception as e:
            self.log.error(f"Failed to load classifier config: {e}")
            raise

    def export_deep_ae_onnx(self) -> None:
        self.log.info("Converting Deep Autoencoder to ONNX...")

        self.deep_ae_model.eval()
        input_dim = self.deep_ae_model.input_dim
        dummy_input = torch.randn(1, input_dim)

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

    def export_classifier_onnx(self) -> None:
        self.log.info("Converting ResNet MLP classifier to ONNX...")

        self.classifier_model.eval()
        input_dim = self.classifier_model.input_dim
        dummy_input = torch.randn(1, input_dim)

        wrapped = nn.Sequential(self.classifier_model, nn.Softmax(dim=1))
        wrapped.eval()

        self.classifier_onnx_path = Path("exports") / "classifier.onnx"

        torch.onnx.export(
            wrapped,
            (dummy_input,),
            self.classifier_onnx_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
            external_data=False,
        )

        self.log.info(f"Saved: {self.classifier_onnx_path}")

        loaded = onnx.load(self.classifier_onnx_path)
        onnx.checker.check_model(loaded)
        self.log.info("ONNX validation passed")

    def build_config_json(self) -> None:
        self.log.info("Building configuration JSON...")

        # AE scaler params
        ae_scaler_params = {
            "mean": self.ae_scaler.mean_.tolist(),
            "std": self.ae_scaler.scale_.tolist(),
            "feature_names": self.feature_names,
        }

        ae_clip_params_json = {
            col: {"lower": float(p["lower"]), "upper": float(p["upper"])}
            for col, p in self.ae_clip_params.items()
        }

        attack_labels = {
            str(i): label for i, label in enumerate(self.label_encoder.classes_)
        }

        n_features = self.classifier_model.input_dim

        self.full_config = {
            "created_at": pd.Timestamp.now().isoformat(),
            "framework": "PyTorch",
            "model": {
                "deep_autoencoder": {
                    "file": "deep_autoencoder.onnx",
                    "input_dim": int(self.deep_ae_model.input_dim),
                    "encoding_dim": int(self.encoding_dim),
                    "ae_feature_names": self.feature_names,
                    "ae_threshold": self.ae_threshold,
                },
                "classifier": {
                    "file": "classifier.onnx",
                    "type": "ResNet MLP",
                    "n_features": int(n_features),
                    "n_classes": int(len(self.label_encoder.classes_)),
                    "classifier_feature_names": self.feature_names
                    + ["ae_anomaly_score"],
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
            "attack_labels": attack_labels,
        }

        self.inference_config = {
            "ae_feature_names": self.feature_names,
            "ae_clip_params": ae_clip_params_json,
            "ae_scaler_mean": ae_scaler_params["mean"],
            "ae_scaler_std": ae_scaler_params["std"],
            "ae_post_clip_min": self.config.post_scaling_clip_min,
            "ae_post_clip_max": self.config.post_scaling_clip_max,
            "ae_threshold": self.ae_threshold,
            "classifier_feature_names": self.feature_names + ["ae_anomaly_score"],
            "attack_labels": attack_labels,
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
        self.log.info("Verifying ONNX models with end-to-end inference test...")

        n_ae_features = self.deep_ae_model.input_dim
        test_ae_input = np.random.randn(1, n_ae_features).astype(np.float32)

        # --- Step 1: AE ---
        session_ae = ort.InferenceSession(str(self.deep_ae_onnx_path))
        ae_reconstructed = session_ae.run(None, {"input": test_ae_input})[0]
        ae_mse = float(np.mean((test_ae_input - ae_reconstructed) ** 2))
        self.log.info(
            f"Deep Autoencoder: input={test_ae_input.shape}, MSE={ae_mse:.6f}"
        )

        # --- Step 2: append ae_anomaly_score → classifier input ---
        ae_anomaly_score = np.array([[ae_mse]], dtype=np.float32)
        cls_input = np.concatenate([test_ae_input, ae_anomaly_score], axis=1)

        # --- Step 3: Classifier ---
        session_cls = ort.InferenceSession(str(self.classifier_onnx_path))
        input_name = session_cls.get_inputs()[0].name
        cls_output = session_cls.run(None, {input_name: cls_input})

        # Output is logits; apply softmax for probabilities
        logits = cls_output[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        predicted_label = int(np.argmax(probs, axis=1)[0])
        class_name = self.label_encoder.classes_[predicted_label]

        self.log.info(
            f"\nClassifier: input={cls_input.shape}"
            f"\n  Predicted: {class_name}"
            f"\n  Probabilities shape: {probs.shape}"
        )

    def print_summary(self) -> None:
        self.log.info("Export Summary...")

        n_features = self.classifier_model.input_dim
        lines = [
            "\nModel Information:",
            f"  Framework: PyTorch (AE + Classifier)",
            (
                f"  Deep AE: {self.deep_ae_model.input_dim} dim -> "
                f"{self.encoding_dim} dim bottleneck "
                f"({len(self.feature_names)} features)"
            ),
            (
                f"  Classifier: ResNet MLP, "
                f"{len(self.label_encoder.classes_)} classes, "
                f"{n_features} features"
            ),
            "",
            "Attack Classes:",
        ]
        for i, cls in enumerate(self.label_encoder.classes_):
            lines.append(f"  {i}: {cls}")
        self.log.info("\n".join(lines))

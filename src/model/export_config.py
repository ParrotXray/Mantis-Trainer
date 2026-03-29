"""
Model Export Configuration Module.

This module defines the configuration dataclass for exporting trained
PyTorch models to ONNX format for deployment and inference optimization.
"""

from dataclasses import dataclass


@dataclass
class ExportConfig:
    """
    Configuration for ONNX Model Export.

    This configuration controls the parameters for converting trained
    PyTorch models to ONNX format, enabling cross-platform deployment
    and optimized inference.

    Attributes:
        Version Information:
            version: Semantic version string for the exported model.
            opset_version: ONNX opset version for compatibility (default: 18).

        Inference Parameters:
            post_scaling_clip_min: Minimum clipping value applied after scaling.
            post_scaling_clip_max: Maximum clipping value applied after scaling.
    """

    # Version Information
    version: str = "1.0.0"
    opset_version: int = 18

    # Inference Parameters
    post_scaling_clip_min: float = -5.0
    post_scaling_clip_max: float = 5.0

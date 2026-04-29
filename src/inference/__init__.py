"""Inference module."""
from .predictor import (
    Segmentor,
    PostProcessor,
    predict_with_postprocess,
    batch_predict_with_postprocess
)

__all__ = [
    "Segmentor",
    "PostProcessor",
    "predict_with_postprocess",
    "batch_predict_with_postprocess"
]
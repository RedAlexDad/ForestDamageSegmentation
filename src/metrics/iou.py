"""IoU (Intersection over Union) metric."""
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def iou_score(y_true, y_pred, smooth: float = 1e-6):
    """
    Compute IoU (Intersection over Union) score.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU score
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)


def iou_loss(y_true, y_pred):
    """IoU loss (1 - IoU score)."""
    return 1 - iou_score(y_true, y_pred)


def binary_iou(y_true, y_pred, threshold: float = 0.5):
    """Binary IoU with threshold."""
    y_pred_binary = K.cast(y_pred > threshold, K.floatx())
    return iou_score(y_true, y_pred_binary)


class MeanIoU(tf.keras.metrics.Metric):
    """Keras metric for mean Intersection over Union."""

    def __init__(self, name="mean_iou", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = K.cast(y_pred > self.threshold, K.floatx())

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred_binary)

        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)

        self.total_iou.assign_add(iou)
        self.count.assign_add(1)

    def result(self):
        return self.total_iou / self.count

    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


def compute_iou_batch(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute IoU for a batch."""
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    y_true = y_true.astype(np.float32)

    intersection = np.sum(y_true * y_pred_binary)
    union = np.sum(y_true) + np.sum(y_pred_binary) - intersection

    return (intersection + 1e-6) / (union + 1e-6)
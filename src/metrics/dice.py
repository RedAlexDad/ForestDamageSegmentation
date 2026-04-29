"""Dice coefficient metric."""
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    """
    Compute Dice coefficient.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def dice_coefficient_loss(y_true, y_pred):
    """Dice loss (1 - dice coefficient)."""
    return 1 - dice_coefficient(y_true, y_pred)


def binary_dice_coefficient(y_true, y_pred, threshold: float = 0.5):
    """
    Binary dice coefficient with threshold.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        threshold: Decision threshold

    Returns:
        Binary Dice coefficient
    """
    y_pred_binary = K.cast(y_pred > threshold, K.floatx())
    return dice_coefficient(y_true, y_pred_binary)


class DiceMetric(tf.keras.metrics.Metric):
    """Keras metric for Dice coefficient."""

    def __init__(self, name="dice", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = K.cast(y_pred > self.threshold, K.floatx())
        dice = dice_coefficient(y_true, y_pred_binary)

        self.total.assign_add(dice)
        self.count.assign_add(1)

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class IoUMetric(tf.keras.metrics.Metric):
    """Keras metric for Intersection over Union."""

    def __init__(self, name="iou", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = K.cast(y_pred > self.threshold, K.floatx())
        intersection = K.sum(y_true * y_pred_binary)
        union = K.sum(y_true) + K.sum(y_pred_binary) - intersection

        iou = intersection / (union + 1e-6)

        self.total.assign_add(iou)
        self.count.assign_add(1)

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def compute_dice_numpy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute dice using numpy arrays."""
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_binary)
    return (2. * intersection) / (
        np.sum(y_true) + np.sum(y_pred_binary) + 1e-6
    )


def compute_iou_numpy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute IoU using numpy arrays."""
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_binary)
    union = np.sum(y_true) + np.sum(y_pred_binary) - intersection
    return intersection / (union + 1e-6)
"""Tversky loss for imbalanced segmentation."""
import tensorflow.keras.backend as K
import tensorflow as tf


def tversky_loss(
    y_true,
    y_pred,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1.0
):
    """
    Tversky loss for handling class imbalance.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Weight for false positives (default: 0.5)
        beta: Weight for false negatives (default: 0.5)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Tversky loss value
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)

    return 1 - (true_pos + smooth) / (
        true_pos + alpha * false_neg + beta * false_pos + smooth
    )


def tversky_focal_loss(
    y_true,
    y_pred,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
    smooth: float = 1.0
):
    """Combined Tversky and Focal loss."""
    tversky = tversky_loss(y_true, y_pred, alpha, beta, smooth)

    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    focal = K.pow(1 - y_pred, gamma)

    return tversky + focal


def focal_tversky_loss(
    y_true,
    y_pred,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    smooth: float = 1.0
):
    """
    Focal Tversky loss - focuses on hard examples.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: False positive penalty (higher = more penalty for FP)
        beta: False negative penalty (higher = more penalty for FN)
        gamma: Focusing parameter
        smooth: Smoothing factor

    Returns:
        Focal Tversky loss
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)

    pt = (true_pos + smooth) / (
        true_pos + alpha * false_neg + beta * false_pos + smooth
    )

    return K.pow(1 - pt, gamma)


class TverskyLoss(tf.keras.layers.Layer):
    """Tversky loss layer."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, name="tversky", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        y_true, y_pred = inputs
        return tversky_loss(y_true, y_pred, self.alpha, self.beta)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta
        })
        return config
"""Loss functions for segmentation."""
import tensorflow.keras.backend as K
import tensorflow as tf


def dice_loss(y_true, y_pred, smooth: float = 1.0):
    """
    Dice loss for binary segmentation.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor

    Returns:
        Dice loss value
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return 1 - (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def focal_loss(y_true, y_pred, alpha: float = 0.25, gamma: float = 2.0):
    """
    Focal loss for handling class imbalance.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Weighting factor
        gamma: Focusing parameter

    Returns:
        Focal loss value
    """
    y_true = K.cast(y_true, tf.float32)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)

    focal_loss = -alpha_t * tf.pow(1 - pt, gamma) * K.log(pt)

    return K.mean(focal_loss)


def combined_loss(
    y_true,
    y_pred,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5
):
    """
    Combined BCE and Dice loss.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        bce_weight: Weight for BCE
        dice_weight: Weight for Dice

    Returns:
        Combined loss value
    """
    from tensorflow.keras.losses import binary_crossentropy

    bce = binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return bce_weight * bce + dice_weight * dice


def tversky_loss(
    y_true,
    y_pred,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1.0
):
    """
    Tversky loss - generalization of Dice for imbalanced data.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: False positive penalty
        beta: False negative penalty
        smooth: Smoothing factor

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


def lovasz_grad(gt_sorted):
    """Compute gradient of Lovasz extension."""
    gts = K.sum(gt_sorted)
    intersection = gt_sorted
    union = gts + gt_sorted - intersection
    jaccard = 1.0 - intersection / (union + 1e-10)

    if len(gt_sorted) > 1:
        jaccard -= 1.0 + 1e-10

    return jaccard


def lovasz_hinge(y_true, y_pred):
    """Lovasz-hinge loss."""
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    signs = 2.0 * y_true - 1.0
    errors = K.abs(y_true - y_pred)

    errors_sorted, perm = tf.math.top_k(errors, k=tf.shape(errors)[0])
    gt_sorted = tf.gather(y_true, perm)

    grad = lovasz_grad(gt_sorted)
    loss = K.dot(errors_sorted, grad)

    return loss


class DiceLoss(tf.keras.losses.Loss):
    """Dice loss as Keras loss."""

    def __init__(self, smooth: float = 1.0, name="dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        return dice_loss(y_true, y_pred, self.smooth)


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss as Keras loss."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, self.alpha, self.gamma)


class TverskyLoss(tf.keras.losses.Loss):
    """Tversky loss as Keras loss."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, name="tversky_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        return tversky_loss(y_true, y_pred, self.alpha, self.beta)
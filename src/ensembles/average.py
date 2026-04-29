"""Ensemble methods for model combination."""
import numpy as np
from tensorflow.keras.models import Model, load_model
import tensorflow as tf


class AverageEnsemble:
    """Average ensemble of models."""

    def __init__(self, models: list = None):
        self.models = models or []

    def add_model(self, model: Model):
        """Add model to ensemble."""
        self.models.append(model)

    def predict(self, images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict using average of all models."""
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        for model in self.models:
            pred = model.predict(images, verbose=0)
            predictions.append(pred)

        avg_pred = np.mean(predictions, axis=0)
        return (avg_pred > threshold).astype(np.float32)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        predictions = [model.predict(images, verbose=0) for model in self.models]
        return np.mean(predictions, axis=0)


class WeightedAverageEnsemble:
    """Weighted average ensemble."""

    def __init__(self, models: list = None, weights: list = None):
        self.models = models or []
        self.weights = weights or []

        if self.models and not self.weights:
            self.weights = [1.0 / len(self.models)] * len(self.models)

        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def add_model(self, model: Model, weight: float = 1.0):
        """Add model with weight."""
        self.models.append(model)
        self.weights.append(weight)
        self._normalize_weights()

    def predict(self, images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict using weighted average."""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(images, verbose=0)
            predictions.append(pred * weight)

        avg_pred = sum(predictions)
        return (avg_pred > threshold).astype(np.float32)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(images, verbose=0)
            predictions.append(pred * weight)

        return sum(predictions)


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""

    def __init__(self, base_models: list = None, meta_model: Model = None):
        self.base_models = base_models or []
        self.meta_model = meta_model
        self.fitted = False

    def add_base_model(self, model: Model):
        """Add base model."""
        self.base_models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10):
        """Fit meta-model on base model predictions."""
        if not self.base_models or not self.meta_model:
            raise ValueError("Need base models and meta model")

        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X, verbose=0)
            base_predictions.append(pred)

        meta_features = np.concatenate(base_predictions, axis=-1)

        self.meta_model.fit(meta_features, y, epochs=epochs)
        self.fitted = True

    def predict(self, images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict using stacking."""
        if not self.fitted:
            raise ValueError("Meta-model not fitted")

        base_predictions = []
        for model in self.base_models:
            pred = model.predict(images, verbose=0)
            base_predictions.append(pred)

        meta_features = np.concatenate(base_predictions, axis=-1)
        predictions = self.meta_model.predict(meta_features, verbose=0)

        return (predictions > threshold).astype(np.float32)


def create_ensemble_from_paths(
    model_paths: list,
    weights: list = None,
    ensemble_type: str = "average"
) -> tuple:
    """Create ensemble from saved model paths."""
    models = []
    for path in model_paths:
        model = load_model(path)
        models.append(model)

    if ensemble_type == "average":
        return AverageEnsemble(models)
    elif ensemble_type == "weighted":
        return WeightedAverageEnsemble(models, weights)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def evaluate_ensemble(ensemble, images: np.ndarray, masks: np.ndarray):
    """Evaluate ensemble performance."""
    predictions = ensemble.predict(images)

    from src.metrics.dice import compute_dice_numpy
    from src.metrics.iou import compute_iou_batch

    dice_scores = []
    iou_scores = []

    for i in range(len(images)):
        dice = compute_dice_numpy(masks[i], predictions[i])
        iou = compute_iou_batch(masks[i], predictions[i])
        dice_scores.append(dice)
        iou_scores.append(iou)

    return {
        "dice_mean": np.mean(dice_scores),
        "dice_std": np.std(dice_scores),
        "iou_mean": np.mean(iou_scores),
        "iou_std": np.std(iou_scores)
    }
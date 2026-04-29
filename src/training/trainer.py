"""Model trainer with MLflow integration."""
import os
import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    CSVLogger, TensorBoard, LearningRateScheduler
)
from tensorflow.keras.models import Model
import tensorflow as tf


class ModelTrainer:
    """Trainer class for segmentation models."""

    def __init__(
        self,
        model: Model,
        config: dict = None,
        logger=None,
        experiment_name: str = "forest-segmentation"
    ):
        self.model = model
        self.config = config or {}
        self.logger = logger
        self.experiment_name = experiment_name
        self.history = None

    def get_callbacks(
        self,
        checkpoint_path: str = "checkpoints/best_model.h5",
        patience: int = 10,
        reduce_patience: int = 5,
        verbose: int = 1
    ):
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=reduce_patience,
                verbose=verbose
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=verbose
            ),
            CSVLogger("logs/training.log")
        ]

        if os.path.exists("logs/tensorboard"):
            callbacks.append(TensorBoard("logs/tensorboard"))

        return callbacks

    def train(
        self,
        train_dataset,
        val_dataset,
        epochs: int = 50,
        callbacks: list = None,
        verbose: int = 1
    ):
        """Train the model."""
        if self.logger:
            self.logger.log_params(self.config)

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks or self.get_callbacks(),
            verbose=verbose
        )

        if self.logger:
            self._log_metrics()

        return self.history

    def evaluate(self, test_dataset):
        """Evaluate on test data."""
        results = self.model.evaluate(test_dataset, verbose=0)
        return dict(zip(self.model.metrics_names, results))

    def predict(self, images):
        """Predict on images."""
        return self.model.predict(images, verbose=0)

    def _log_metrics(self):
        """Log metrics to MLflow."""
        if not self.logger or not self.history:
            return

        history = self.history.history
        for metric_name in history:
            for epoch, value in enumerate(history[metric_name]):
                self.logger.log_metrics({metric_name: value}, step=epoch)

    def save(self, path: str):
        """Save model."""
        self.model.save(path)

    def load(self, path: str):
        """Load model."""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)


def train_with_generator(
    model: Model,
    train_gen,
    val_gen,
    epochs: int = 50,
    steps_per_epoch: int = None,
    validation_steps: int = None,
    config: dict = None,
    logger=None
):
    """Train model using generator."""
    trainer = ModelTrainer(model, config, logger)
    callbacks = trainer.get_callbacks()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    return history


def evaluate_batch(model: Model, images: np.ndarray, masks: np.ndarray):
    """Evaluate on batch data."""
    predictions = model.predict(images, verbose=0)

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


def plot_training_curves(history):
    """Plot training curves from history."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history.get("loss", []), label="train")
    axes[0].plot(history.history.get("val_loss", []), label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Metrics
    if "dice" in history.history:
        axes[1].plot(history.history.get("dice", []), label="train_dice")
        axes[1].plot(history.history.get("val_dice", []), label="val_dice")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Dice Coefficient")
    axes[1].legend()

    plt.tight_layout()
    return fig
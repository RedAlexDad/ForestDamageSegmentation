"""Predictor for inference."""
import numpy as np
from tensorflow.keras.models import Model, load_model
import cv2


class Segmentor:
    """Predictor for forest damage segmentation."""

    def __init__(self, model: Model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    @classmethod
    def from_path(cls, model_path: str, threshold: float = 0.5):
        """Load model from path."""
        model = load_model(model_path)
        return cls(model, threshold)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict mask for single image."""
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        pred = self.model.predict(image, verbose=0)[0]
        return (pred > self.threshold).astype(np.float32)

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Predict masks for batch."""
        predictions = self.model.predict(images, verbose=0)
        return (predictions > self.threshold).astype(np.float32)

    def predict_with_confidence(self, image: np.ndarray) -> tuple:
        """Predict with confidence scores."""
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        pred = self.model.predict(image, verbose=0)[0]
        mask = (pred > self.threshold).astype(np.float32)
        confidence = np.max(pred)

        return mask, confidence


class PostProcessor:
    """Post-processing for predictions."""

    @staticmethod
    def apply_morphology(
        mask: np.ndarray,
        kernel_size: int = 5,
        operation: str = "open"
    ) -> np.ndarray:
        """Apply morphological operations."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == "open":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == "erode":
            mask = cv2.erode(mask, kernel)
        elif operation == "dilate":
            mask = cv2.dilate(mask, kernel)

        return mask

    @staticmethod
    def remove_small_regions(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
        """Remove small connected regions."""
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered_mask[labels == i] = 1

        return filtered_mask.astype(np.float32)

    @staticmethod
    def smooth_boundaries(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Smooth mask boundaries."""
        kernel = cv2.getGaussianKernel(kernel_size, -1)
        mask = cv2.filter2D(mask, -1, kernel)
        return (mask > 0.5).astype(np.float32)

    @staticmethod
    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """Fill holes in mask."""
        mask = mask.astype(np.uint8)

        h, w = mask.shape
        filled = mask.copy()

        mask_border = np.zeros((h + 2, w + 2), np.uint8)
        mask_border[1:h+1, 1:w+1] = mask

        cv2.floodFill(filled, mask_border, (0, 0), 1)

        return filled[1:h+1, 1:w+1].astype(np.float32)


def predict_with_postprocess(
    model: Model,
    image: np.ndarray,
    threshold: float = 0.5,
    use_morphology: bool = True,
    min_region_size: int = 100
) -> np.ndarray:
    """Predict with post-processing."""
    predictor = Segmentor(model, threshold)

    mask = predictor.predict(image)

    if use_morphology:
        mask = PostProcessor.apply_morphology(mask)

    if min_region_size > 0:
        mask = PostProcessor.remove_small_regions(mask, min_region_size)

    return mask


def batch_predict_with_postprocess(
    model: Model,
    images: np.ndarray,
    threshold: float = 0.5,
    use_morphology: bool = True,
    min_region_size: int = 100,
    batch_size: int = 8
) -> list:
    """Batch predict with post-processing."""
    predictor = Segmentor(model, threshold)
    results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        masks = predictor.predict_batch(batch)

        for mask in masks:
            if use_morphology:
                mask = PostProcessor.apply_morphology(mask)

            if min_region_size > 0:
                mask = PostProcessor.remove_small_regions(mask, min_region_size)

            results.append(mask)

    return results
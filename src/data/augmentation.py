"""Augmentation pipeline using imgaug."""
import numpy as np
import cv2
import imgaug.augmenters as iaa


def get_training_augmentor() -> iaa.Sequential:
    """Get augmentation pipeline for training."""
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.3),
        iaa.Affine(
            rotate=(-15, 15),
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1)
        ),
        iaa.OneOf([
            iaa.GaussBlur((0, 3)),
            iaa.MedianBlur(k=(3, 7))
        ]),
        iaa.AdditiveGaussianNoise(
            scale=(0, 0.05 * 255),
            per_channel=True
        ),
        iaa.LinearContrast((0.8, 1.2), per_channel=True),
        iaa.Grayscale(alpha=(0.0, 0.3)),
    ])


def get_light_augmentor() -> iaa.Sequential:
    """Get lighter augmentation for validation."""
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-5, 5)),
    ])


def apply_augmentation(images: np.ndarray, masks: np.ndarray, aug) -> tuple:
    """Apply augmentation to batch."""
    if images.ndim == 4:
        batch_size = len(images)
        images_aug = []
        masks_aug = []

        for i in range(batch_size):
            img = images[i]
            mask = masks[i]

            if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] <= 3):
                img_mask = np.concatenate([img, mask[:, :, np.newaxis]], axis=2)
                augmented = aug(image=img_mask)
                img = augmented[:, :, :-1]
                mask = augmented[:, :, -1]
            else:
                augmented = aug(image=img, mask=mask)
                img = augmented[0]
                mask = augmented[1]

            images_aug.append(img)
            masks_aug.append(mask)

        return np.array(images_aug), np.array(masks_aug)
    else:
        augmented = aug(image=images, mask=masks)
        return augmented[0], augmented[1]


def random_flip(image: np.ndarray, mask: np.ndarray) -> tuple:
    """Random horizontal and vertical flip."""
    if np.random.random() > 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    if np.random.random() > 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()

    return image, mask


def random_rotation(image: np.ndarray, mask: np.ndarray, max_angle: int = 15) -> tuple:
    """Random rotation."""
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    if image.ndim == 3:
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    else:
        image = cv2.warpAffine(image, M, (w, h))

    return image, mask


def random_brightness(image: np.ndarray, mask: np.ndarray, factor: float = 0.2) -> tuple:
    """Random brightness adjustment."""
    delta = np.random.uniform(-factor, factor)
    image = np.clip(image + delta, 0, 1)
    return image, mask


def random_contrast(image: np.ndarray, mask: np.ndarray, factor: float = 0.2) -> tuple:
    """Random contrast adjustment."""
    alpha = np.random.uniform(1 - factor, 1 + factor)
    image = np.clip(image * alpha, 0, 1)
    return image, mask


def elastic_transform(image: np.ndarray, mask: np.ndarray, alpha: float = 30, sigma: float = 5) -> tuple:
    """Elastic deformation (simplified)."""
    shape = image.shape[:2]
    
    random_state = np.random.RandomState(None)
    dx = cv2.GaussianBlur(
        (random_state.rand(*shape) * 2 - 1),
        (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (random_state.rand(*shape) * 2 - 1),
        (0, 0), sigma
    ) * alpha

    x, y_mesh = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices_y = np.reshape(y_mesh + dy, (-1, 1))
    indices_x = np.reshape(x + dx, (-1, 1))

    if image.ndim == 3:
        image_map = image.reshape((*shape, -1))
        image = cv2.remap(image_map, indices_x, indices_y, cv2.INTER_LINEAR)
        image = image.reshape(shape + (image.shape[-1],))
    else:
        image = cv2.remap(image, indices_x, indices_y, cv2.INTER_LINEAR)
        image = image.reshape(shape)

    mask_map = mask.reshape((*shape, -1))
    mask = cv2.remap(mask_map, indices_x, indices_y, cv2.INTER_LINEAR)
    mask = mask.reshape(shape)

    return image, mask
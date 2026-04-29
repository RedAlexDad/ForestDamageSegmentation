"""Dataset classes for training."""
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence


class SatelliteDataset(Sequence):
    """Keras Sequence for satellite imagery."""

    MASK_LABELS = [
        '-', 'Сплошная рубка', 'Проходная рубка',
        'Лесная дорога', 'Ветровал', 'Пожар',
        'Усыхание', 'Выборочная рубка'
    ]

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        batch_size: int = 8,
        shuffle: bool = True,
        augmentations: callable = None,
        preprocessing: callable = None
    ):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.indices = np.arange(len(self.images))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        batch_masks = []

        for i in batch_indices:
            image = self.images[i]
            mask = self.masks[i]

            if self.augmentations:
                augmented = self.augmentations(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            if self.preprocessing:
                image = self.preprocessing(image)

            batch_images.append(image)
            batch_masks.append(mask)

        return (
            np.array(batch_images),
            np.array(batch_masks)
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class SimpleDataset:
    """Simple numpy dataset without Keras Sequence."""

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        batch_size: int = 8
    ):
        self.images = images.astype(np.float32)
        self.masks = masks.astype(np.float32)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration

        start = self._idx * self.batch_size
        end = min(start + self.batch_size, len(self.images))

        self._idx += 1

        return self.images[start:end], self.masks[start:end]

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.images))
        return self.images[start:end], self.masks[start:end]


def create_datasets(
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 8,
    val_split: float = 0.2,
    seed: int = 42
) -> tuple:
    """Create train and validation datasets."""
    np.random.seed(seed)
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * val_split)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_images = images[train_indices]
    train_masks = masks[train_indices]
    val_images = images[val_indices]
    val_masks = masks[val_indices]

    train_ds = SimpleDataset(train_images, train_masks, batch_size)
    val_ds = SimpleDataset(val_images, val_masks, batch_size)

    return train_ds, val_ds


def load_image(path: str, normalize: bool = True) -> np.ndarray:
    """Load single image from path."""
    from skimage import io
    img = io.imread(path)
    if normalize:
        img = img / 65536.0
    return img


def save_image(path: str, img: np.ndarray):
    """Save image to path."""
    from skimage import io
    if img.max() <= 1.0:
        img = (img * 65535).astype(np.uint16)
    io.imsave(path, img)


def tensor_to_image(tensor: np.ndarray) -> np.ndarray:
    """Convert model output tensor to displayable image."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).astype(np.uint8)
    return tensor


def image_to_tensor(image: np.ndarray) -> np.ndarray:
    """Convert image to model input tensor."""
    if image.max() > 1.0:
        image = image / 255.0
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    return image
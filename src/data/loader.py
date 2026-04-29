"""Data loading utilities for satellite imagery."""
import os
import glob
import numpy as np
from skimage import io


class SatelliteTileLoader:
    """Loader for satellite tiles from directory."""

    CHANNEL_NEW = list(range(1, 14))
    CHANNEL_OLD = list(range(14, 27))
    CHANNEL_MASK = 0

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.tile_paths = []

    def load_tiles(self, pattern: str = "*with_object.tif") -> list:
        """Load all tile paths matching pattern."""
        search_path = os.path.join(self.data_dir, "*", pattern)
        self.tile_paths = sorted(glob.glob(search_path))
        return self.tile_paths

    def load_tile(self, path: str, normalize: bool = True) -> np.ndarray:
        """Load single tile as numpy array."""
        img = io.imread(path)
        if normalize:
            img = img / 65536.0
        return img

    def get_rgb_new(self, img: np.ndarray) -> np.ndarray:
        """Extract RGB from new image (channels 4,3,2 -> B,G,R)."""
        c1, c2, c3 = 4, 3, 2
        stack = np.stack([
            img[:, :, c1] / (img[:, :, c1].max() + 1e-8),
            img[:, :, c2] / (img[:, :, c2].max() + 1e-8),
            img[:, :, c3] / (img[:, :, c3].max() + 1e-8)
        ], axis=2)
        return np.clip(stack, 0, 1)

    def get_rgb_old(self, img: np.ndarray) -> np.ndarray:
        """Extract RGB from old image (channels 17,16,15 -> B,G,R)."""
        c1, c2, c3 = 17, 16, 15
        stack = np.stack([
            img[:, :, c1] / (img[:, :, c1].max() + 1e-8),
            img[:, :, c2] / (img[:, :, c2].max() + 1e-8),
            img[:, :, c3] / (img[:, :, c3].max() + 1e-8)
        ], axis=2)
        return np.clip(stack, 0, 1)

    def get_mask(self, img: np.ndarray) -> np.ndarray:
        """Extract binary mask (channel 0)."""
        return (img[:, :, 0] > 0).astype(np.float32)

    def get_diff_channels(self, img: np.ndarray, n_channels: int = 6) -> np.ndarray:
        """Get difference channels between old and new images."""
        new_img = img[:, :, 1:14]
        old_img = img[:, :, 14:27]
        diff = np.abs(new_img - old_img)
        if n_channels <= 6:
            diff = diff[:, :, :n_channels]
        return diff

    def get_all_channels(self, img: np.ndarray, n_channels: int = 6) -> np.ndarray:
        """Get all available channels (old + new + diff)."""
        new_img = img[:, :, 1:14]
        old_img = img[:, :, 14:27]
        diff = np.abs(new_img - old_img)

        if n_channels == 6:
            channels = np.concatenate([
                new_img[:, :, :3],
                old_img[:, :, :3]
            ], axis=2)
        elif n_channels == 13:
            channels = np.concatenate([
                new_img,
                old_img
            ], axis=2)
        else:
            channels = diff

        return channels


def load_train_data(data_dir: str, n_channels: int = 6) -> tuple:
    """Load all training data as arrays."""
    loader = SatelliteTileLoader(data_dir)
    tile_paths = loader.load_tiles()

    images = []
    masks = []

    for path in tile_paths:
        img = loader.load_tile(path, normalize=True)

        if n_channels == 6:
            x = loader.get_all_channels(img, n_channels=6)
        else:
            x = img[:, :, 1:14]

        mask = loader.get_mask(img)

        images.append(x)
        masks.append(mask)

    return np.array(images), np.array(masks)


def split_train_val(
    images: np.ndarray,
    masks: np.ndarray,
    val_split: float = 0.2,
    seed: int = 42
) -> tuple:
    """Split data into train and validation sets."""
    np.random.seed(seed)
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * val_split)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return (
        images[train_indices],
        masks[train_indices],
        images[val_indices],
        masks[val_indices]
    )
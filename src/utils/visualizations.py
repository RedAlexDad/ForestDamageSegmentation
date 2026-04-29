"""Visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import matplotlib.patches as mpatches


def visualize_prediction(
    image: np.ndarray,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    title: str = None
) -> plt.Figure:
    """Visualize prediction results."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(mask_true, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(mask_pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    overlay = np.zeros((*mask_true.shape, 3))
    overlay[:, :, 1] = mask_pred
    overlay[:, :, 0] = mask_true
    overlay[:, :, 2] = mask_true & mask_pred
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def visualize_mask(
    mask: np.ndarray,
    labels: list = None,
    title: str = "Mask"
) -> plt.Figure:
    """Visualize mask with labels."""
    if labels is None:
        labels = ['-', 'Clear cut', 'Selection cut', 'Forest road',
                 'Windthrow', 'Fire', 'Drying', 'Selection']

    my_colors = ['g', 'b', 'r', 'black', 'c', 'm', 'y', 'black']
    cmap = mpl_colors.ListedColormap(my_colors)

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = mpl_colors.BoundaryNorm(bounds, cmap.N)

    values = np.unique(mask.ravel()).astype(int)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(mask, cmap=cmap, norm=norm)

    patches = [mpatches.Patch(color=my_colors[values[i]],
               label=f"{values[i]}: {labels[values[i]]}")
              for i in range(len(values))]

    ax.legend(loc="lower right", handles=patches, bbox_to_anchor=(1, 1))
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    return fig


def plot_training_history(history) -> plt.Figure:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    if "loss" in history.history:
        axes[0].plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Dice
    if "dice" in history.history:
        axes[1].plot(history.history["dice"], label="train_dice")
    if "val_dice" in history.history:
        axes[1].plot(history.history["val_dice"], label="val_dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Dice Coefficient")
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_comparison(results: dict, metric: str = "dice") -> plt.Figure:
    """Plot comparison of results."""
    names = list(results.keys())
    values = [results[n].get(f"{metric}_mean", 0) for n in names]
    stds = [results[n].get(f"{metric}_std", 0) for n in names]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(names, values, yerr=stds, capsize=5)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Comparison")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig
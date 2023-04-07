import os
import os.path as path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches

from ._scroll import Scroll
from .patches import export_patches, get_patches


def make_patch_dataset(
    scroll: Scroll,
    patch_size: int,
    num_patches: int,
    holdout_region: Tuple[float, float, float, float],
    export: Optional[str] = None,
    show: bool = False,
    seed: int = 0,
    train_frac: float = 0.7,
):
    """Make patch data set.

    Args:
        scroll: The scroll to pull patches from.
        patch_size: The edge size of the square patches.
        num_patches: The total number of patches to sample.
        holdout_region: A (x, y, w, h) region of the image
            to avoid sampling from, in fractions from 0 to 1.
        export: Path to data export directory.
        show: If True, visualize the patches.
        seed: The random seed for sampling.
        train_frac: The fraction of patches to use for training.
    """
    patch_splits = get_patches(
        scroll,
        patch_size=patch_size,
        holdout_region=holdout_region,
        num_patches=num_patches,
        seed=seed,
        train_frac=train_frac,
    )

    if show:
        _, axs = plt.subplots(ncols=2, dpi=150)
        axs[0].imshow(scroll.load(num_slices=1)[0], cmap="gray")
        axs[1].imshow(scroll.ink_labels, cmap="gray")
        for ax in axs:
            for patch in patch_splits.train:
                ax.add_patch(
                    patches.Rectangle(
                        (patch.left, patch.top),
                        patch.width,
                        patch.height,
                        linewidth=1,
                        edgecolor="red",
                        facecolor="none",
                        alpha=0.8,
                    )
                )
            for patch in patch_splits.val:
                ax.add_patch(
                    patches.Rectangle(
                        (patch.left, patch.top),
                        patch.width,
                        patch.height,
                        linewidth=1,
                        edgecolor="blue",
                        facecolor="none",
                        alpha=0.8,
                    )
                )
            for patch in patch_splits.test:
                ax.add_patch(
                    patches.Rectangle(
                        (patch.left, patch.top),
                        patch.width,
                        patch.height,
                        linewidth=1,
                        edgecolor="yellow",
                        facecolor="none",
                        alpha=0.8,
                    )
                )
            ax.axis("off")
        plt.show()

    if export is not None:
        if not path.exists(export):
            os.makedirs(export)
        export_patches(scroll, patch_splits, export)

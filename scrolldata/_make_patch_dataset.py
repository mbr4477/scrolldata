from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import patches

from ._scroll import Scroll
from .patches import export_patches, get_patches


def make_patch_dataset(
    scroll: Scroll,
    patch_size: int,
    num_patches: int,
    holdout_region: Tuple[float, float, float, float],
    export: bool = False,
    show: bool = False,
    seed: int = 0,
    train_frac: float = 0.7,
):
    """Make patch data set."""
    patch_splits = get_patches(
        scroll,
        patch_size=patch_size,
        holdout_region=holdout_region,
        num_patches=num_patches,
        seed=seed,
        train_frac=train_frac,
    )

    if export:
        export_patches(scroll, patch_splits, ".")

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

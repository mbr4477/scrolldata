import logging
import os
import os.path as path
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from . import LOGGER_NAME
from ._scroll import Scroll

logger = logging.getLogger(LOGGER_NAME)


@dataclass
class Patch:
    """A patch.

    Attrs:
        top: The patch top index.
        left: The patch left index.
        height: The height of the patch.
        width: The width of the path.
    """

    top: int
    left: int
    height: int
    width: int


@dataclass
class PatchSplits:
    """A dataclass to store the train/val/test data splits.

    Attrs:
        train: The training patches.
        val: The validation patches.
        test: The testing patches.
    """

    train: Sequence[Patch]
    val: Sequence[Patch]
    test: Sequence[Patch]


def get_patches(
    scroll: Scroll,
    patch_size: int,
    holdout_region: Tuple[float, float, float, float],
    num_patches: int,
    seed: int = 0,
    train_frac: float = 0.7,
) -> PatchSplits:
    """Randomly sample a data set of square patches.

    Args:
        scroll: The Scroll to sample patches from.
        patch_size: The width/height of the square patches.
        holdout_region: A region to avoid when sampling patches.
            (x, y, width, height) in fractions of the input (0 to 1).
        num_patches: The total number of patches to sample
            for train/val/test.
        seed: The random seed.
        train_frac: The fraction of patches to use as training data.

    Returns:
        The PatchSplits.
    """
    mask = scroll.mask

    holdout_mask = np.zeros_like(mask)
    holdout_idx_space = [
        int(x * mask.shape[(i + 1) % 2]) for i, x in enumerate(holdout_region)
    ]
    holdout_mask[
        holdout_idx_space[1] : holdout_idx_space[1] + holdout_idx_space[3],
        holdout_idx_space[0] : holdout_idx_space[0] + holdout_idx_space[2],
    ] = 1

    train_mask = mask & (~holdout_mask)
    selected_patches = np.zeros_like(train_mask)

    def is_inside_train_mask(i: int, j: int) -> bool:
        return (train_mask[i : i + patch_size, j : j + patch_size] == 1).all()

    def is_not_overlapping_with_other_patch(i: int, j: int) -> bool:
        return (selected_patches[i : i + patch_size, j : j + patch_size] == 0).all()

    def is_valid_training_patch(i: int, j: int) -> bool:
        return is_inside_train_mask(i, j)

    possible_i, possible_j = np.where(train_mask)
    possible_patch_indices = list(zip(possible_i.tolist(), possible_j.tolist()))

    patches = []
    logger.info(f"Sampling {num_patches} patches")
    MAX_ATTEMPTS = 1024
    random.seed(seed)
    for _ in tqdm(range(num_patches)):
        done = False
        attempts = 0
        while not done and attempts < MAX_ATTEMPTS:
            attempts += 1
            i, j = random.choice(possible_patch_indices)
            if is_valid_training_patch(i, j):
                patches.append(Patch(i, j, patch_size, patch_size))
                selected_patches[i : i + patch_size, j : j + patch_size] = 1
                done = True
        if attempts >= MAX_ATTEMPTS:
            raise RuntimeError(
                f"Failed to sample a valid patch in {MAX_ATTEMPTS} attempts"
            )

    # Divide into training, validation, and testing
    indices = np.arange(len(patches))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_len = int(len(patches) * train_frac)
    val_len = int(len(patches) * (1 - train_frac) / 2)
    train_idx = indices[:train_len]
    val_idx = indices[train_len : train_len + val_len]
    test_idx = indices[-val_len:]

    return PatchSplits(
        [patches[i] for i in train_idx],
        [patches[i] for i in val_idx],
        [patches[i] for i in test_idx],
    )


def export_patches(scroll: Scroll, patch_splits: PatchSplits, dir: str):
    """Export the patches to disk for efficient future loading.
    The patch splits describe the coordinates of patches to extract
    from the given scroll.

    Inside the given directory, the function creates `train`, `val`, and `test` folders.
    Each folder contains one subfolder per example which itself contains
    `inputs.npy` and `targets.npy` numpy files.

    Args:
        scroll: The scroll to use.
        patch_splits: The train/val/test patches.
        dir: The directory to export the data to.
    """

    def load_splits(
        *splits: Sequence[Patch],
    ) -> Tuple[Sequence[Sequence[np.ndarray]], Sequence[Sequence[np.ndarray]]]:
        """A utility function to load in all the data splits from the scorll data."""
        splits_data: List[List[np.ndarray]] = []
        splits_labels: List[List[np.ndarray]] = []
        for slice_i in tqdm(range(scroll.num_slices)):
            slice_data = scroll.load(start_slice=slice_i, num_slices=1)
            for split_i in range(len(splits)):
                if slice_i == 0:
                    splits_data.append([])

                for patch_i, patch in enumerate(splits[split_i]):
                    patch_slice = slice_data[
                        :,
                        patch.top : patch.top + patch.height,
                        patch.left : patch.left + patch.width,
                    ]
                    if slice_i == 0:
                        splits_data[split_i].append(patch_slice)
                    else:
                        splits_data[split_i][patch_i] = np.concatenate(
                            (splits_data[split_i][patch_i], patch_slice)
                        )

        ink_labels = scroll.ink_labels
        for split_i in range(len(splits)):
            splits_labels.append([])

            for patch_i, patch in enumerate(splits[split_i]):
                label_slice = ink_labels[
                    patch.top : patch.top + patch.height,
                    patch.left : patch.left + patch.width,
                ].reshape(1, patch.height, patch.width)
                splits_labels[split_i].append(label_slice)

        return splits_data, splits_labels

    def export_to_folder(name: str, x: Sequence[np.ndarray], y: Sequence[np.ndarray]):
        """A utility function for writing a subset to disk."""
        # Create a folder for each split
        folder = path.join(dir, name)
        logger.info(f"Writing {len(x)} {name} patches to disk ...")
        for i in tqdm(range(len(x))):
            folder_name = path.join(folder, f"{i:05}")
            os.makedirs(folder_name, exist_ok=True)
            inputs_name = path.join(folder_name, "inputs.npy")
            targets_name = path.join(folder_name, "targets.npy")
            np.save(inputs_name, x[i])
            np.save(targets_name, y[i])

    (train_x, val_x, test_x), (train_y, val_y, test_y) = load_splits(
        patch_splits.train, patch_splits.val, patch_splits.test
    )
    export_to_folder("train", train_x, train_y)
    del train_x, train_y

    export_to_folder("val", val_x, val_y)
    del val_x, val_y

    export_to_folder("test", test_x, test_y)
    del test_x, test_y

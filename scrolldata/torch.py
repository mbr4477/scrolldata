import glob
from os import path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """A lazy loaded torch patch data set.
    Expects a folder containing a subfolder for each example.
    Each subfolder should contain an `inputs.npy` and `targets.npy`
    numpy array file.
    
    Example:

    ```python
    dataset = PatchDataset("./dataset/train")
    item = dataset[0]
    inputs = item["inputs"]
    targets = item["targets"]
    ```
    """

    def __init__(self, dir: str):
        """
        Args:
            dir: The root directory of the data set.
        """
        super().__init__()
        self._patch_folders = sorted(glob.glob(path.join(dir, "*")))
        print(self._patch_folders[:10])

    def __len__(self) -> int:
        """Returns the length of the data set."""
        return len(self._patch_folders)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns a single example as a dictionary of inputs and targets.
        
        Returns:
            Dictionary with `inputs` and `targets` keys.
        """
        assert type(idx) == int, "Slicing not supported!"
        folder = self._patch_folders[idx]
        inputs_file = path.join(folder, "inputs.npy")
        targets_file = path.join(folder, "targets.npy")
        return {
            "inputs": torch.tensor(np.load(inputs_file).astype(np.float32)),
            "targets": torch.tensor(np.load(targets_file).astype(int)),
        }

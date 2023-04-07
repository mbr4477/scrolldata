import numpy as np


def f05_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute the F0.5 score.
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview/evaluation

    Args:
        predicted: The NxHxW batch of binary predictions as integers.
        actual: The NxHxW batch of binary ground truth predictions as integers.

    Returns:
        The mean-aggregated F0.5 score across the batch.
    """
    batch_size = predicted.shape[0]
    true_pos = ((predicted == 1) & (actual == 1)).reshape(batch_size, -1).sum(-1)
    false_pos = ((predicted == 1) & (actual == 0)).reshape(batch_size, -1).sum(-1)
    false_neg = ((predicted == 0) & (actual == 1)).reshape(batch_size, -1).sum(-1)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    return ((1.25 * precision * recall) / (0.25 * precision + recall)).item()


def run_length_encoding(x: np.ndarray) -> str:
    """Run-length coding based on
    https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0
    and
    https://www.kaggle.com/code/hackerpoet/even-faster-run-length-encoder/script

    Adapted to fix off-by-one errors and C-style instead of Fortran-style encoding.

    Args:
        x: The HxW input image of probabilities from 0 to 1 per pixel.

    Returns:
        The encoding str.
    """
    flat_img = x.flatten('F')
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    # Find the starts using "rising edges"
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    starts = np.append(np.array([flat_img[0] == 1]), starts)

    # Find the ends using "falling edges"
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    ends = np.append(ends, np.array([flat_img[-1] == 1]))

    starts_ix = np.where(starts)[0] + 1
    ends_ix = np.where(ends)[0] + 1

    lengths = ends_ix - starts_ix + 1

    # Create the run length encoding str
    encoded = ""
    for start, length in zip(starts_ix, lengths):
        encoded += f"{start} {length} "

    return encoded.strip()

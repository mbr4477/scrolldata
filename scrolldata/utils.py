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

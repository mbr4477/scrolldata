from scrolldata.utils import f05_score
import numpy as np


def test_f05_score():
    cases = [
        {
            "preds": np.array([[[0, 0, 0, 0], [1, 1, 1, 1]]]),
            "truth": np.array([[[0, 1, 0, 1], [1, 0, 1, 0]]]),
            "expected": 0.5,
        },
        {
            "preds": np.array([[[1, 1, 1, 1], [1, 1, 1, 1]]]),
            "truth": np.array([[[0, 1, 0, 1], [1, 0, 1, 0]]]),
            "expected": 5.0 / 9.0,
        },
        {
            "preds": np.array([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
            "truth": np.array([[[0, 1, 0, 1], [1, 0, 1, 0]]]),
            "expected": None,
        },
    ]
    for case in cases:
        out = f05_score(case["preds"], case["truth"])
        if case["expected"] is None:
            assert np.isnan(out)
        else:
            assert out == case["expected"]

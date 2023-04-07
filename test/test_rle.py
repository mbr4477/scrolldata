from scrolldata.utils import run_length_encoding
import numpy as np


def test_rle():
    cases = [
        {
            "inputs": np.array(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                ]
            ),
            "expected": "5 4",
        },
        {
            "inputs": np.array(
                [
                    [1, 1, 1, 0],
                    [0, 1, 1, 1],
                ]
            ),
            "expected": "1 3 6 3",
        },
        {
            "inputs": np.array(
                [
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                ]
            ),
            "expected": "1 1 3 1 5 1 7 1",
        },
    ]
    for case in cases:
        out = run_length_encoding(case["inputs"])
        assert out == case["expected"]

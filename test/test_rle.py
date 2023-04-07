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
            "expected": "2 1 4 1 6 1 8 1",
        },
        {
            "inputs": np.array(
                [
                    [1, 1, 1, 0],
                    [0, 1, 1, 1],
                ]
            ),
            "expected": "1 1 3 4 8 1",
        },
        {
            "inputs": np.array(
                [
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                ]
            ),
            "expected": "1 2 5 2",
        },
    ]
    for case in cases:
        out = run_length_encoding(case["inputs"])
        assert out == case["expected"]

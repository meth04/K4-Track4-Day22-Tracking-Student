"""Kiểm tra phát hiện kết quả MOT sai mà không sử dụng dữ liệu thật."""

import numpy as np
import pytest

from validate_submission import validate_rows


def test_valid_sparse_frames():
    validate_rows(np.array([[1, 1, 0, 0, 10, 20, .8, -1, -1, -1],
                            [3, 1, 1, 1, 10, 20, .7, -1, -1, -1]]), 5)


@pytest.mark.parametrize('column,value', [(0, 0), (0, 6), (1, 1.5), (4, 0), (5, -1), (6, 1.1), (7, 0), (2, float('nan'))])
def test_invalid_values_fail(column, value):
    row = np.array([[1, 1, 0, 0, 10, 20, .8, -1, -1, -1]], dtype=float)
    row[0, column] = value
    with pytest.raises(ValueError):
        validate_rows(row, 5)


def test_duplicate_id_same_frame_fails():
    row = np.array([[1, 1, 0, 0, 10, 20, .8, -1, -1, -1]])
    with pytest.raises(ValueError):
        validate_rows(np.vstack([row, row]), 5)

"""Kiểm tra lọc cache và dấu vân tay file mà không cần mô hình hoặc dữ liệu lab."""

import hashlib
import numpy as np

from lab_experiments import file_sha256, filter_detections


def test_confidence_filter_strict_and_no_mutation():
    detections = np.array([[1, 2, 3, 4, .15, 0], [5, 6, 7, 8, .3, 0], [2, 3, 4, 5, .8, 0]])
    result = filter_detections(detections, .3)
    assert result.shape == (1, 6)
    assert result[0, 4] == .8
    result[0, 0] = 999
    assert detections[2, 0] == 2


def test_empty_detections():
    assert filter_detections(np.empty((0, 6)), .3).shape == (0, 6)


def test_sha256_matches_content(tmp_path):
    target = tmp_path / 'result.txt'
    target.write_bytes(b'ket qua thuc te')
    assert file_sha256(target) == hashlib.sha256(b'ket qua thuc te').hexdigest()

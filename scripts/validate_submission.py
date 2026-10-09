"""Kiểm tra file MOT, số frame đã xử lý và checksum trước khi nộp GitHub."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from lab_experiments import ROOT, file_sha256


def validate_rows(rows: np.ndarray, frames: int) -> None:
    """Kiểm tra các điều kiện hợp lệ của file MOT.

    Args:
        rows: Mảng kết quả MOT.
        frames: Số frame trong chuỗi ảnh nguồn.

    Raises:
        ValueError: Khi sai định dạng, trùng ID hoặc hộp không hợp lệ.
    """
    if rows.ndim != 2 or rows.shape[1] != 10 or not len(rows):
        raise ValueError('File MOT phải có dữ liệu và đúng 10 cột.')
    if not np.isfinite(rows).all():
        raise ValueError('File MOT chứa NaN hoặc vô cực.')
    if not np.equal(rows[:, :2], np.floor(rows[:, :2])).all():
        raise ValueError('Frame và ID phải là số nguyên.')
    if (rows[:, 0] < 1).any() or (rows[:, 0] > frames).any() or (rows[:, 1] < 1).any():
        raise ValueError('Frame hoặc ID nằm ngoài phạm vi.')
    if len(np.unique(rows[:, :2], axis=0)) != len(rows):
        raise ValueError('Trùng ID trong cùng một frame.')
    if (rows[:, 4:6] <= 0).any() or (rows[:, 6] < 0).any() or (rows[:, 6] > 1).any():
        raise ValueError('Kích thước hộp hoặc confidence không hợp lệ.')
    if not (rows[:, 7:] == -1).all():
        raise ValueError('Ba cột cuối phải là -1.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab-data-root', type=Path, default=ROOT / 'data')
    arguments = parser.parse_args()
    results = []
    for index in range(1, 6):
        video = f'video_{index}'
        expected = len(list((arguments.lab_data_root / video / 'img1').glob('*.jpg')))
        target = ROOT / 'submission/results' / f'{video}.txt'
        metadata = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
        data = np.loadtxt(target, delimiter=',', ndmin=2)
        validate_rows(data, expected)
        assert metadata['frames_processed'] == expected and metadata['full_sequence']
        assert file_sha256(target) == metadata['result_sha256']
        assert target.read_bytes() == (ROOT / 'runs/nop_bai' / target.name).read_bytes()
        record = {'video': video, 'frames_processed': expected, 'rows': len(data),
                  'sha256': file_sha256(target), 'passed': True}
        results.append(record)
        print(f'{video}: hợp lệ, xử lý đủ {expected} frame, {len(data)} dòng.')
    (ROOT / 'evidence/validation.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

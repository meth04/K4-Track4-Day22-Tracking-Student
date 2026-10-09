"""Thử nghiệm có lưu cache detector, giữ nguyên mô hình và cấu hình tracker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
dependency_marker = ROOT / '.cache/deps/.python-version'
if (dependency_marker.exists() and dependency_marker.read_text().strip()
        == f'{sys.version_info.major}.{sys.version_info.minor}'):
    sys.path.insert(0, str(ROOT / '.cache/deps'))
sys.path.insert(0, str(ROOT.parent / 'TrackEval'))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import cv2
import numpy as np


def filter_detections(detections: np.ndarray, conf: float) -> np.ndarray:
    """Lọc confidence từ đầu ra detector đã kiểm tra tương đương.

    Args:
        detections: Mảng hộp người N x 6.
        conf: Ngưỡng confidence của detector.

    Returns:
        Bản sao các hộp có confidence lớn hơn ngưỡng.
    """
    return detections[detections[:, 4] > conf].copy()


def file_sha256(path: Path) -> str:
    """Tính SHA-256 của một file.

    Args:
        path: Đường dẫn file cần kiểm tra.

    Returns:
        Chuỗi SHA-256 dạng hex.
    """
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_detector(video: str, device: str) -> None:
    """Lưu hộp từng frame và kiểm tra cache với suy luận trực tiếp.

    Args:
        video: Tên video trong thư mục data.
        device: Thiết bị PyTorch dùng chạy detector.

    Raises:
        RuntimeError: Khi cache không tương đương với suy luận trực tiếp.
    """
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    model = YOLO(ROOT / 'yolo26n.pt')
    paths = sorted((ROOT / 'data' / video / 'img1').glob('*.jpg'))
    target = ROOT / '.cache/detections' / f'{video}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = {}
    started = time.perf_counter()
    checks = []
    sample_indices = set(np.linspace(0, len(paths) - 1, 12, dtype=int).tolist())
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f'Không đọc được ảnh {path}')
        for iou in (.4, .5, .7):
            result = model.predict(frame, conf=.15, iou=iou, imgsz=640,
                                   classes=[0], device=device, verbose=False)[0]
            det = result.boxes.data.cpu().numpy().copy()
            frames[f'f{index + 1}_i{int(iou*10)}'] = det
            if index in sample_indices:
                for conf in (.3, .5):
                    direct = model.predict(frame, conf=conf, iou=iou, imgsz=640,
                                           classes=[0], device=device, verbose=False)[0]
                    expected = filter_detections(det, conf)
                    actual = direct.boxes.data.cpu().numpy()
                    passed = expected.shape == actual.shape and np.allclose(expected, actual, atol=1e-5)
                    checks.append({'frame': index + 1, 'conf': conf, 'iou': iou, 'passed': bool(passed)})
                    if not passed:
                        raise RuntimeError(f'Cache khác suy luận trực tiếp: {video}, frame {index+1}')
        if (index + 1) % 100 == 0:
            print(f'[detector] {video}: {index+1}/{len(paths)}', flush=True)
    np.savez_compressed(target, **frames)
    manifest = {'video': video, 'frames': len(paths), 'detector': 'yolo26n.pt',
                'detector_sha256': file_sha256(ROOT / 'yolo26n.pt'), 'imgsz': 640,
                'classes': [0], 'half': False, 'device': device,
                'end2end': bool(model.model.model[-1].end2end),
                'seconds': time.perf_counter() - started, 'equivalence_checks': checks,
                'cache_sha256': file_sha256(target)}
    (ROOT / 'evidence' / f'{video}_detector.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[detector] {video}: đã lưu {len(paths)} frame', flush=True)


def track(video: str, tracker_name: str, conf: float, iou: float,
          limit: int, output: Path, device: str) -> None:
    """Chạy tracker mặc định trên các hộp detector đã lưu.

    Args:
        video: Tên chuỗi ảnh.
        tracker_name: Tên tracker được lab cho phép.
        conf: Ngưỡng confidence của YOLO.
        iou: Ngưỡng IoU của YOLO, chọn cache riêng cho từng ngưỡng.
        limit: Số frame thử, 0 nghĩa là toàn bộ.
        output: Thư mục lưu file MOT và metadata.
        device: Thiết bị PyTorch.
    """
    import torch
    from boxmot.tracker_zoo import create_tracker, get_tracker_config
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    config = get_tracker_config(tracker_name)
    tracker = create_tracker(tracker_name, config, ROOT / 'osnet_x0_25_msmt17.pt',
                             torch.device(device), False, False)
    paths = sorted((ROOT / 'data' / video / 'img1').glob('*.jpg'))
    if limit:
        paths = paths[:limit]
    output.mkdir(parents=True, exist_ok=True)
    records, counts = [], []
    ids = set()
    started = time.perf_counter()
    with np.load(ROOT / '.cache/detections' / f'{video}.npz') as cache:
        for index, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f'Không đọc được ảnh {path}')
            detections = filter_detections(cache[f'f{index+1}_i{int(iou*10)}'], conf)
            tracks = tracker.update(detections, frame)
            counts.append(len(tracks))
            for row in tracks:
                x1, y1, x2, y2, tid, score = row[:6]
                ids.add(int(tid))
                records.append(f'{index+1},{int(tid)},{x1:.2f},{y1:.2f},{x2-x1:.2f},'
                               f'{y2-y1:.2f},{score:.4f},-1,-1,-1')
            if (index + 1) % 150 == 0:
                print(f'[tracker] {video} {tracker_name} conf={conf}: {index+1}/{len(paths)}', flush=True)
    target = output / f'{video}.txt'
    target.write_text('\n'.join(records) + '\n', encoding='utf-8')
    metadata = {'video': video, 'tracker': tracker_name, 'conf': conf, 'iou': iou,
                'frames_processed': len(paths), 'full_sequence': not limit,
                'seconds': time.perf_counter() - started, 'rows': len(records),
                'unique_ids': len(ids), 'mean_tracks': float(np.mean(counts)),
                'tracker_config_sha256': file_sha256(config),
                'reid_sha256': file_sha256(ROOT / 'osnet_x0_25_msmt17.pt'),
                'result_sha256': file_sha256(target), 'device': device, 'half': False}
    (output / f'{video}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['cache', 'track'])
    parser.add_argument('--video', required=True)
    parser.add_argument('--tracker', default='bytetrack', choices=['bytetrack', 'ocsort', 'botsort', 'strongsort', 'deepocsort'])
    parser.add_argument('--conf', type=float, default=.3)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/trials')
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    if arguments.action == 'cache':
        cache_detector(arguments.video, arguments.device)
    else:
        track(arguments.video, arguments.tracker, arguments.conf, arguments.iou,
              arguments.max_frames, arguments.out, arguments.device)

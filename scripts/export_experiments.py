"""Xuất kết quả thử nghiệm và bảng metric có nguồn để kiểm tra trên GitHub."""

import csv
import json
from pathlib import Path
import shutil

from lab_experiments import ROOT

if __name__ == '__main__':
    target = ROOT / 'submission/experiments'
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for directory in sorted((ROOT / 'runs/trials').iterdir()):
        if not directory.is_dir():
            continue
        for metadata_path in sorted(directory.glob('video_*.json')):
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            video = metadata['video']
            destination = target / directory.name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metadata_path, destination / metadata_path.name)
            shutil.copy2(metadata_path.with_suffix('.txt'), destination / f'{video}.txt')
            record = {'trial': directory.name, 'video': video, 'tracker': metadata['tracker'],
                      'conf': metadata['conf'], 'iou': metadata['iou'],
                      'frames': metadata['frames_processed'], 'full_sequence': metadata['full_sequence'],
                      'tracker_seconds': round(metadata['seconds'], 3), 'rows': metadata['rows'],
                      'unique_ids': metadata['unique_ids'], 'HOTA': '', 'MOTA': '', 'IDF1': '',
                      'IDSW': '', 'FN': '', 'FP': '', 'result_sha256': metadata['result_sha256']}
            if video == 'video_1':
                metric_dir = ROOT.parent / 'TrackEval/data/trackers/mot_challenge/LAB21-train' / f'than_{directory.name}'
                summary = metric_dir / 'pedestrian_summary.txt'
                if summary.exists():
                    lines = summary.read_text().splitlines()
                    metrics = dict(zip(lines[0].split(), lines[1].split()))
                    record.update({key: metrics[key] for key in ['HOTA', 'MOTA', 'IDF1', 'IDSW']})
                    record.update({'FN': metrics['CLR_FN'], 'FP': metrics['CLR_FP']})
                    shutil.copy2(summary, destination / 'video_1_metrics.txt')
                    detail = metric_dir / 'pedestrian_detailed.csv'
                    if detail.exists():
                        shutil.copy2(detail, destination / 'video_1_metrics_detailed.csv')
            records.append(record)
    with (ROOT / 'submission/EXPERIMENTS.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Đã xuất {len(records)} lượt thử, chỉ video_1 có metric.', flush=True)

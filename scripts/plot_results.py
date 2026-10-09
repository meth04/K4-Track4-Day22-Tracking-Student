"""Vẽ biểu đồ từ CSV kết quả thật, chỉ dùng metric của video_1."""

import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lab_experiments import ROOT

if __name__ == '__main__':
    with (ROOT / 'submission/EXPERIMENTS.csv').open(encoding='utf-8-sig') as stream:
        records = [row for row in csv.DictReader(stream) if row['video'] == 'video_1']
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    names = ['bytetrack', 'ocsort', 'botsort', 'strongsort', 'deepocsort']
    baseline = [next(row for row in records if row['tracker'] == name
                     and float(row['conf']) == .3 and float(row['iou']) == .5) for name in names]
    values = [float(row['HOTA']) for row in baseline]
    axes[0].barh(names[::-1], values[::-1], color=['#64748b', '#64748b', '#2563eb', '#64748b', '#64748b'])
    axes[0].set_xlim(0, 35)
    axes[0].set_xlabel('HOTA (0–100)')
    axes[0].set_title('5 tracker: conf=0.30, iou=0.50')
    for index, value in enumerate(values[::-1]):
        axes[0].text(value + .25, index, f'{value:.3f}', va='center', fontsize=9)
    grid = np.zeros((3, 3))
    for row_index, conf in enumerate([.15, .3, .5]):
        for column_index, iou in enumerate([.4, .5, .7]):
            row = next(row for row in records if row['tracker'] == 'botsort'
                       and float(row['conf']) == conf and float(row['iou']) == iou)
            grid[row_index, column_index] = float(row['HOTA'])
    colored = axes[1].imshow(grid, cmap='Blues', vmin=25, vmax=31)
    axes[1].set_xticks(range(3), ['0.40', '0.50', '0.70'])
    axes[1].set_yticks(range(3), ['0.15', '0.30', '0.50'])
    axes[1].set_xlabel('iou detector')
    axes[1].set_ylabel('conf detector')
    axes[1].set_title('BoT-SORT: HOTA trên 600 frame')
    for row_index in range(3):
        for column_index in range(3):
            value = grid[row_index, column_index]
            axes[1].text(column_index, row_index, f'{value:.3f}', ha='center', va='center',
                         color='white' if value > 28.8 else 'black', fontsize=11)
    figure.colorbar(colored, ax=axes[1], fraction=.046)
    lines = (ROOT / 'evidence/video_1_metrics.txt').read_text().splitlines()
    final = dict(zip(lines[0].split(), lines[1].split()))
    metrics = ['HOTA', 'MOTA', 'IDF1']
    positions = np.arange(3)
    axes[2].bar(positions - .18, [float(baseline[0][key]) for key in metrics], .36,
                label='ByteTrack baseline', color='#94a3b8')
    axes[2].bar(positions + .18, [float(final[key]) for key in metrics], .36,
                label='BoT-SORT bản nộp', color='#2563eb')
    axes[2].set_xticks(positions, metrics)
    axes[2].set_ylim(0, 35)
    axes[2].set_title('Bản nộp chạy trực tiếp')
    axes[2].set_ylabel('Điểm (0–100)')
    axes[2].legend(loc='lower right', fontsize=8)
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt='%.3f', fontsize=8, padding=3)
    figure.suptitle('video_1 — cùng detector YOLO26n, ảnh 640 px, chỉ lớp người', fontsize=13)
    figure.savefig(ROOT / 'evidence/video_1_comparison.png', dpi=180)
    plt.close(figure)
    print('Đã vẽ biểu đồ video_1 từ CSV và summary TrackEval.')

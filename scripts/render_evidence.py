"""Vẽ ID từ kết quả MOT để xem video và kiểm chứng nhận xét bằng hình ảnh."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from lab_experiments import ROOT


def render(mot: Path, video: str, output: Path, frames: list[int],
           save_video: bool, label: str) -> None:
    """Tạo ảnh theo thời gian và video có vẽ ID từ file kết quả.

    Args:
        mot: File kết quả MOT 10 cột.
        video: Tên video tương ứng dữ liệu ảnh.
        output: Đường dẫn ảnh JPEG đầu ra.
        frames: Các frame cần chụp bằng chứng, đánh số từ 1.
        save_video: Có ghi video MP4 đầy đủ hay không.
        label: Nhãn cấu hình hiển thị trên ảnh và video.

    Raises:
        RuntimeError: Khi không mở được VideoWriter để ghi video.
    """
    data = np.loadtxt(mot, delimiter=',', ndmin=2)
    paths = sorted((ROOT / 'data' / video / 'img1').glob('*.jpg'))
    tiles, writer = [], None
    output.parent.mkdir(parents=True, exist_ok=True)
    if save_video:
        writer = cv2.VideoWriter(str(output.with_suffix('.mp4')),
                                cv2.VideoWriter_fourcc(*'mp4v'), 20, (960, 540))
        if not writer.isOpened():
            raise RuntimeError('Không mở được VideoWriter.')
    for index, path in enumerate(paths, 1):
        if not save_video and index not in frames:
            continue
        image = cv2.imread(str(path))
        for row in data[data[:, 0] == index]:
            tid = int(row[1])
            x, y, width, height = row[2:6]
            rng = np.random.default_rng(tid * 9973 + 17)
            color = tuple(int(channel) for channel in rng.integers(64, 255, size=3))
            cv2.rectangle(image, (int(x), int(y)), (int(x+width), int(y+height)), color, 3)
            cv2.putText(image, str(tid), (int(x), max(20, int(y)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .9, color, 3)
        image = cv2.resize(image, (960, 540))
        cv2.rectangle(image, (0, 0), (960, 36), (0, 0, 0), -1)
        cv2.putText(image, f'{video} | frame {index} | {label}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        if writer is not None:
            writer.write(image)
        if index in frames:
            tiles.append(image)
    if writer is not None:
        writer.release()
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[index:index+2]) for index in range(0, len(tiles), 2)])
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'Đã vẽ bằng chứng: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mot', type=Path, required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[1, 30, 60, 90, 120, 150])
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    render(args.mot, args.video, args.out, args.frames, args.save_video, args.label)

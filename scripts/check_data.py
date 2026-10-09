#!/usr/bin/env python
"""Kiểm tra gói lab_data giảng viên phát đã đủ ảnh (và nhãn video luyện) chưa.

Ví dụ:
    python scripts/check_data.py --lab-data-root "$LAB_DATA"
"""

from __future__ import annotations

import argparse
from pathlib import Path

VIDEOS = ["video_1", "video_2", "video_3", "video_4", "video_5"]
PRACTICE_VIDEO = "video_1"

CONTEXT = {
    "video_1": "Quảng trường ngoài trời, camera TĨNH, ban ngày, mật độ trung bình — video luyện, có nhãn",
    "video_2": "Phố, camera TĨNH trên cao, BAN ĐÊM, mật độ RẤT đông",
    "video_3": "Camera DI CHUYỂN, độ phân giải thấp, khung hình chậm",
    "video_4": "TRONG NHÀ, camera di chuyển tiến về phía trước, phản chiếu kính",
    "video_5": "Quay từ XE BUS tại giao lộ đông, rung lắc mạnh",
}


def check(lab_data_root: Path) -> bool:
    """In trạng thái từng video và báo gói lab có dùng được không.

    Args:
        lab_data_root: Thư mục lab_data giảng viên phát.

    Returns:
        True nếu cả năm video có ảnh và video_1 có file nhãn.
    """
    all_ok = True
    for name in VIDEOS:
        img_dir = lab_data_root / name / "img1"
        n_imgs = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        gt_file = lab_data_root / name / "gt" / "gt.txt"
        has_gt = gt_file.exists()
        if name == PRACTICE_VIDEO:
            ok = n_imgs > 0 and has_gt
            gt_note = "có nhãn" if has_gt else "THIẾU nhãn video luyện"
        else:
            ok = n_imgs > 0
            gt_note = "không có nhãn (đúng)" if not has_gt else "có nhãn — không dùng để tự chấm"
        all_ok &= ok
        status = "OK   " if ok else "THIẾU"
        print(f"[{status}] {name:8s}  {n_imgs:4d} ảnh  {gt_note}  — {CONTEXT[name]}")
    preview_dir = lab_data_root / "preview"
    n_preview = len(list(preview_dir.glob("*.mp4"))) if preview_dir.exists() else 0
    print(f"\nPreview .mp4: {n_preview} file trong {preview_dir}")
    if all_ok:
        print("Gói dữ liệu đủ để chạy scripts/run_tracking.py.")
    else:
        print("Thiếu ảnh hoặc thiếu nhãn video_1. Nhờ giảng viên phát lại gói lab_data.")
    return all_ok


def main() -> None:
    """Đọc ``--lab-data-root`` và in kết quả kiểm tra gói dữ liệu."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab-data-root", required=True, type=Path, help="Thư mục lab_data giảng viên phát")
    args = parser.parse_args()
    if not check(args.lab_data_root):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

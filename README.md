# Lab Tracking — bài cá nhân Nguyễn Văn Thân

**Họ tên:** Nguyễn Văn Thân · **MSV:** 2A202602859 · **Ngày thực hiện:** 09/10/2026.

Đã chạy đủ **4.137 frame của 5 video**, thực hiện **53 lượt thử cấu hình**, giữ nguyên detector YOLO26 nano, ảnh 640 px, lớp người và trọng số Re-ID theo đề.

## Bài nộp

- **[Báo cáo cá nhân đầy đủ](submission/BAO_CAO_NguyenVanThan_2A202602859.md)**: lựa chọn, số liệu thật, cấu hình đã loại, nhận xét từng cảnh và ảnh có ID.
- **[5 file kết quả chính thức](runs/nop_bai/)**: `video_1.txt` … `video_5.txt`, kèm metadata số frame và SHA-256, đúng đường dẫn đề yêu cầu. [submission/results/](submission/results/) chứa bản sao giống hệt để xem cùng báo cáo. Git chỉ theo dõi 5 file TXT và 5 metadata JSON trong `runs/nop_bai/`; video và các lượt chạy khác được bỏ qua.
- [Notebook đã chạy](on_tap_metrics.ipynb): 3 câu True/False và detector một frame.
- [Bảng 53 lượt thử](submission/EXPERIMENTS.csv), [file thử và metric gốc](submission/experiments/), [baseline đủ frame](submission/baseline/).
- [Cấu hình cuối](submission/selection.json), [kiểm tra bài nộp](evidence/validation.json), [môi trường GPU](evidence/runtime.json), [ảnh bằng chứng](evidence/).

| Video | Frame xử lý | Tracker | conf | iou |
|---|---:|---|---:|---:|
| video_1 | 600 | BoT-SORT | 0.30 | 0.70 |
| video_2 | 1.050 | BoT-SORT | 0.15 | 0.50 |
| video_3 | 837 | BoT-SORT | 0.30 | 0.40 |
| video_4 | 900 | BoT-SORT | 0.30 | 0.40 |
| video_5 | 750 | BoT-SORT | 0.15 | 0.50 |

Chỉ `video_1` có nhãn: **HOTA 29.969 · MOTA 19.025 · IDF1 29.703** (thang 0–100). HOTA tăng **3.057 điểm** so với ByteTrack baseline. [Đầu ra TrackEval gốc](evidence/video_1_evaluation.log). Bốn video còn lại được đánh giá bằng quan sát, không tự tạo nhãn hoặc điền metric.

## Chạy lại trên Windows

Tạo môi trường theo phần dưới, đặt dữ liệu trong `data/` hoặc truyền đường dẫn riêng. Sau đó:

```powershell
conda activate cv_robotics_lab21
$env:LAB_DATA = "$PWD\data"
python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Device cpu -TrackEvalRoot ..\TrackEval
```

Nếu đã cài PyTorch CUDA tương thích, đổi `-Device cpu` thành `-Device cuda:0`. Script đọc `submission/selection.json`, chạy toàn bộ frame, tạo video xem tại máy, chấm riêng `video_1` và kiểm tra file nộp. Có thể truyền `-LabData` và `-Python` khi cần.

Lần thực hiện này tận dụng môi trường PyTorch CUDA có sẵn để tránh tải lại vài GB; thư viện bổ sung được đặt riêng tại `.cache/deps` và không đưa lên Git. Môi trường Conda `cv_robotics_lab21` Python 3.11 cũng đã cài đầy đủ để chạy notebook, chấm và kiểm tra. Mọi bản nộp cuối chạy trực tiếp bằng `run_tracking.py`, không đọc cache detector. Chi tiết cách dùng cache và kiểm tra sai số ở báo cáo.

## Việc cần làm

1. Tạo môi trường một lần:

```bash
conda env create -f environment.yml
conda activate cv_robotics_lab21
pip install --no-deps boxmot==10.0.42
git clone https://github.com/JonathonLuiten/TrackEval.git
pip install -e TrackEval/
```

2. Tải ảnh năm video: [data_lab21.zip](https://drive.google.com/file/d/1UeVPQd6j5pSzxoJDcKJrerT9SL3vJLDt/view?usp=sharing). Giải nén, rồi gán đường dẫn thư mục chứa `video_1` … `video_5`:

```bash
export LAB_DATA=/đường/dẫn/lab_data
python scripts/check_data.py --lab-data-root "$LAB_DATA"
```

Cả năm video phải có ảnh. Chỉ `video_1` có nhãn. Trên PowerShell dùng `$env:LAB_DATA = 'C:\duong-dan\lab_data'` và truyền `--lab-data-root "$env:LAB_DATA"`.

3. Mở `on_tap_metrics.ipynb` bằng kernel env này. Đọc bảng MOTA / IDF1 / HOTA, rồi chạy YOLO trên một ảnh `video_1`.

4. Chạy tracker. Bản thử có thể giới hạn frame. Bản nộp thì không.

```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_1/img1" \
  --seq-name video_1 \
  --tracker bytetrack --conf 0.3 --iou 0.5 \
  --out runs/nop_bai --save-video
```

Đổi `--seq-name` và thư mục `img1` cho `video_2` … `video_5`. Tracker được chọn: `bytetrack`, `ocsort`, `botsort`, `strongsort`, `deepocsort`.

5. Chấm số **chỉ** `video_1`:

```bash
python scripts/evaluate_practice.py \
  --trackeval-root ~/TrackEval \
  --lab-data-root "$LAB_DATA" \
  --submission runs/nop_bai/video_1.txt \
  --run-name nhom01_video1
```

`video_2` đến `video_5` không có nhãn. Xem `preview/video_N.mp4` và video có vẽ ID, rồi ghi điều bạn thấy.

## Luật chơi

| Khóa | Bạn chọn |
|---|---|
| Detector `yolo26n.pt`, ảnh 640 px, lớp người, Re-ID `osnet_x0_25_msmt17.pt` | Tracker, `--conf`, `--iou` của detector |

## Nộp

- `video_1.txt` … `video_5.txt` trong `runs/nop_bai/` (đủ frame, đúng tên).
- `submission_template/BAO_CAO_mau.md` đã điền. Số HOTA / MOTA / IDF1 chỉ bắt buộc cho `video_1`.

Chi tiết từng bước, sự cố, và lịch 2 giờ: [HUONG_DAN.md](HUONG_DAN.md).

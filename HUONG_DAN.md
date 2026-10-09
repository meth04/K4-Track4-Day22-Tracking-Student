# Lab: So sánh tracker trên 5 video — hướng dẫn học viên

**Thời lượng:** 2 giờ · **Nhóm:** 2 người/máy · **Môi trường:** conda `cv_robotics_lab21`

Bạn đã nghe buổi lý thuyết (IoU, Hungarian, tracker, MOTA / IDF1 / HOTA). Buổi này không cài tracker từ đầu. Bạn **chạy tracker có sẵn** trên video thật và chọn cấu hình hợp từng cảnh.

## Luật chơi

Cả lớp dùng **chung một detector**:

| Thành phần | Giá trị | Bạn được đổi? |
|---|---|---|
| Detector | YOLO26 nano, `yolo26n.pt` (Ultralytics) | Không |
| Kích thước ảnh | 640 px | Không |
| Lớp | người đi bộ | Không |
| Re-ID | `osnet_x0_25_msmt17.pt` (tự tải lần đầu) | Không |
| Tracker | `bytetrack`, `ocsort`, `botsort`, `strongsort`, `deepocsort` | Có, từng video |
| Ngưỡng detector | `--conf`, `--iou` | Có — đây là ngưỡng của YOLO, không phải ngưỡng bên trong tracker |

`botsort`, `strongsort`, `deepocsort` dùng thêm ngoại hình (Re-ID). `bytetrack` và `ocsort` chủ yếu dựa vào chuyển động.

## Năm video

Giảng viên phát thư mục `lab_data`. Xem `preview/video_N.mp4` để nắm cảnh, rồi chạy tracker trên thư mục ảnh `img1/` (bản nộp chính thức).

| Video | Ngữ cảnh | Cách đánh giá |
|---|---|---|
| `video_1` | Quảng trường, camera tĩnh, ban ngày, mật độ vừa | Có nhãn. Sau khi chọn cấu hình, chạy `evaluate_practice.py` để đọc HOTA / MOTA / IDF1 |
| `video_2` | Phố, camera tĩnh trên cao, ban đêm, rất đông | Chỉ bằng mắt |
| `video_3` | Camera di chuyển, ảnh nhỏ, ít khung hình/giây | Chỉ bằng mắt |
| `video_4` | Trong nhà, camera tiến tới, kính phản chiếu | Chỉ bằng mắt |
| `video_5` | Trên xe bus, giao lộ đông, rung lắc | Chỉ bằng mắt |

`video_2` đến `video_5` **không có nhãn** trong gói của bạn. Đừng tìm file nhãn hay tên bộ dữ liệu công khai. Việc của bạn là nhìn video có vẽ ID và quyết định tracker nào hợp cảnh đó.

## Thời gian biểu

| Phút | Việc |
|---|---|
| 0–15 | Tạo env, cài TrackEval, `export LAB_DATA`, chạy `check_data.py` |
| 15–35 | `on_tap_metrics.ipynb`: đọc MOTA / IDF1 / HOTA, rồi YOLO một frame của `video_1` |
| 35–45 | Baseline: `video_1` + ByteTrack, tối đa 150 frame, xem video thử |
| 45–80 | Mỗi video thử ít nhất 2 tracker, mỗi lần đổi một tham số, rồi chạy **đủ frame** cho bản nộp |
| 80–110 | Chấm số **chỉ video_1**; bốn video kia viết quan sát; điền báo cáo |
| 110–120 | Thảo luận: tracker nào hợp cảnh nào, vì sao |

Nếu máy chậm, bớt lượt quét tham số ở video đông. Vẫn nộp đủ 5 file kết quả full-frame cho cấu hình bạn chọn.

## Bước C — Cài đặt

Một lần trên máy:

```bash
conda env create -f environment.yml
conda activate cv_robotics_lab21
pip install --no-deps boxmot==10.0.42
git clone https://github.com/JonathonLuiten/TrackEval.git
pip install -e TrackEval/
```

Lần sau chỉ cần `conda activate cv_robotics_lab21`.

`requirements.txt` đã ghim NumPy tương thích với Python mới. Cài BoxMOT bằng `--no-deps` sau các thư viện runtime để tránh metadata cũ ép NumPy 1.23.1. Giữ đúng BoxMOT 10.0.42 theo đề. Script đã chuyển chuỗi `--device` sang `torch.device` cho BoxMOT và đặt detector lên thiết bị đã chọn.

Tải ảnh năm video: [data_lab21.zip](https://drive.google.com/file/d/1UeVPQd6j5pSzxoJDcKJrerT9SL3vJLDt/view?usp=sharing). Giải nén, rồi gán `LAB_DATA` tới thư mục chứa `video_1` … `video_5`:

```bash
export LAB_DATA=/đường/dẫn/lab_data
python scripts/check_data.py --lab-data-root "$LAB_DATA"
```

Kỳ vọng: cả 5 video có ảnh; chỉ `video_1` ghi "có nhãn".

Mở `on_tap_metrics.ipynb` bằng kernel của env `cv_robotics_lab21` (biến `LAB_DATA` phải có trong terminal đó). Phần đầu là bảng metric. Phần sau chạy YOLO trên một ảnh `video_1`.

## Bước D — Baseline

```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_1/img1" \
  --seq-name video_1 \
  --tracker bytetrack --conf 0.3 --iou 0.5 \
  --out runs/thu_nhanh --save-video --max-frames 150 --device cpu
```

Mở `runs/thu_nhanh/video_1_preview.mp4` (nếu không mở được, dùng VLC). Mỗi người một màu. ID đổi màu giữa chừng là dấu hiệu đổi danh tính.

`--max-frames 150` chỉ để thử. Bản nộp phải chạy lại **không** có `--max-frames`.

Có GPU thì thêm `--device cuda:0`.

## Bước E — Thử có hệ thống

Với **mỗi** video:

1. Chạy ít nhất hai tracker: một tracker chủ yếu theo chuyển động (`bytetrack` hoặc `ocsort`) và một tracker có Re-ID (`botsort`, `strongsort`, hoặc `deepocsort`).
2. Với tracker có vẻ hơn, thử `--conf` 0.15 / 0.3 / 0.5 và `--iou` 0.4 / 0.5 / 0.7. Mỗi lần chỉ đổi một số. Ghi bạn thấy gì (nhiều hộp giả, mất người, ID nhảy khi hai người đi ngang nhau).
3. Chọn một cấu hình, chạy đủ frame, ghi vào **cùng** thư mục nộp:

```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_2/img1" \
  --seq-name video_2 \
  --tracker strongsort --conf 0.25 --iou 0.5 \
  --out runs/nop_bai --save-video
```

Cuối bước này `runs/nop_bai/` phải có `video_1.txt` … `video_5.txt`.

Gợi ý khi xem:

- Cùng một người đang đi đều mà đổi màu ID: đổi danh tính.
- Hộp nhảy sang người bên cạnh lúc hai người cắt nhau: tracker gán nhầm, hay gặp khi cảnh đông và chỉ dùng chuyển động.
- Người rõ trong ảnh nhưng không có ID: `--conf` có thể đang cao.
- Hộp nhấp nháy trên nền, bóng, vật không phải người: `--conf` có thể đang thấp.

## Bước F — Số liệu video_1 và báo cáo

Chỉ `video_1`:

```bash
python scripts/evaluate_practice.py \
  --trackeval-root ~/TrackEval \
  --lab-data-root "$LAB_DATA" \
  --submission runs/nop_bai/video_1.txt \
  --run-name nhom01_video1
```

- **MOTA** cao khi ít bỏ sót, ít hộp giả, ít lần đổi ID. Đổi ID bị trừ theo số lần, không theo việc ID sai kéo dài bao lâu.
- **IDF1** nhạy với việc giữ đúng danh tính suốt một track dài.
- **HOTA** cân bằng phát hiện đúng và giữ đúng danh tính. Dùng HOTA khi cần một số tổng.

Dán bảng của `video_1` vào báo cáo. Với `video_2`–`video_5`, cột quan sát là mô tả bạn thấy trên video, không phải số HOTA.

Điền [`submission_template/BAO_CAO_mau.md`](submission_template/BAO_CAO_mau.md).

## Nộp bài

- Năm file `video_1.txt` … `video_5.txt` (chạy đủ frame, đúng tên).
- Báo cáo đã điền: tracker, conf, iou, lý do, và quan sát.
- `video_1` có thêm bảng HOTA / MOTA / IDF1.
- Ít nhất hai video có đoạn giải thích vì sao tracker đó hợp cảnh (chuyển động hay Re-ID, camera đứng yên hay chuyển động, đông hay thưa, sáng hay tối).
- Không đổi file trọng số detector, kích thước ảnh, hay mô hình Re-ID trong bài nộp chính.

Giảng viên chấm chất lượng track của cả năm video sau buổi. Trong giờ lab bạn không có nhãn của bốn video khó.

## Sự cố thường gặp

| Hiện tượng | Việc nên làm |
|---|---|
| `No module named 'trackeval'` | `pip install -e` đúng thư mục TrackEval đã clone, trong env `cv_robotics_lab21` |
| `module 'numpy' has no attribute 'float'` | Gọi `evaluate_practice.py`, đừng gọi thẳng script trong thư mục TrackEval |
| Script từ chối file không phải `video_1.txt` | Đúng thiết kế: chỉ video luyện được chấm số |
| Chạy rất chậm | Thêm `--max-frames 150` khi **thử**; bỏ khi ra file nộp |
| Video preview không mở | Thử VLC |
| `check_data.py` báo thiếu ảnh | Sai đường dẫn `LAB_DATA`, hoặc gói chưa tải xong |

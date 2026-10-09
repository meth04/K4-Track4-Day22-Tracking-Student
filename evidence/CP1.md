# CP1 — quan sát và giả thuyết trước thử nghiệm

**Nguyễn Văn Thân — 2A202602859 — bài cá nhân.**

Dữ liệu có 600 / 1.050 / 837 / 900 / 750 frame tương ứng video_1 đến video_5. Chỉ video_1 có nhãn. Gói hiện tại không có thư mục preview; quan sát ảnh đầu, một phần ba, hai phần ba và cuối chuỗi trước khi thử, rồi tạo video có vẽ ID từ chuỗi ảnh để xem chuyển động.

1. **video_1:** camera tĩnh, người đi qua nhau ở tiền cảnh. Giả thuyết: ByteTrack là baseline hợp lý; Re-ID có thể giúp nối lại người bị che khuất. Cần kiểm tra bằng HOTA, IDF1 và hình ảnh, không suy ra chất lượng chỉ từ số ID.
2. **video_2:** camera cao, đèn sáng mạnh và nhiều người nhỏ ở xa. Giả thuyết: conf thấp giúp giữ hộp yếu nhưng có thể tăng hộp giả; BoT-SORT có ngoại hình có thể giảm gán nhầm khi người giao nhau.
3. **video_3:** ảnh 640 × 480, camera di chuyển, người ở gần che phần lớn ảnh. Giả thuyết: OC-SORT xử lý quỹ đạo có thể khác ByteTrack; tracker có bù camera và Re-ID có thể tốt hơn, nhưng ngoại hình mờ là giới hạn.
4. **video_4:** camera đi trong trung tâm thương mại, mặt sàn và kính phản chiếu. Giả thuyết: cần cân bằng conf để tránh hộp trên phản chiếu; Re-ID có thể giữ người đi trước camera qua che khuất.
5. **video_5:** camera trên phương tiện, phối cảnh thay đổi mạnh khi rẽ, người nhỏ ở xa. Giả thuyết: bù chuyển động camera có thể giúp; mọi tracker đều khó nếu detector bỏ sót người nhỏ. Cần xem các đoạn trước và sau khi camera rẽ.

Notebook trả lời 3 câu: **True / False / True**. Đây là nhận định theo bảng ví dụ của đề, không phải số đo của năm video. Notebook chạy YOLO một frame để phân biệt hộp phát hiện với track có ID.

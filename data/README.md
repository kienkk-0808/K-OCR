# Cấu trúc thư mục `data/`

```
data/
  train/
    images/           <- ảnh biển số dùng để train
    labels.txt        <- nhãn tương ứng
  val/
    images/           <- ảnh biển số dùng để validate
    labels.txt        <- nhãn tương ứng
```

## Định dạng `labels.txt`

Mỗi dòng: đường dẫn ảnh (tương đối so với thư mục chứa `labels.txt`), theo sau
là ký tự **Tab**, rồi đến biển số dạng chữ:

```
images/plate_00001.jpg	29A-12345
images/plate_00002.jpg	30F-123.45
```

- Ký tự cho phép: `0-9`, `A-Z` (viết hoa), `-`, `.` (xem `charset.py`).
- Không dùng dấu, không dùng chữ thường.
- Ảnh có thể là ảnh biển số đã được crop sẵn (không cần detect trong dự án
  này — model chỉ làm nhiệm vụ OCR trên ảnh đã crop).
- Ảnh sẽ tự động được resize về `112x224` (H x W) khi load, không cần resize
  trước.

## Chưa có dữ liệu thật?

Chạy script sau để tạo dữ liệu giả lập (synthetic), giúp kiểm tra nhanh toàn
bộ pipeline train/tune/infer trước khi có dữ liệu thật:

```bash
python generate_dummy_data.py --data-dir data --train-samples 200 --val-samples 40
```

**Lưu ý:** dữ liệu giả lập chỉ để test code chạy được, không dùng để đánh giá
độ chính xác thật. Khi có ảnh biển số thật, hãy thay thế toàn bộ nội dung
trong `data/train` và `data/val` theo đúng định dạng ở trên.

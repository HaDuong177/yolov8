# BÁO CÁO ĐÁNH GIÁ CHI TIẾT HIỆU NĂNG MÔ HÌNH YOLOV8s
## HỆ THỐNG PHÁT HIỆN ĐÁM CHÁY VÀ KHÓI (FIRE & SMOKE DETECTION)

---

## 1. TỔNG QUAN HỆ THỐNG VÀ CẤU HÌNH THỰC NGHIỆM

* **Kiến trúc mô hình:** YOLOv8s (Ultralytics YOLOv8 Small Detection Model)
* **Tổng số tham số (Parameters):** 11.12 triệu tham số (72 layers)
* **Độ phức tạp tính toán:** 28.4 GFLOPs
* **Phần cứng thực thi:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
* **Môi trường phần mềm:** Python 3.11.4, PyTorch 2.5.1 + CUDA 12.1, Ultralytics 8.3.108
* **Thời gian huấn luyện:** ~3.65 giờ (13,154 giây) cho 50 Epochs
* **Kích thước ảnh đầu vào (Image Size):** 640 × 640
* **Batch size:** 16
* **Tối ưu hóa (Optimizer):** Auto (SGD/AdamW với Warmup 3 epochs)
* **Tập dữ liệu phân loại (Classes):**
  * `Class 0`: **Fire** (Ngọn lửa / Đám cháy)
  * `Class 1`: **Smoke** (Khói)

---

## 2. KẾT QUẢ ĐỊNH LƯỢNG CHI TIẾT

### 2.1. Đánh giá trên tập Kiểm thử độc lập (Test Set — 1,000 ảnh)

| Phân loại (Class) | Số lượng nhãn (Instances) | Precision (Độ chính xác) | Recall (Độ bao phủ) | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Toàn bộ (All)** | **2,177** | **78.8%** | **73.9%** | **77.4%** | **47.0%** |
| **Lửa (`Fire`)** | 1,839 | **86.1%** | **81.1%** | **87.1%** | **56.1%** |
| **Khói (`Smoke`)** | 338 | **71.5%** | **66.7%** | **67.7%** | **37.8%** |

* **Chỉ số mAP@0.75 trên tập Test:** **50.6%**

---

### 2.2. Đánh giá trên tập Xác thực (Validation Set — 2,000 ảnh)

| Phân loại (Class) | Số lượng nhãn (Instances) | Precision (Độ chính xác) | Recall (Độ bao phủ) | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Toàn bộ (All)** | **4,312** | **80.1%** | **69.5%** | **75.1%** | **45.1%** |
| **Lửa (`Fire`)** | 3,627 | **87.5%** | **79.2%** | **85.5%** | **55.0%** |
| **Khói (`Smoke`)** | 685 | **72.7%** | **59.9%** | **64.8%** | **35.3%** |

* **Chỉ số mAP@0.75 trên tập Val:** **48.0%**

---

### 2.3. Tốc độ suy luận (Inference Speed trên GPU RTX 3050 Laptop)

* **Tiền xử lý (Preprocess):** 0.3 ms / ảnh
* **Thời gian suy luận (Inference):** 10.7 ms / ảnh (~93.5 FPS)
* **Hậu xử lý Non-Max Suppression (Postprocess):** 1.0 ms / ảnh
* $\rightarrow$ **Tổng độ trễ xử lý:** **~12.0 ms / ảnh (~83 FPS)**
* **Đánh giá thời gian thực:** Vượt xa chuẩn camera giám sát thông thường (25–30 FPS), hoàn toàn đáp ứng khả năng giám sát trực tiếp đa luồng (multi-stream).

---

## 3. PHÂN TÍCH CHUYÊN SÂU (DEEP DIVE)

### 3.1. Điểm mạnh nổi bật
1. **Khả năng tổng quát hóa xuất sắc (Không Overfitting):**
   * Chỉ số mAP@0.5 trên tập Test đạt **77.4%**, cao hơn tập Validation (**75.1%**).
   * Điều này chứng minh mô hình không bị hiện tượng "học vẹt", các đặc trưng trích xuất có khả năng thích nghi cao với dữ liệu thực tế mới.
2. **Độ nhạy và độ tin cậy của lớp Đám cháy (Fire):**
   * Đạt Precision **86.1%**, Recall **81.1%** và mAP@0.5 lên tới **87.1%**.
   * Với màu sắc đặc trưng và độ sáng tương phản, ngọn lửa được phát hiện chính xác, hạn chế tình trạng báo động nhầm.
3. **Quá trình hội tụ ổn định:**
   * Hàm mất mát phân loại huấn luyện (`train/cls_loss`) giảm sâu từ `2.30` xuống `0.62`.
   * Hàm mất mát xác thực (`val/cls_loss`) giảm từ `2.56` xuống `0.77`, các đường cong suy giảm mượt mà.

### 3.2. Hạn chế và Thách thức kỹ thuật
1. **Hiện tượng mất cân bằng mẫu (Class Imbalance):**
   * Tỉ lệ nhãn giữa Lửa và Khói bị chênh lệch lớn: trên tập Test có **1,839 nhãn Fire** nhưng chỉ có **338 nhãn Smoke** (chênh lệch hơn 5.4 lần).
   * Hậu quả là lớp Khói có mAP@0.5 thấp hơn (**67.7%**) và Recall chỉ đạt **66.7%**, vẫn còn bỏ sót khoảng 33% các trường hợp khói loãng.
2. **Đặc thù hình học của Khói (Amorphous Object):**
   * Khói không có biên dạng cố định, thay đổi theo hướng gió và dễ hòa lẫn vào nền trời, mây hoặc sương mù.
3. **Cảnh báo dữ liệu nhãn hỗn hợp:**
   * Tập dữ liệu gốc có một số ảnh nhãn dạng segmentation/polygon xen lẫn bounding box. Cần chuẩn hóa dữ liệu về thuần định dạng YOLO detection box.

---

## 4. KIẾN NGHỊ TỐI ƯU HÓA & TRIỂN KHAI THỰC TẾ

1. **Cấu hình ngưỡng tin cậy (Confidence Threshold) phù hợp:**
   * **Chế độ cảnh báo sớm (Early Fire Alarm):** Thiết lập `conf = 0.25 - 0.35` để tăng Recall, ưu tiên phát hiện sớm khói và tàn lửa từ xa.
   * **Chế độ kích hoạt thiết bị dập lửa / báo động khẩn:** Thiết lập `conf = 0.50 - 0.60` để triệt tiêu báo động giả.
2. **Tăng cường dữ liệu cho lớp Khói (Data Augmentation):**
   * Bổ sung thêm 1,500 - 2,000 ảnh đặc thù về khói (khói trắng, khói đen, khói ban ngày/ban đêm) để cân bằng tỉ lệ 1:1 với lớp lửa.
3. **Mở rộng số chu kỳ huấn luyện:**
   * Huấn luyện 70 - 80 epochs kết hợp kỹ thuật Cosine Learning Rate Schedule để tối ưu thêm 2 - 4% mAP.

---

## 5. DANH MỤC TÀI NGUYÊN & CÁCH TÁI HIỆN

* **Trọng số mô hình tốt nhất:** `runs_fire_detection/train_yolov8s/weights/best.pt`
* **Trọng số chu kỳ cuối:** `runs_fire_detection/train_yolov8s/weights/last.pt`
* **File nhật ký tiến trình:** `runs_fire_detection/train_yolov8s/results.csv`
* **Tài liệu PDF xuất bản:** `Bao_Cao_Danh_Gia_YOLOv8_Fire_Detection.pdf`
* **Script tạo lại file PDF:** Chạy lệnh:
  ```bash
  python generate_pdf_report.py
  ```
* **Lệnh kiểm tra lại chỉ số bằng Python:**
  ```python
  from ultralytics import YOLO
  model = YOLO('runs_fire_detection/train_yolov8s/weights/best.pt')
  metrics = model.val(data='data.yaml', split='test')
  ```

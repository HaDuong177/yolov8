# DỰ ÁN: HỆ THỐNG PHÁT HIỆN ĐÁM CHÁY VÀ KHÓI THỜI GIAN THỰC SỬ DỤNG MÔ HÌNH YOLOv8s

## *Fire and Smoke Real-Time Detection System Using YOLOv8s Deep Learning Model*

---

**Thông tin dự án:**

| Hạng mục | Chi tiết |
|:---|:---|
| **Tên đề tài** | Xây dựng hệ thống phát hiện đám cháy và khói thời gian thực sử dụng mô hình YOLOv8s |
| **Lĩnh vực** | Thị giác máy tính (Computer Vision) — Phát hiện đối tượng (Object Detection) |
| **Công nghệ chính** | YOLOv8s (Ultralytics), PyTorch, CUDA |
| **Phần cứng** | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) |
| **Ngôn ngữ lập trình** | Python 3.11.4 |

---

# MỤC LỤC

1. [LỜI MỞ ĐẦU](#1-lời-mở-đầu)
2. [TỔNG QUAN](#2-tổng-quan)
3. [GIỚI THIỆU CHUNG](#3-giới-thiệu-chung)
4. [CƠ SỞ LÝ THUYẾT](#4-cơ-sở-lý-thuyết)
5. [PHƯƠNG PHÁP](#5-phương-pháp)
6. [THỰC NGHIỆM VÀ KẾT QUẢ](#6-thực-nghiệm-và-kết-quả)
7. [KẾT LUẬN](#7-kết-luận)
8. [HƯỚNG PHÁT TRIỂN](#8-hướng-phát-triển)
9. [TÀI LIỆU THAM KHẢO](#9-tài-liệu-tham-khảo)

---

# 1. LỜI MỞ ĐẦU

Cháy nổ là một trong những hiểm họa nghiêm trọng nhất đối với con người, tài sản và môi trường. Theo thống kê của Cục Cảnh sát Phòng cháy, Chữa cháy và Cứu nạn, Cứu hộ (Bộ Công an Việt Nam), mỗi năm cả nước ghi nhận hàng ngàn vụ cháy gây thiệt hại lớn về người và tài sản. Trên phạm vi toàn cầu, Tổ chức Phòng cháy Quốc tế (NFPA — National Fire Protection Association) ước tính thiệt hại do hỏa hoạn lên tới hàng chục tỷ USD mỗi năm. Điểm chung của phần lớn các vụ cháy lớn là **sự chậm trễ trong phát hiện ban đầu** — khi ngọn lửa hoặc cột khói còn nhỏ và có thể kiểm soát được.

Các hệ thống phát hiện cháy truyền thống, như cảm biến nhiệt (heat sensor) và cảm biến khói (smoke detector), tuy đã được triển khai rộng rãi nhưng tồn tại nhiều hạn chế cố hữu:

- **Phạm vi giám sát hẹp:** Mỗi cảm biến chỉ bao phủ một vùng không gian nhỏ, không hiệu quả cho các khu vực mở rộng như rừng, nhà kho công nghiệp, hoặc bãi đỗ xe.
- **Tốc độ phản hồi chậm:** Cảm biến nhiệt cần nhiệt độ tăng đến ngưỡng nhất định mới kích hoạt, nghĩa là đám cháy đã bùng phát trước khi báo động.
- **Tỉ lệ báo động giả cao:** Khói từ nấu ăn, bụi bẩn, hoặc hơi nước có thể kích hoạt cảm biến khói sai lệch.
- **Không cung cấp thông tin trực quan:** Các cảm biến truyền thống chỉ đưa ra tín hiệu nhị phân (có/không), không xác định được vị trí, quy mô hay diễn biến của đám cháy.

Trong bối cảnh đó, sự phát triển mạnh mẽ của **Trí tuệ nhân tạo (AI)** và **Thị giác máy tính (Computer Vision)** đã mở ra một hướng tiếp cận mới: sử dụng camera giám sát kết hợp các mô hình học sâu (deep learning) để phát hiện đám cháy và khói từ hình ảnh/video theo thời gian thực. Phương pháp này cho phép giám sát diện rộng, phản hồi tức thì, và cung cấp thông tin trực quan phong phú (vị trí, kích thước, loại đối tượng).

Dự án này tập trung vào việc **xây dựng và đánh giá một hệ thống phát hiện đám cháy và khói thời gian thực** sử dụng mô hình **YOLOv8s** (You Only Look Once version 8 — Small) — một trong những kiến trúc phát hiện đối tượng hiện đại nhất hiện nay, với ưu thế vượt trội về cân bằng giữa tốc độ xử lý và độ chính xác.

---

# 2. TỔNG QUAN

## 2.1. Bối cảnh và Tính cấp thiết

Phát hiện cháy sớm (early fire detection) là yếu tố then chốt trong việc giảm thiểu thiệt hại. Nghiên cứu cho thấy nếu đám cháy được phát hiện trong **3 phút đầu tiên**, khả năng dập tắt thành công lên tới **90%**. Tuy nhiên, sau 5-10 phút, đám cháy có thể lan rộng gấp nhiều lần và trở nên khó kiểm soát.

Hệ thống camera giám sát (CCTV) đã được lắp đặt phổ biến tại các tòa nhà, nhà máy, khu công nghiệp và rừng. Tuy nhiên, phần lớn các hệ thống này chỉ **ghi hình thụ động**, dựa vào con người theo dõi 24/7 — một phương pháp không hiệu quả và dễ bỏ sót. Việc tích hợp AI vào hệ thống camera có sẵn cho phép **tự động hóa** quá trình giám sát, biến camera từ thiết bị ghi hình thụ động thành **bộ phát hiện thông minh chủ động**.

## 2.2. Mục tiêu của dự án

Dự án hướng tới các mục tiêu cụ thể sau:

1. **Xây dựng mô hình phát hiện đối tượng** có khả năng nhận diện hai lớp đối tượng: **Ngọn lửa (Fire)** và **Khói (Smoke)** từ hình ảnh.
2. **Đạt độ chính xác cao** với chỉ số mAP@0.5 ≥ 75% trên tập kiểm thử độc lập.
3. **Đảm bảo tốc độ xử lý thời gian thực** với tốc độ suy luận ≥ 30 FPS trên phần cứng GPU phổ thông (RTX 3050).
4. **Đánh giá toàn diện** hiệu năng mô hình thông qua các chỉ số Precision, Recall, mAP@0.5, mAP@0.5:0.95, ma trận nhầm lẫn (Confusion Matrix), và phân tích các đường cong P-R, F1.
5. **Đề xuất giải pháp tối ưu hóa** và hướng triển khai thực tế cho hệ thống.

## 2.3. Phạm vi nghiên cứu

| Hạng mục | Nội dung |
|:---|:---|
| **Đối tượng phát hiện** | Ngọn lửa (Fire) và Khói (Smoke) |
| **Loại dữ liệu** | Hình ảnh tĩnh (RGB) từ nhiều nguồn |
| **Mô hình sử dụng** | YOLOv8s (Ultralytics) |
| **Phương pháp đánh giá** | Precision, Recall, mAP@0.5, mAP@0.5:0.95, F1-score |
| **Phần cứng** | NVIDIA RTX 3050 Laptop GPU (4GB VRAM) |

---

# 3. GIỚI THIỆU CHUNG

## 3.1. Bài toán phát hiện đối tượng (Object Detection)

Phát hiện đối tượng là một bài toán cốt lõi trong Thị giác máy tính, với mục tiêu xác định **vị trí** (localization) và **phân loại** (classification) các đối tượng quan tâm trong một hình ảnh hoặc video. Khác với bài toán phân loại ảnh (image classification) chỉ trả lời câu hỏi "Ảnh này chứa gì?", phát hiện đối tượng trả lời đồng thời hai câu hỏi: **"Có gì?"** và **"Ở đâu?"**.

Kết quả đầu ra của bài toán phát hiện đối tượng bao gồm:
- **Bounding box** $(x, y, w, h)$: Hình chữ nhật bao quanh đối tượng, xác định bởi tọa độ tâm $(x, y)$, chiều rộng $w$ và chiều cao $h$.
- **Class label**: Nhãn phân loại (ví dụ: Fire, Smoke).
- **Confidence score**: Điểm tin cậy thể hiện mức độ chắc chắn của dự đoán.

## 3.2. Các phương pháp phát hiện đối tượng

### 3.2.1. Phương pháp hai giai đoạn (Two-Stage Detectors)
Các mô hình hai giai đoạn thực hiện phát hiện qua hai bước:
1. **Giai đoạn 1 — Region Proposal:** Sinh ra các vùng ứng viên (region proposals) có khả năng chứa đối tượng.
2. **Giai đoạn 2 — Classification & Refinement:** Phân loại từng vùng ứng viên và tinh chỉnh bounding box.

Đại diện tiêu biểu: **R-CNN** (Girshick et al., 2014), **Fast R-CNN** (Girshick, 2015), **Faster R-CNN** (Ren et al., 2016), **Mask R-CNN** (He et al., 2017).

**Ưu điểm:** Độ chính xác cao, đặc biệt với đối tượng nhỏ.
**Nhược điểm:** Tốc độ chậm, khó triển khai thời gian thực.

### 3.2.2. Phương pháp một giai đoạn (One-Stage Detectors)
Các mô hình một giai đoạn thực hiện dự đoán bounding box và phân loại **đồng thời** trong một lần truyền xuôi (single forward pass), loại bỏ bước sinh vùng ứng viên.

Đại diện tiêu biểu: **YOLO** (Redmon et al., 2016), **SSD** (Liu et al., 2016), **RetinaNet** (Lin et al., 2017), **EfficientDet** (Tan et al., 2020).

**Ưu điểm:** Tốc độ nhanh, phù hợp ứng dụng thời gian thực.
**Nhược điểm:** Có thể kém chính xác hơn hai giai đoạn đối với đối tượng nhỏ hoặc chồng chéo.

## 3.3. Dòng mô hình YOLO — Lịch sử phát triển

YOLO (You Only Look Once) là dòng mô hình phát hiện đối tượng một giai đoạn được giới thiệu lần đầu bởi Joseph Redmon vào năm 2016. Qua các phiên bản, YOLO liên tục được cải tiến:

| Phiên bản | Năm | Đóng góp chính | Tác giả / Nhóm phát triển |
|:---|:---:|:---|:---|
| YOLOv1 | 2016 | Mô hình phát hiện đối tượng một giai đoạn đầu tiên, chia ảnh thành lưới S×S | Redmon et al. |
| YOLOv2 | 2017 | Batch Normalization, anchor boxes, multi-scale training | Redmon & Farhadi |
| YOLOv3 | 2018 | Darknet-53 backbone, Feature Pyramid Network (FPN), dự đoán đa tỉ lệ | Redmon & Farhadi |
| YOLOv4 | 2020 | CSPDarknet53, Mosaic augmentation, BoF/BoS strategies | Bochkovskiy et al. |
| YOLOv5 | 2020 | Reimplementation bằng PyTorch, Auto-anchor, dễ triển khai | Glenn Jocher (Ultralytics) |
| YOLOv6 | 2022 | EfficientRep backbone, tối ưu cho triển khai công nghiệp | Meituan |
| YOLOv7 | 2022 | E-ELAN, compound scaling, auxiliary head training | Wang et al. |
| **YOLOv8** | **2023** | **C2f module, Decoupled Head, Anchor-free, Task-Aligned Assigner** | **Glenn Jocher (Ultralytics)** |

## 3.4. Lý do chọn YOLOv8s

YOLOv8s (Small) được lựa chọn cho dự án này vì các lý do sau:

1. **Cân bằng tốc độ - độ chính xác:** YOLOv8s có 11.12 triệu tham số, nằm ở mức cân bằng giữa mô hình siêu nhẹ (Nano) và mô hình lớn (Large/XLarge).
2. **Kiến trúc hiện đại:** Sử dụng các cải tiến mới nhất bao gồm anchor-free detection, decoupled head, và C2f module.
3. **Hệ sinh thái hoàn chỉnh:** Ultralytics cung cấp API Python và CLI thống nhất, hỗ trợ training, validation, prediction, export một cách đơn giản.
4. **Khả năng triển khai đa nền tảng:** Hỗ trợ xuất mô hình sang ONNX, TensorRT, CoreML, TFLite cho nhiều thiết bị khác nhau.
5. **Phù hợp phần cứng:** Hoạt động hiệu quả trên GPU phổ thông như RTX 3050 (4GB VRAM).

---

# 4. CƠ SỞ LÝ THUYẾT

## 4.1. Kiến trúc YOLOv8

YOLOv8 được thiết kế theo kiến trúc ba thành phần chính: **Backbone**, **Neck**, và **Head**.

![Kiến trúc tổng quan của mô hình YOLOv8s — gồm 3 thành phần: Backbone CSPDarknet, Neck FPN + PAN, và Detection Head Decoupled Head](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/yolov8_architecture_1791458913245.jpg)

### 4.1.1. Backbone — CSPDarknet với C2f Module

Backbone là thành phần **trích xuất đặc trưng** (feature extractor) từ ảnh đầu vào. YOLOv8 sử dụng kiến trúc CSPDarknet (Cross Stage Partial Darknet) với cải tiến quan trọng là module **C2f** (Cross Stage Partial Bottleneck with 2 Convolutions).

**Module C2f** là cải tiến so với module C3 trong YOLOv5, kết hợp ý tưởng từ ELAN (Efficient Layer Aggregation Network):
- Chia đặc trưng thành hai nhánh qua convolution $1 \times 1$
- Nhánh thứ hai đi qua $n$ bottleneck blocks
- Kết nối (concatenate) tất cả đầu ra trung gian
- Trộn lại bằng convolution $1 \times 1$ cuối cùng

Công thức tổng quát:
$$\mathbf{y} = \text{Conv}_{1\times1}\left(\text{Concat}\left(\mathbf{x}_1, B_1(\mathbf{x}_2), B_2(B_1(\mathbf{x}_2)), \ldots, B_n(\ldots)\right)\right)$$

Trong đó $B_i$ là bottleneck block thứ $i$, $\mathbf{x}_1$ và $\mathbf{x}_2$ là hai nhánh sau phép chia.

**SPPF (Spatial Pyramid Pooling — Fast):** Nằm ở cuối backbone, thực hiện pooling đa tỉ lệ để tổng hợp ngữ cảnh toàn cục mà không thay đổi kích thước ảnh đầu vào. SPPF sử dụng ba lớp MaxPool $5\times5$ nối tiếp (thay vì song song như SPP gốc) để giảm thời gian tính toán.

### 4.1.2. Neck — Feature Pyramid Network (FPN) + Path Aggregation Network (PAN)

Neck thực hiện nhiệm vụ **tổng hợp đặc trưng đa tỉ lệ** (multi-scale feature fusion), cho phép mô hình phát hiện đối tượng ở nhiều kích thước khác nhau.

- **FPN (Feature Pyramid Network):** Truyền thông tin ngữ nghĩa (semantic) từ tầng sâu (top-down) xuống tầng nông bằng upsampling và concatenation.
- **PAN (Path Aggregation Network):** Truyền thông tin chi tiết (localization) từ tầng nông (bottom-up) lên tầng sâu bằng downsampling.

Kết quả là ba bản đồ đặc trưng ở ba tỉ lệ khác nhau:
- **P3** ($80 \times 80$): Phát hiện đối tượng nhỏ
- **P4** ($40 \times 40$): Phát hiện đối tượng trung bình
- **P5** ($20 \times 20$): Phát hiện đối tượng lớn

### 4.1.3. Head — Decoupled Detection Head (Anchor-Free)

YOLOv8 sử dụng **Decoupled Head** — tách riêng nhánh phân loại (classification) và nhánh hồi quy vị trí (regression), thay vì dùng chung một nhánh như các phiên bản trước. Đây là cải tiến quan trọng giúp tăng hiệu năng.

**Anchor-Free Detection:** YOLOv8 loại bỏ hoàn toàn anchor boxes, thay vào đó dự đoán trực tiếp:
- **Bounding box:** 4 giá trị $(x_{center}, y_{center}, w, h)$ tại mỗi vị trí trên grid
- **Class probabilities:** Xác suất thuộc từng lớp

**Distribution Focal Loss (DFL):** YOLOv8 sử dụng DFL để hồi quy bounding box, mô hình hóa từng cạnh hộp như một phân phối xác suất rời rạc thay vì một giá trị duy nhất, giúp cải thiện độ chính xác localization.

### 4.1.4. Task-Aligned Assigner

YOLOv8 sử dụng **Task-Aligned Assigner** để gán nhãn (label assignment) trong quá trình huấn luyện, thay thế IoU-based assigner truyền thống. Chỉ số alignment được tính theo:

$$t = s^\alpha \cdot u^\beta$$

Trong đó:
- $s$: Classification score (điểm phân loại)
- $u$: IoU giữa predicted box và ground truth box
- $\alpha, \beta$: Siêu tham số cân bằng (mặc định $\alpha = 0.5, \beta = 6.0$)

## 4.2. Hàm mất mát (Loss Functions)

YOLOv8 sử dụng hàm mất mát tổng hợp gồm ba thành phần:

$$\mathcal{L}_{total} = \lambda_{box} \cdot \mathcal{L}_{box} + \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{dfl} \cdot \mathcal{L}_{dfl}$$

### 4.2.1. Box Loss — CIoU Loss
$$\mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

Trong đó:
- $IoU$: Intersection over Union giữa predicted box và ground truth
- $\rho(b, b^{gt})$: Khoảng cách Euclidean giữa tâm hai hộp
- $c$: Đường chéo hình chữ nhật nhỏ nhất bao quanh cả hai hộp
- $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$: Tham số consistency tỉ lệ
- $\alpha = \frac{v}{(1 - IoU) + v}$: Hệ số cân bằng

### 4.2.2. Classification Loss — Binary Cross-Entropy với Sigmoid
$$\mathcal{L}_{cls} = -\sum_{i} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### 4.2.3. Distribution Focal Loss (DFL)
$$\mathcal{L}_{dfl} = -\left[ (y_{i+1} - y) \log(S_i) + (y - y_i) \log(S_{i+1}) \right]$$

Trong đó $S_i$ là softmax probability, $y_i$ và $y_{i+1}$ là hai giá trị nguyên gần nhất với target $y$.

## 4.3. Các chỉ số đánh giá (Evaluation Metrics)

### 4.3.1. Precision (Độ chính xác)
$$\text{Precision} = \frac{TP}{TP + FP}$$
Tỉ lệ dự đoán đúng trong tổng số dự đoán dương tính. Precision cao nghĩa là ít báo động giả.

### 4.3.2. Recall (Độ bao phủ / Độ nhạy)
$$\text{Recall} = \frac{TP}{TP + FN}$$
Tỉ lệ đối tượng thực sự được phát hiện đúng. Recall cao nghĩa là ít bỏ sót.

### 4.3.3. F1-Score
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Trung bình điều hòa của Precision và Recall, phản ánh sự cân bằng giữa hai chỉ số.

### 4.3.4. mAP (mean Average Precision)
$$mAP = \frac{1}{N} \sum_{i=1}^{N} AP_i$$

Trong đó $AP_i$ là Average Precision của lớp thứ $i$, được tính bằng diện tích dưới đường cong Precision-Recall.

- **mAP@0.5:** AP tính tại ngưỡng IoU = 0.5
- **mAP@0.5:0.95:** Trung bình AP tại các ngưỡng IoU từ 0.5 đến 0.95 (bước 0.05), chỉ số nghiêm ngặt hơn.

### 4.3.5. IoU (Intersection over Union)
$$IoU = \frac{|A \cap B|}{|A \cup B|}$$
Đo mức độ trùng khớp giữa predicted bounding box ($A$) và ground truth bounding box ($B$).

## 4.4. Non-Maximum Suppression (NMS)

NMS là bước hậu xử lý loại bỏ các bounding box trùng lặp, giữ lại box có confidence cao nhất cho mỗi đối tượng. Thuật toán:

1. Sắp xếp tất cả box theo confidence score giảm dần
2. Chọn box có score cao nhất, thêm vào danh sách kết quả
3. Tính IoU giữa box được chọn với tất cả box còn lại
4. Loại bỏ các box có IoU > ngưỡng (mặc định 0.7)
5. Lặp lại bước 2-4 cho đến khi hết box

## 4.5. Kỹ thuật tăng cường dữ liệu (Data Augmentation)

YOLOv8 tích hợp nhiều kỹ thuật augmentation trong quá trình huấn luyện:

| Kỹ thuật | Mô tả | Tham số mặc định |
|:---|:---|:---:|
| **Mosaic** | Ghép 4 ảnh thành 1 ảnh huấn luyện | `mosaic=1.0` |
| **HSV Augmentation** | Biến đổi ngẫu nhiên Hue, Saturation, Value | `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4` |
| **Horizontal Flip** | Lật ngang ngẫu nhiên | `fliplr=0.5` |
| **Scale** | Co giãn ngẫu nhiên | `scale=0.5` |
| **Translation** | Dịch chuyển ngẫu nhiên | `translate=0.1` |
| **Erasing** | Xóa ngẫu nhiên vùng ảnh | `erasing=0.4` |
| **RandAugment** | Auto-augmentation policy | `auto_augment=randaugment` |

---

# 5. PHƯƠNG PHÁP

## 5.1. Kiến trúc hệ thống tổng quan

![Kiến trúc tổng quan hệ thống phát hiện đám cháy và khói thời gian thực sử dụng YOLOv8s — từ nguồn dữ liệu đầu vào đến hệ thống cảnh báo](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/fire_detection_system_1791458926625.jpg)

Hệ thống được thiết kế theo pipeline gồm 5 giai đoạn:

1. **Thu thập dữ liệu đầu vào:** Từ camera CCTV, drone, hoặc ảnh vệ tinh
2. **Tiền xử lý (Preprocessing):** Resize ảnh về $640 \times 640$, chuẩn hóa giá trị pixel
3. **Suy luận mô hình (Inference):** YOLOv8s thực hiện dự đoán trên GPU
4. **Hậu xử lý (Postprocessing):** Non-Maximum Suppression, lọc theo ngưỡng tin cậy
5. **Đầu ra (Output):** Hiển thị kết quả phát hiện, kích hoạt cảnh báo

## 5.2. Tập dữ liệu (Dataset)

### 5.2.1. Nguồn dữ liệu
Tập dữ liệu được thu thập và tổng hợp từ nhiều nguồn, bao gồm các tập dữ liệu công khai về fire detection trên Roboflow Universe, ảnh tự thu thập, và các tập dữ liệu benchmark. Dữ liệu bao gồm nhiều kịch bản đa dạng:
- Đám cháy trong nhà (indoor fire)
- Cháy rừng (wildfire)
- Cháy phương tiện giao thông
- Khói trắng, khói đen, khói ban ngày/ban đêm
- Các trường hợp gây nhiễu: ánh hoàng hôn, đèn LED, sương mù

### 5.2.2. Phân chia dữ liệu

Dữ liệu được chia thành ba tập theo tỉ lệ chuẩn:

| Tập dữ liệu | Số lượng ảnh | Mục đích |
|:---|:---:|:---|
| **Training (Huấn luyện)** | ~7,000 ảnh | Huấn luyện mô hình |
| **Validation (Xác thực)** | 2,000 ảnh | Theo dõi hiệu năng và điều chỉnh siêu tham số |
| **Test (Kiểm thử)** | 1,000 ảnh | Đánh giá cuối cùng, độc lập |

### 5.2.3. Định dạng nhãn YOLO

Mỗi ảnh có một file nhãn `.txt` tương ứng, chứa các dòng có định dạng:

```
<class_id> <x_center> <y_center> <width> <height>
```

Trong đó tất cả giá trị tọa độ được chuẩn hóa về khoảng $[0, 1]$ so với kích thước ảnh.

### 5.2.4. Cấu hình dữ liệu (data.yaml)

```yaml
path: c:/Users/Administrator/Documents/yolov8
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: Fire
  1: Smoke
```

### 5.2.5. Phân bố nhãn trong tập dữ liệu

![Phân bố nhãn labels trong tập dữ liệu huấn luyện — thể hiện sự mất cân bằng giữa lớp Fire và Smoke, cùng phân bố vị trí và kích thước bounding box](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/labels.jpg)

![Biểu đồ tương quan correlogram giữa các tham số bounding box — phân tích mối quan hệ giữa tọa độ x, y, chiều rộng và chiều cao của các nhãn](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/labels_correlogram.jpg)

## 5.3. Cấu hình huấn luyện

### 5.3.1. Phần cứng và phần mềm

| Thành phần | Chi tiết |
|:---|:---|
| **GPU** | NVIDIA GeForce RTX 3050 Laptop (4GB VRAM) |
| **CUDA** | 12.1 |
| **Python** | 3.11.4 |
| **PyTorch** | 2.5.1 + CUDA 12.1 |
| **Ultralytics** | 8.3.108 |
| **Hệ điều hành** | Windows |

### 5.3.2. Siêu tham số huấn luyện

| Siêu tham số | Giá trị | Ghi chú |
|:---|:---:|:---|
| **Mô hình cơ sở** | `yolov8s.pt` | Pre-trained trên COCO dataset |
| **Epochs** | 50 | Số chu kỳ huấn luyện |
| **Batch size** | 16 | Tối ưu cho 4GB VRAM |
| **Image size** | 640 × 640 | Kích thước ảnh đầu vào |
| **Optimizer** | Auto (SGD) | Tự động chọn bởi Ultralytics |
| **Learning rate (lr0)** | 0.01 | Tốc độ học khởi tạo |
| **Learning rate final (lrf)** | 0.01 | Tốc độ học cuối cùng |
| **Momentum** | 0.937 | Hệ số momentum |
| **Weight decay** | 0.0005 | Chính quy hóa L2 |
| **Warmup epochs** | 3.0 | Số epoch khởi động |
| **IoU threshold** | 0.7 | Ngưỡng NMS |
| **AMP** | True | Mixed Precision Training |
| **Close mosaic** | 10 | Tắt mosaic ở 10 epoch cuối |
| **Workers** | 2 | Tối ưu cho Windows |
| **Pretrained** | True | Transfer learning từ COCO |

### 5.3.3. Mã nguồn huấn luyện

```python
import torch
from ultralytics import YOLO

def main():
    # Kiểm tra GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.2f} GB")

    # Khởi tạo mô hình YOLOv8s (pre-trained trên COCO)
    model = YOLO("yolov8s.pt")

    # Huấn luyện
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project="runs_fire_detection",
        name="train_yolov8s",
        save=True
    )

    # Đánh giá trên tập test
    metrics = model.val(data="data.yaml", split="test", device=0)

if __name__ == "__main__":
    main()
```

### 5.3.4. Quá trình Transfer Learning

Transfer Learning là kỹ thuật quan trọng trong dự án này. Mô hình YOLOv8s được khởi tạo từ trọng số pre-trained trên tập COCO (330K ảnh, 80 lớp đối tượng). Các lớp backbone đã học được các đặc trưng tổng quát (cạnh, texture, hình dạng) từ COCO, và được fine-tune trên dữ liệu Fire/Smoke cụ thể. Điều này giúp:
- Giảm thời gian huấn luyện đáng kể
- Cải thiện hiệu năng, đặc biệt khi dữ liệu domain-specific hạn chế
- Hội tụ nhanh hơn

---

# 6. THỰC NGHIỆM VÀ KẾT QUẢ

## 6.1. Quá trình huấn luyện

### 6.1.1. Thông tin mô hình

| Hạng mục | Chi tiết |
|:---|:---|
| **Kiến trúc** | YOLOv8s (72 layers) |
| **Tổng tham số** | 11.12 triệu |
| **Độ phức tạp** | 28.4 GFLOPs |
| **Thời gian huấn luyện** | ~3.65 giờ (13,154 giây) |
| **Số epochs** | 50 |

### 6.1.2. Minh họa batch huấn luyện

![Batch huấn luyện đầu tiên train_batch0 — minh họa dữ liệu đầu vào với augmentation Mosaic, HSV, Flip và các ground truth bounding box cho lớp Fire và Smoke](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/train_batch0.jpg)

### 6.1.3. Biểu đồ quá trình huấn luyện (Training Curves)

![Toàn bộ biểu đồ quá trình huấn luyện qua 50 epochs — bao gồm: Box Loss, Classification Loss, DFL Loss train và val, cùng chỉ số Precision, Recall, mAP@0.5 và mAP@0.5:0.95](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/results.png)

**Phân tích đường cong huấn luyện:**

- **Box Loss (train/val):** Giảm đều đặn qua các epoch, cho thấy mô hình ngày càng dự đoán bounding box chính xác hơn.
- **Classification Loss:** Giảm mạnh từ 2.30 xuống 0.62 (train) và từ 2.56 xuống 0.77 (val), chứng tỏ mô hình phân biệt tốt giữa Fire và Smoke.
- **DFL Loss:** Giảm ổn định, phản ánh sự cải thiện trong hồi quy phân phối vị trí bounding box.
- **Precision & Recall:** Tăng dần và ổn định từ epoch 20 trở đi.
- **mAP@0.5:** Tăng nhanh trong 15 epoch đầu, sau đó hội tụ quanh 75-77%.
- **Không có dấu hiệu overfitting:** Khoảng cách giữa train loss và val loss nhỏ và ổn định.

## 6.2. Kết quả đánh giá định lượng

### 6.2.1. Kết quả trên tập Kiểm thử độc lập (Test Set — 1,000 ảnh)

| Phân loại (Class) | Số nhãn (Instances) | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Toàn bộ (All)** | **2,177** | **78.8%** | **73.9%** | **77.4%** | **47.0%** |
| **Lửa (Fire)** | 1,839 | **86.1%** | **81.1%** | **87.1%** | **56.1%** |
| **Khói (Smoke)** | 338 | **71.5%** | **66.7%** | **67.7%** | **37.8%** |

- **mAP@0.75 trên tập Test:** **50.6%**

### 6.2.2. Kết quả trên tập Xác thực (Validation Set — 2,000 ảnh)

| Phân loại (Class) | Số nhãn (Instances) | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Toàn bộ (All)** | **4,312** | **80.1%** | **69.5%** | **75.1%** | **45.1%** |
| **Lửa (Fire)** | 3,627 | **87.5%** | **79.2%** | **85.5%** | **55.0%** |
| **Khói (Smoke)** | 685 | **72.7%** | **59.9%** | **64.8%** | **35.3%** |

- **mAP@0.75 trên tập Val:** **48.0%**

### 6.2.3. Tốc độ suy luận (Inference Speed)

| Giai đoạn | Thời gian | Tốc độ |
|:---|:---:|:---:|
| **Tiền xử lý (Preprocess)** | 0.3 ms/ảnh | — |
| **Suy luận (Inference)** | 10.7 ms/ảnh | ~93.5 FPS |
| **Hậu xử lý NMS** | 1.0 ms/ảnh | — |
| **Tổng (End-to-End)** | **~12.0 ms/ảnh** | **~83 FPS** |

> [!IMPORTANT]
> Tốc độ 83 FPS vượt xa yêu cầu camera giám sát thông thường (25-30 FPS), đáp ứng hoàn toàn khả năng giám sát trực tiếp đa luồng (multi-stream monitoring).

## 6.3. Phân tích các đường cong đánh giá

### 6.3.1. Đường cong Precision-Recall (PR Curve)

![Đường cong Precision-Recall — thể hiện mối quan hệ giữa Precision và Recall ở các ngưỡng confidence khác nhau cho từng lớp Fire và Smoke](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/PR_curve.png)

**Phân tích:**
- Đường PR của lớp **Fire** nằm cao và rộng, cho thấy mô hình duy trì precision cao ngay cả khi recall tăng. AP = 87.1%.
- Đường PR của lớp **Smoke** thấp hơn, phản ánh sự khó khăn trong phát hiện khói loãng và khói hòa lẫn nền. AP = 67.7%.
- Diện tích dưới đường cong (AUC) tổng thể đạt mức tốt.

### 6.3.2. Đường cong F1-Score

![Đường cong F1-Score theo ngưỡng Confidence — xác định ngưỡng tối ưu để cân bằng giữa Precision và Recall](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/F1_curve.png)

**Phân tích:**
- Đường F1 đạt đỉnh tại ngưỡng confidence khoảng **0.3-0.4**, cho thấy đây là ngưỡng tối ưu cho cân bằng giữa phát hiện sớm và giảm báo động giả.

### 6.3.3. Đường cong Precision và Recall

````carousel
![Đường cong Precision theo ngưỡng Confidence — Precision tăng khi tăng ngưỡng, phản ánh mô hình tự tin hơn ở ngưỡng cao](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/P_curve.png)
<!-- slide -->
![Đường cong Recall theo ngưỡng Confidence — Recall giảm khi tăng ngưỡng, phản ánh sự đánh đổi giữa phát hiện đầy đủ và lọc chặt](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/R_curve.png)
````

## 6.4. Ma trận nhầm lẫn (Confusion Matrix)

````carousel
![Ma trận nhầm lẫn Confusion Matrix — thể hiện tỉ lệ phân loại đúng sai giữa các lớp Fire, Smoke và Background](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/confusion_matrix.png)
<!-- slide -->
![Ma trận nhầm lẫn chuẩn hóa Normalized Confusion Matrix — cho thấy tỉ lệ phần trăm phân loại của từng lớp](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/confusion_matrix_normalized.png)
````

**Phân tích ma trận nhầm lẫn:**
- Lớp **Fire** có tỉ lệ phân loại đúng cao, ít bị nhầm lẫn với Background hoặc Smoke.
- Lớp **Smoke** có tỉ lệ bỏ sót (FN) cao hơn, đặc biệt với khói loãng bị phân loại nhầm thành Background.
- Hiện tượng nhầm lẫn giữa Fire và Smoke tương đối thấp, cho thấy mô hình phân biệt tốt hai lớp này.

## 6.5. Kết quả trực quan (Visual Results)

### 6.5.1. So sánh Ground Truth và Prediction trên tập Validation

````carousel
![Ground Truth — Nhãn thật trên batch validation, các bounding box được gán thủ công bởi annotator](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/val_batch0_labels.jpg)
<!-- slide -->
![Prediction — Dự đoán của mô hình trên cùng batch validation, bounding box và confidence score do YOLOv8s tự dự đoán](C:/Users/Administrator/.gemini/antigravity-ide/brain/f8d47223-e66d-4dc2-aa99-c9946a977f81/val_batch0_pred.jpg)
````

**Nhận xét trực quan:**
- Mô hình phát hiện chính xác vị trí và kích thước ngọn lửa trong đa số trường hợp.
- Các dự đoán khớp tốt với ground truth, confidence score thường nằm trong khoảng 0.5-0.9.
- Một số trường hợp khói loãng hoặc khói xa bị bỏ sót hoặc có confidence thấp.

## 6.6. Phân tích chuyên sâu

### 6.6.1. Điểm mạnh nổi bật

1. **Khả năng tổng quát hóa xuất sắc (Không Overfitting):**
   - mAP@0.5 trên tập Test đạt **77.4%**, cao hơn tập Validation (**75.1%**).
   - Điều này chứng minh mô hình không bị hiện tượng "học vẹt", các đặc trưng trích xuất có khả năng thích nghi cao với dữ liệu thực tế mới.

2. **Độ nhạy và độ tin cậy cao cho lớp Fire:**
   - Precision **86.1%**, Recall **81.1%**, mAP@0.5 đạt **87.1%**.
   - Với màu sắc đặc trưng (cam, đỏ, vàng) và độ sáng tương phản cao, ngọn lửa được phát hiện chính xác.

3. **Tốc độ xử lý vượt trội:**
   - **83 FPS** trên GPU RTX 3050, gấp gần 3 lần yêu cầu camera giám sát thông thường.
   - Cho phép giám sát đồng thời 2-3 luồng camera trên một GPU duy nhất.

4. **Quá trình hội tụ ổn định:**
   - Hàm mất mát giảm mượt mà, không có hiện tượng dao động bất thường.

### 6.6.2. Hạn chế và thách thức

1. **Mất cân bằng dữ liệu (Class Imbalance):**
   - Tỉ lệ nhãn Fire:Smoke = 5.4:1, gây thiên lệch về lớp Fire.
   - Lớp Smoke có Recall chỉ **66.7%**, bỏ sót ~33% trường hợp.

2. **Đặc thù hình học của khói:**
   - Khói không có biên dạng cố định (amorphous object).
   - Khói loãng dễ hòa lẫn vào nền trời, mây, sương mù.
   - Khói đen và khói trắng có đặc trưng thị giác khác biệt lớn.

3. **mAP@0.5:0.95 còn hạn chế:**
   - Đạt **47.0%** (tổng) và **37.8%** (Smoke), cho thấy localization chưa thật sự chặt chẽ ở ngưỡng IoU cao.

---

# 7. KẾT LUẬN

## 7.1. Tóm tắt kết quả

Dự án đã **hoàn thành thành công** việc xây dựng và đánh giá hệ thống phát hiện đám cháy và khói thời gian thực sử dụng mô hình YOLOv8s. Các mục tiêu đề ra đều được đáp ứng:

| Mục tiêu | Kết quả | Đánh giá |
|:---|:---|:---:|
| mAP@0.5 ≥ 75% trên tập Test | **77.4%** | ✅ Đạt |
| Tốc độ ≥ 30 FPS trên RTX 3050 | **83 FPS** | ✅ Vượt mục tiêu |
| Phát hiện đồng thời Fire & Smoke | Fire: 87.1%, Smoke: 67.7% | ✅ Đạt |
| Không overfitting | Test mAP > Val mAP | ✅ Đạt |

## 7.2. Đóng góp chính

1. **Xây dựng thành công** mô hình YOLOv8s fine-tuned cho bài toán fire detection, đạt hiệu năng cao trên cả hai tập Validation và Test.
2. **Chứng minh tính khả thi** của việc sử dụng mô hình one-stage detector hiện đại cho giám sát cháy thời gian thực trên phần cứng GPU phổ thông.
3. **Phân tích toàn diện** các điểm mạnh và hạn chế, đặc biệt về thách thức phát hiện khói trong các điều kiện khó.
4. **Đề xuất cấu hình triển khai** phù hợp với các kịch bản thực tế (cảnh báo sớm vs. kích hoạt thiết bị dập lửa).

## 7.3. Kiến nghị triển khai thực tế

1. **Chế độ cảnh báo sớm (Early Fire Alarm):** Thiết lập `conf = 0.25 - 0.35` để tăng Recall, ưu tiên phát hiện sớm.
2. **Chế độ kích hoạt thiết bị dập lửa:** Thiết lập `conf = 0.50 - 0.60` để giảm báo động giả.
3. **Giám sát đa luồng:** Có thể xử lý 2-3 camera đồng thời trên một GPU RTX 3050.

---

# 8. HƯỚNG PHÁT TRIỂN

## 8.1. Cải thiện hiệu năng mô hình

1. **Cân bằng dữ liệu lớp Smoke:**
   - Bổ sung 1,500 - 2,000 ảnh khói đa dạng (khói trắng, khói đen, khói ban ngày/đêm).
   - Áp dụng kỹ thuật oversampling hoặc class-weighted loss.

2. **Tăng số epoch huấn luyện:**
   - Mở rộng lên 70-80 epochs kết hợp Cosine Annealing Learning Rate Schedule.
   - Dự kiến cải thiện 2-4% mAP.

3. **Thử nghiệm kiến trúc lớn hơn:**
   - YOLOv8m (Medium) hoặc YOLOv8l (Large) trên GPU mạnh hơn.
   - So sánh với các mô hình khác: RT-DETR, YOLOv9, YOLOv10.

## 8.2. Mở rộng chức năng

1. **Phát hiện trên video thời gian thực:**
   - Tích hợp xử lý video stream từ camera IP/RTSP.
   - Kết hợp tracking (DeepSORT, BoT-SORT) để theo dõi diễn biến đám cháy.

2. **Phân đoạn ngữ nghĩa (Semantic Segmentation):**
   - Sử dụng YOLOv8-seg để phân đoạn chính xác vùng lửa/khói thay vì bounding box.
   - Cung cấp thông tin chi tiết hơn về quy mô và hình dạng đám cháy.

3. **Ước lượng mức độ nghiêm trọng:**
   - Phân loại thêm mức độ cháy: nhỏ / trung bình / lớn.
   - Dựa trên kích thước bounding box và confidence score.

## 8.3. Triển khai Edge Computing

1. **Xuất mô hình tối ưu:**
   - ONNX Runtime cho server deployment.
   - TensorRT cho NVIDIA Jetson (Nano, Xavier, Orin).
   - TFLite cho thiết bị Android.

2. **Tối ưu hóa mô hình:**
   - Quantization (INT8/FP16) để giảm kích thước và tăng tốc.
   - Pruning để loại bỏ các tham số không cần thiết.
   - Knowledge Distillation từ mô hình lớn sang mô hình nhỏ.

## 8.4. Tích hợp hệ thống IoT

1. **Kết nối với hệ thống cảnh báo:**
   - Gửi push notification qua Firebase Cloud Messaging.
   - Kích hoạt sprinkler / alarm qua relay IoT.
   - Dashboard giám sát web-based.

2. **Kết hợp dữ liệu đa phương thức:**
   - Tích hợp dữ liệu từ cảm biến nhiệt, cảm biến khí.
   - Fusion dữ liệu camera hồng ngoại (thermal) và camera RGB.

---

# 9. TÀI LIỆU THAM KHẢO

## Bài báo khoa học (arXiv & Hội nghị)

[1] **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A.** (2016). "You Only Look Once: Unified, Real-Time Object Detection." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv: [1506.02640](https://arxiv.org/abs/1506.02640)

[2] **Redmon, J., & Farhadi, A.** (2017). "YOLO9000: Better, Faster, Stronger." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv: [1612.08242](https://arxiv.org/abs/1612.08242)

[3] **Redmon, J., & Farhadi, A.** (2018). "YOLOv3: An Incremental Improvement." *arXiv preprint*. arXiv: [1804.02767](https://arxiv.org/abs/1804.02767)

[4] **Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M.** (2020). "YOLOv4: Optimal Speed and Accuracy of Object Detection." *arXiv preprint*. arXiv: [2004.10934](https://arxiv.org/abs/2004.10934)

[5] **Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M.** (2023). "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv: [2207.02696](https://arxiv.org/abs/2207.02696)

[6] **Jocher, G., Chaurasia, A., & Qiu, J.** (2023). "Ultralytics YOLOv8." *GitHub repository*. Disponible: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

[7] **Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S.** (2017). "Feature Pyramid Networks for Object Detection." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv: [1612.03144](https://arxiv.org/abs/1612.03144)

[8] **Liu, S., Qi, L., Qin, H., Shi, J., & Jia, J.** (2018). "Path Aggregation Network for Instance Segmentation." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv: [1803.01534](https://arxiv.org/abs/1803.01534)

[9] **Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D.** (2020). "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression." *Proceedings of the AAAI Conference on Artificial Intelligence*. arXiv: [1911.08287](https://arxiv.org/abs/1911.08287)

[10] **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). "Focal Loss for Dense Object Detection." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. arXiv: [1708.02002](https://arxiv.org/abs/1708.02002)

## Bài báo về Fire Detection

[11] **Li, P., & Zhao, W.** (2020). "Image Fire Detection Algorithms Based on Convolutional Neural Networks." *Case Studies in Thermal Engineering*, 19, 100625. DOI: [10.1016/j.csite.2020.100625](https://doi.org/10.1016/j.csite.2020.100625)

[12] **Saponara, S., Elhanashi, A., & Gagliardi, A.** (2021). "Real-time video fire/smoke detection based on CNN in antifire surveillance systems." *Journal of Real-Time Image Processing*, 18, 889-900. DOI: [10.1007/s11554-020-01044-0](https://doi.org/10.1007/s11554-020-01044-0)

[13] **Park, M., Ko, B. C.** (2020). "Two-Step Real-Time Night-Time Fire Detection in an Urban Environment Using Static ELASTIC-YOLOv3 and Temporal Fire-Tube." *Sensors*, 20(8), 2202. DOI: [10.3390/s20082202](https://doi.org/10.3390/s20082202)

[14] **Abdusalomov, A. B., Islam, B. M. S., Nasimov, R., Mukhiddinov, M., & Whangbo, T. K.** (2023). "An Improved Forest Fire Detection Method Based on the Detectron2 Model and a Deep Learning Approach." *Sensors*, 23(3), 1512. DOI: [10.3390/s23031512](https://doi.org/10.3390/s23031512)

[15] **Wang, D., Deng, L., & Ni, B.** (2022). "Real-time Smoke and Fire Detection Based on Improved YOLOv5." *2022 IEEE 5th International Conference on Electronics Technology (ICET)*. DOI: [10.1109/ICET55676.2022.9824403](https://doi.org/10.1109/ICET55676.2022.9824403)

## Tài liệu kỹ thuật và Framework

[16] **Ultralytics YOLOv8 Documentation.** Disponible: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

[17] **PyTorch Documentation.** Disponible: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

[18] **Roboflow Universe — Fire and Smoke Datasets.** Disponible: [https://universe.roboflow.com/](https://universe.roboflow.com/)

[19] **NVIDIA CUDA Toolkit Documentation.** Disponible: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)

## Sách tham khảo

[20] **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. Disponible: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

[21] **Géron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition). O'Reilly Media.

---

> [!NOTE]
> **Tài nguyên dự án:**
> - Trọng số mô hình tốt nhất: `runs_fire_detection/train_yolov8s/weights/best.pt`
> - Trọng số chu kỳ cuối: `runs_fire_detection/train_yolov8s/weights/last.pt`
> - File kết quả huấn luyện: `runs_fire_detection/train_yolov8s/results.csv`
> - Báo cáo đánh giá gốc: `BAO_CAO_DANH_GIA.md`

---

*Dự án được thực hiện trên môi trường Windows, sử dụng NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM), Python 3.11.4, PyTorch 2.5.1 + CUDA 12.1, Ultralytics YOLOv8 v8.3.108.*

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, ListFlowable, ListItem
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm, mm

# ============================================================
# 1. ĐĂNG KÝ FONT TIẾNG VIỆT
# ============================================================
pdfmetrics.registerFont(TTFont('Arial', r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', r'C:\Windows\Fonts\ariali.ttf'))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', r'C:\Windows\Fonts\arialbi.ttf'))

# ============================================================
# 2. NUMBERED CANVAS (Header + Footer + Page Number)
# ============================================================
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Arial", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header
        if self._pageNumber > 1:
            self.drawString(50, 842 - 30, "Dự Án Phát Hiện Đám Cháy & Khói — YOLOv8s Fire & Smoke Detection System")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(50, 842 - 36, 595 - 50, 842 - 36)

        # Footer
        self.drawString(50, 25, "Hệ Thống Phát Hiện Đám Cháy & Khói Thời Gian Thực — YOLOv8s")
        self.drawRightString(595 - 50, 25, f"Trang {self._pageNumber} / {page_count}")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(50, 38, 595 - 50, 38)

        self.restoreState()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
IMG_DIR = r"runs_fire_detection\train_yolov8s"

def img_path(filename):
    return os.path.join(IMG_DIR, filename)

def add_image(story, filepath, width, caption, caption_style):
    """Add image with caption if file exists."""
    if os.path.exists(filepath):
        story.append(Image(filepath, width=width, height=width * 0.6))
        story.append(Paragraph(caption, caption_style))
        story.append(Spacer(1, 8))
    else:
        story.append(Paragraph(f"<i>[Hình ảnh không tìm thấy: {filepath}]</i>", caption_style))

def make_table(headers, rows, col_widths, header_bg="#1e3a8a"):
    """Create a styled table."""
    all_data = [headers] + rows
    t = Table(all_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Arial-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('FONTNAME', (0, 1), (-1, -1), 'Arial'),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ]))
    return t


# ============================================================
# 4. MAIN FUNCTION
# ============================================================
def create_full_report():
    output_file = "DU_AN_PHAT_HIEN_DAM_CHAY_YOLOV8S.pdf"

    doc = SimpleDocTemplate(
        output_file, pagesize=A4,
        leftMargin=45, rightMargin=45,
        topMargin=45, bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # -- Custom Styles --
    title = ParagraphStyle('Title', parent=styles['Normal'], fontName='Arial-Bold',
                           fontSize=18, leading=24, textColor=colors.HexColor("#0f172a"), alignment=1, spaceAfter=4)
    subtitle = ParagraphStyle('Subtitle', parent=styles['Normal'], fontName='Arial-Italic',
                              fontSize=11, leading=16, textColor=colors.HexColor("#475569"), alignment=1, spaceAfter=10)
    h1 = ParagraphStyle('H1', parent=styles['Normal'], fontName='Arial-Bold',
                        fontSize=14, leading=20, textColor=colors.HexColor("#0f172a"), spaceBefore=14, spaceAfter=6)
    h2 = ParagraphStyle('H2', parent=styles['Normal'], fontName='Arial-Bold',
                        fontSize=11, leading=16, textColor=colors.HexColor("#1e3a8a"), spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle('H3', parent=styles['Normal'], fontName='Arial-Bold',
                        fontSize=9.5, leading=14, textColor=colors.HexColor("#334155"), spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle('Body', parent=styles['Normal'], fontName='Arial',
                          fontSize=9.5, leading=14, textColor=colors.HexColor("#334155"), spaceAfter=5)
    body_bold = ParagraphStyle('BodyBold', parent=body, fontName='Arial-Bold')
    bullet = ParagraphStyle('Bullet', parent=body, leftIndent=15, spaceAfter=3)
    caption = ParagraphStyle('Caption', parent=styles['Normal'], fontName='Arial-Italic',
                             fontSize=8.5, leading=12, textColor=colors.HexColor("#64748b"),
                             alignment=1, spaceAfter=6)
    code_style = ParagraphStyle('Code', parent=styles['Normal'], fontName='Courier',
                                fontSize=8, leading=11, textColor=colors.HexColor("#1e293b"),
                                backColor=colors.HexColor("#f1f5f9"), leftIndent=10, spaceAfter=6)

    # ============================================================
    story = []

    # ========== TRANG BÌA ==========
    story.append(Spacer(1, 80))
    story.append(Paragraph("DỰ ÁN", ParagraphStyle('Pre', parent=title, fontSize=13, spaceAfter=2)))
    story.append(Paragraph("HỆ THỐNG PHÁT HIỆN ĐÁM CHÁY VÀ KHÓI<br/>THỜI GIAN THỰC SỬ DỤNG MÔ HÌNH YOLOv8s", title))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=10))
    story.append(Paragraph("<i>Fire and Smoke Real-Time Detection System<br/>Using YOLOv8s Deep Learning Model</i>", subtitle))
    story.append(Spacer(1, 30))

    # Info table
    info_data = [
        ["Lĩnh vực", "Thị giác máy tính — Phát hiện đối tượng (Object Detection)"],
        ["Công nghệ", "YOLOv8s (Ultralytics), PyTorch 2.5.1, CUDA 12.1"],
        ["Phần cứng", "NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)"],
        ["Ngôn ngữ", "Python 3.11.4"],
        ["Mô hình", "YOLOv8s — 11.12M params, 28.4 GFLOPs, 72 layers"],
    ]
    info_t = Table(info_data, colWidths=[100, 400])
    info_t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Arial-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Arial'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#334155")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_t)
    story.append(PageBreak())

    # ========== MỤC LỤC ==========
    story.append(Paragraph("MỤC LỤC", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=10))
    toc_items = [
        "1. Lời mở đầu",
        "2. Tổng quan",
        "3. Giới thiệu chung",
        "4. Cơ sở lý thuyết",
        "5. Phương pháp",
        "6. Thực nghiệm và kết quả",
        "7. Kết luận",
        "8. Hướng phát triển",
        "9. Tài liệu tham khảo"
    ]
    for item in toc_items:
        story.append(Paragraph(item, ParagraphStyle('TOC', parent=body, fontSize=11, leading=18, leftIndent=20)))
    story.append(PageBreak())

    # ========== CHƯƠNG 1: LỜI MỞ ĐẦU ==========
    story.append(Paragraph("CHƯƠNG 1: LỜI MỞ ĐẦU", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))
    story.append(Paragraph(
        "Cháy nổ là một trong những hiểm họa nghiêm trọng nhất đối với con người, tài sản và môi trường. "
        "Theo thống kê của Cục Cảnh sát Phòng cháy, Chữa cháy và Cứu nạn, Cứu hộ (Bộ Công an Việt Nam), "
        "mỗi năm cả nước ghi nhận hàng ngàn vụ cháy gây thiệt hại lớn về người và tài sản. "
        "Trên phạm vi toàn cầu, Tổ chức Phòng cháy Quốc tế (NFPA — National Fire Protection Association) "
        "ước tính thiệt hại do hỏa hoạn lên tới hàng chục tỷ USD mỗi năm. "
        "Điểm chung của phần lớn các vụ cháy lớn là <b>sự chậm trễ trong phát hiện ban đầu</b> — "
        "khi ngọn lửa hoặc cột khói còn nhỏ và có thể kiểm soát được.", body))

    story.append(Paragraph("Các hệ thống phát hiện cháy truyền thống (cảm biến nhiệt, cảm biến khói) tồn tại nhiều hạn chế:", body))
    for item in [
        "<b>Phạm vi giám sát hẹp:</b> Mỗi cảm biến chỉ bao phủ một vùng không gian nhỏ.",
        "<b>Tốc độ phản hồi chậm:</b> Cảm biến nhiệt cần nhiệt độ tăng đến ngưỡng mới kích hoạt.",
        "<b>Tỉ lệ báo động giả cao:</b> Khói từ nấu ăn, bụi bẩn có thể kích hoạt sai lệch.",
        "<b>Không cung cấp thông tin trực quan:</b> Chỉ đưa ra tín hiệu nhị phân (có/không)."
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Paragraph(
        "Sự phát triển của <b>Trí tuệ nhân tạo (AI)</b> và <b>Thị giác máy tính (Computer Vision)</b> "
        "đã mở ra hướng tiếp cận mới: sử dụng camera giám sát kết hợp mô hình học sâu (deep learning) "
        "để phát hiện đám cháy và khói từ hình ảnh/video theo thời gian thực.", body))
    story.append(Paragraph(
        "Dự án này tập trung xây dựng và đánh giá hệ thống phát hiện đám cháy và khói thời gian thực "
        "sử dụng mô hình <b>YOLOv8s</b> (You Only Look Once version 8 — Small) — một trong những "
        "kiến trúc phát hiện đối tượng hiện đại nhất, với ưu thế vượt trội về cân bằng giữa tốc độ và độ chính xác.", body))

    # ========== CHƯƠNG 2: TỔNG QUAN ==========
    story.append(PageBreak())
    story.append(Paragraph("CHƯƠNG 2: TỔNG QUAN", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))

    story.append(Paragraph("2.1. Bối cảnh và Tính cấp thiết", h2))
    story.append(Paragraph(
        "Phát hiện cháy sớm (early fire detection) là yếu tố then chốt trong việc giảm thiểu thiệt hại. "
        "Nghiên cứu cho thấy nếu đám cháy được phát hiện trong <b>3 phút đầu tiên</b>, "
        "khả năng dập tắt thành công lên tới <b>90%</b>. Tuy nhiên, sau 5-10 phút, đám cháy "
        "có thể lan rộng gấp nhiều lần và trở nên khó kiểm soát.", body))
    story.append(Paragraph(
        "Hệ thống camera giám sát (CCTV) đã được lắp đặt phổ biến nhưng chỉ ghi hình thụ động. "
        "Việc tích hợp AI cho phép <b>tự động hóa</b> giám sát, biến camera thành bộ phát hiện thông minh chủ động.", body))

    story.append(Paragraph("2.2. Mục tiêu của dự án", h2))
    objectives = [
        "Xây dựng mô hình phát hiện hai lớp đối tượng: <b>Ngọn lửa (Fire)</b> và <b>Khói (Smoke)</b>.",
        "Đạt độ chính xác mAP@0.5 ≥ 75% trên tập kiểm thử độc lập.",
        "Đảm bảo tốc độ xử lý thời gian thực ≥ 30 FPS trên GPU RTX 3050.",
        "Đánh giá toàn diện thông qua Precision, Recall, mAP, Confusion Matrix, và các đường cong P-R, F1.",
        "Đề xuất giải pháp tối ưu hóa và hướng triển khai thực tế."
    ]
    for i, obj in enumerate(objectives, 1):
        story.append(Paragraph(f"<b>{i}.</b> {obj}", bullet))

    # ========== CHƯƠNG 3: GIỚI THIỆU CHUNG ==========
    story.append(PageBreak())
    story.append(Paragraph("CHƯƠNG 3: GIỚI THIỆU CHUNG", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))

    story.append(Paragraph("3.1. Bài toán phát hiện đối tượng (Object Detection)", h2))
    story.append(Paragraph(
        "Phát hiện đối tượng là bài toán cốt lõi trong Thị giác máy tính, xác định đồng thời "
        "<b>vị trí</b> (localization) và <b>phân loại</b> (classification) các đối tượng trong ảnh. "
        "Kết quả gồm: Bounding box (x, y, w, h), Class label, và Confidence score.", body))

    story.append(Paragraph("3.2. Phương pháp hai giai đoạn vs. một giai đoạn", h2))
    story.append(Paragraph(
        "<b>Two-Stage Detectors</b> (R-CNN, Faster R-CNN): Sinh vùng ứng viên → Phân loại. Chính xác nhưng chậm.", body))
    story.append(Paragraph(
        "<b>One-Stage Detectors</b> (YOLO, SSD, RetinaNet): Dự đoán đồng thời trong một lần forward pass. Nhanh, phù hợp real-time.", body))

    story.append(Paragraph("3.3. Lịch sử phát triển dòng YOLO", h2))
    yolo_data = [
        ["Phiên bản", "Năm", "Đóng góp chính"],
        ["YOLOv1", "2016", "Mô hình one-stage đầu tiên, chia ảnh thành lưới S×S"],
        ["YOLOv2", "2017", "Batch Normalization, anchor boxes, multi-scale training"],
        ["YOLOv3", "2018", "Darknet-53, FPN, dự đoán đa tỉ lệ"],
        ["YOLOv4", "2020", "CSPDarknet53, Mosaic augmentation, BoF/BoS"],
        ["YOLOv5", "2020", "PyTorch reimplementation, Auto-anchor, dễ triển khai"],
        ["YOLOv7", "2022", "E-ELAN, compound scaling, auxiliary head"],
        ["YOLOv8", "2023", "C2f module, Decoupled Head, Anchor-free, Task-Aligned Assigner"],
    ]
    story.append(make_table(yolo_data[0], yolo_data[1:], [60, 40, 400], "#1e3a8a"))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.4. Lý do chọn YOLOv8s", h2))
    for reason in [
        "<b>Cân bằng tốc độ - độ chính xác:</b> 11.12M tham số, nằm giữa Nano và Large.",
        "<b>Kiến trúc hiện đại:</b> Anchor-free, Decoupled Head, C2f module.",
        "<b>Hệ sinh thái hoàn chỉnh:</b> Ultralytics API hỗ trợ train/val/predict/export.",
        "<b>Triển khai đa nền tảng:</b> Xuất ONNX, TensorRT, CoreML, TFLite.",
        "<b>Phù hợp phần cứng:</b> Hiệu quả trên GPU phổ thông RTX 3050 (4GB VRAM)."
    ]:
        story.append(Paragraph(f"• {reason}", bullet))

    # ========== CHƯƠNG 4: CƠ SỞ LÝ THUYẾT ==========
    story.append(PageBreak())
    story.append(Paragraph("CHƯƠNG 4: CƠ SỞ LÝ THUYẾT", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))

    story.append(Paragraph("4.1. Kiến trúc YOLOv8", h2))
    story.append(Paragraph(
        "YOLOv8 gồm 3 thành phần chính: <b>Backbone</b> (CSPDarknet + C2f + SPPF), "
        "<b>Neck</b> (FPN + PAN), và <b>Head</b> (Decoupled Detection Head, Anchor-Free).", body))

    # Architecture image
    arch_img = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        r"..\.gemini\antigravity-ide\brain\f8d47223-e66d-4dc2-aa99-c9946a977f81\yolov8_architecture_1791458913245.jpg")
    # Fallback paths
    if not os.path.exists(arch_img):
        arch_img = r"C:\Users\Administrator\.gemini\antigravity-ide\brain\f8d47223-e66d-4dc2-aa99-c9946a977f81\yolov8_architecture_1791458913245.jpg"
    if os.path.exists(arch_img):
        story.append(Image(arch_img, width=500, height=280))
        story.append(Paragraph("<i>Hình 1: Kiến trúc tổng quan mô hình YOLOv8s — Backbone, Neck, Detection Head</i>", caption))

    story.append(Paragraph("4.1.1. Backbone — CSPDarknet với C2f Module", h3))
    story.append(Paragraph(
        "Backbone trích xuất đặc trưng từ ảnh đầu vào. Module <b>C2f</b> (cải tiến từ C3 trong YOLOv5) "
        "kết hợp ý tưởng ELAN: chia đặc trưng thành hai nhánh, nhánh thứ hai đi qua n bottleneck blocks, "
        "concatenate tất cả đầu ra trung gian, rồi trộn bằng convolution 1×1.", body))
    story.append(Paragraph(
        "<b>SPPF (Spatial Pyramid Pooling — Fast):</b> Nằm ở cuối backbone, thực hiện pooling đa tỉ lệ "
        "bằng 3 lớp MaxPool 5×5 nối tiếp để tổng hợp ngữ cảnh toàn cục.", body))

    story.append(Paragraph("4.1.2. Neck — FPN + PAN", h3))
    story.append(Paragraph(
        "<b>FPN:</b> Truyền thông tin ngữ nghĩa top-down. <b>PAN:</b> Truyền thông tin localization bottom-up. "
        "Kết quả: 3 bản đồ đặc trưng: P3 (80×80) — đối tượng nhỏ, P4 (40×40) — trung bình, P5 (20×20) — lớn.", body))

    story.append(Paragraph("4.1.3. Head — Decoupled Head (Anchor-Free)", h3))
    story.append(Paragraph(
        "YOLOv8 tách riêng nhánh Classification và Regression (Decoupled Head). "
        "Loại bỏ hoàn toàn anchor boxes, dự đoán trực tiếp bounding box tại mỗi vị trí grid. "
        "Sử dụng <b>Distribution Focal Loss (DFL)</b> để mô hình hóa từng cạnh hộp như phân phối xác suất rời rạc.", body))

    story.append(Paragraph("4.2. Hàm mất mát (Loss Functions)", h2))
    story.append(Paragraph("<b>L_total = λ_box · L_box + λ_cls · L_cls + λ_dfl · L_dfl</b>", body_bold))
    story.append(Paragraph("• <b>Box Loss (CIoU):</b> L_CIoU = 1 - IoU + ρ²(b,b_gt)/c² + αv — Xét khoảng cách tâm, tỉ lệ khung hình.", bullet))
    story.append(Paragraph("• <b>Classification Loss:</b> Binary Cross-Entropy với Sigmoid cho phân loại đa nhãn.", bullet))
    story.append(Paragraph("• <b>DFL:</b> Distribution Focal Loss cho hồi quy bounding box chính xác hơn.", bullet))

    story.append(Paragraph("4.3. Các chỉ số đánh giá", h2))
    metrics_data = [
        ["Chỉ số", "Công thức", "Ý nghĩa"],
        ["Precision", "TP / (TP + FP)", "Tỉ lệ dự đoán đúng — ít báo động giả"],
        ["Recall", "TP / (TP + FN)", "Tỉ lệ phát hiện đúng — ít bỏ sót"],
        ["F1-Score", "2×P×R / (P+R)", "Trung bình điều hòa Precision & Recall"],
        ["mAP@0.5", "Mean AP tại IoU=0.5", "Chỉ số tổng hợp chính"],
        ["mAP@0.5:0.95", "Mean AP, IoU 0.5→0.95", "Chỉ số nghiêm ngặt hơn"],
        ["IoU", "|A∩B| / |A∪B|", "Mức trùng khớp predicted vs ground truth"],
    ]
    story.append(make_table(metrics_data[0], metrics_data[1:], [80, 160, 260], "#475569"))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4.4. Kỹ thuật tăng cường dữ liệu (Data Augmentation)", h2))
    aug_data = [
        ["Kỹ thuật", "Mô tả", "Tham số"],
        ["Mosaic", "Ghép 4 ảnh thành 1", "mosaic=1.0"],
        ["HSV Augmentation", "Biến đổi Hue/Saturation/Value", "h=0.015, s=0.7, v=0.4"],
        ["Horizontal Flip", "Lật ngang ngẫu nhiên", "fliplr=0.5"],
        ["Scale / Translate", "Co giãn và dịch chuyển", "scale=0.5, translate=0.1"],
        ["Erasing", "Xóa ngẫu nhiên vùng ảnh", "erasing=0.4"],
        ["RandAugment", "Auto-augmentation policy", "auto_augment=randaugment"],
    ]
    story.append(make_table(aug_data[0], aug_data[1:], [100, 220, 180], "#475569"))

    # ========== CHƯƠNG 5: PHƯƠNG PHÁP ==========
    story.append(PageBreak())
    story.append(Paragraph("CHƯƠNG 5: PHƯƠNG PHÁP", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))

    story.append(Paragraph("5.1. Kiến trúc hệ thống tổng quan", h2))
    sys_img = r"C:\Users\Administrator\.gemini\antigravity-ide\brain\f8d47223-e66d-4dc2-aa99-c9946a977f81\fire_detection_system_1791458926625.jpg"
    if os.path.exists(sys_img):
        story.append(Image(sys_img, width=500, height=280))
        story.append(Paragraph("<i>Hình 2: Kiến trúc hệ thống phát hiện đám cháy và khói — Pipeline từ đầu vào đến cảnh báo</i>", caption))

    story.append(Paragraph(
        "Hệ thống gồm 5 giai đoạn: (1) Thu thập từ camera CCTV/drone, (2) Tiền xử lý resize 640×640, "
        "(3) Suy luận YOLOv8s trên GPU, (4) Hậu xử lý NMS, (5) Hiển thị & cảnh báo.", body))

    story.append(Paragraph("5.2. Tập dữ liệu", h2))
    dataset_data = [
        ["Tập dữ liệu", "Số lượng ảnh", "Mục đích"],
        ["Training (Huấn luyện)", "~7,000 ảnh", "Huấn luyện mô hình"],
        ["Validation (Xác thực)", "2,000 ảnh", "Theo dõi hiệu năng & điều chỉnh"],
        ["Test (Kiểm thử)", "1,000 ảnh", "Đánh giá cuối cùng, độc lập"],
    ]
    story.append(make_table(dataset_data[0], dataset_data[1:], [140, 120, 240], "#0284c7"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Dữ liệu đa dạng: cháy trong nhà, cháy rừng, cháy phương tiện, khói trắng/đen, ban ngày/đêm, "
                           "và các trường hợp gây nhiễu (hoàng hôn, đèn LED, sương mù).", body))

    # Label distribution
    story.append(Paragraph("5.2.1. Phân bố nhãn trong tập dữ liệu", h3))
    labels_img = img_path("labels.jpg")
    if os.path.exists(labels_img):
        story.append(Image(labels_img, width=480, height=300))
        story.append(Paragraph("<i>Hình 3: Phân bố nhãn — mất cân bằng giữa Fire và Smoke, phân bố vị trí/kích thước bounding box</i>", caption))

    story.append(Paragraph("5.3. Cấu hình huấn luyện", h2))
    config_data2 = [
        ["Siêu tham số", "Giá trị", "Siêu tham số", "Giá trị"],
        ["Mô hình cơ sở", "yolov8s.pt (COCO)", "Epochs", "50"],
        ["Batch size", "16", "Image size", "640 × 640"],
        ["Optimizer", "Auto (SGD)", "Learning rate", "lr0=0.01, lrf=0.01"],
        ["Momentum", "0.937", "Weight decay", "0.0005"],
        ["Warmup epochs", "3.0", "AMP", "True (Mixed Precision)"],
        ["IoU threshold", "0.7", "Close mosaic", "10 (cuối)"],
    ]
    story.append(make_table(config_data2[0], config_data2[1:], [100, 150, 100, 150], "#1e3a8a"))

    # ========== CHƯƠNG 6: THỰC NGHIỆM VÀ KẾT QUẢ ==========
    story.append(PageBreak())
    story.append(Paragraph("CHƯƠNG 6: THỰC NGHIỆM VÀ KẾT QUẢ", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))

    story.append(Paragraph("6.1. Thông tin mô hình", h2))
    model_data = [
        ["Hạng mục", "Chi tiết", "Hạng mục", "Chi tiết"],
        ["Kiến trúc", "YOLOv8s (72 layers)", "Tổng tham số", "11.12 triệu"],
        ["Độ phức tạp", "28.4 GFLOPs", "Thời gian train", "~3.65 giờ (13,154s)"],
    ]
    story.append(make_table(model_data[0], model_data[1:], [90, 160, 90, 160], "#475569"))
    story.append(Spacer(1, 8))

    # Training batch
    story.append(Paragraph("6.1.1. Minh họa batch huấn luyện", h3))
    train_img = img_path("train_batch0.jpg")
    if os.path.exists(train_img):
        story.append(Image(train_img, width=480, height=290))
        story.append(Paragraph("<i>Hình 4: Batch huấn luyện với Mosaic augmentation và ground truth bounding box</i>", caption))

    # Training curves
    story.append(Paragraph("6.1.2. Biểu đồ quá trình huấn luyện", h3))
    results_img = img_path("results.png")
    if os.path.exists(results_img):
        story.append(Image(results_img, width=500, height=300))
        story.append(Paragraph("<i>Hình 5: Đường cong huấn luyện qua 50 epochs — Box/Cls/DFL Loss, Precision, Recall, mAP</i>", caption))

    story.append(Paragraph(
        "<b>Phân tích:</b> Tất cả hàm loss giảm đều đặn, Precision/Recall tăng và ổn định từ epoch 20. "
        "Khoảng cách train-val loss nhỏ → <b>không có dấu hiệu overfitting</b>.", body))

    # ======= KẾT QUẢ ĐỊNH LƯỢNG =======
    story.append(PageBreak())
    story.append(Paragraph("6.2. Kết quả đánh giá định lượng", h2))

    story.append(Paragraph("A. Kết quả trên tập Kiểm thử độc lập (Test Set — 1,000 ảnh)", h3))
    test_results = [
        ["Phân loại", "Số nhãn", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
        ["Toàn bộ (All)", "2,177", "78.8%", "73.9%", "77.4%", "47.0%"],
        ["Lửa (Fire)", "1,839", "86.1%", "81.1%", "87.1%", "56.1%"],
        ["Khói (Smoke)", "338", "71.5%", "66.7%", "67.7%", "37.8%"],
    ]
    story.append(make_table(test_results[0], test_results[1:], [80, 60, 80, 80, 80, 80], "#0284c7"))
    story.append(Paragraph("• <b>mAP@0.75 trên tập Test:</b> 50.6%", bullet))
    story.append(Spacer(1, 8))

    story.append(Paragraph("B. Kết quả trên tập Xác thực (Validation Set — 2,000 ảnh)", h3))
    val_results = [
        ["Phân loại", "Số nhãn", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
        ["Toàn bộ (All)", "4,312", "80.1%", "69.5%", "75.1%", "45.1%"],
        ["Lửa (Fire)", "3,627", "87.5%", "79.2%", "85.5%", "55.0%"],
        ["Khói (Smoke)", "685", "72.7%", "59.9%", "64.8%", "35.3%"],
    ]
    story.append(make_table(val_results[0], val_results[1:], [80, 60, 80, 80, 80, 80], "#475569"))
    story.append(Paragraph("• <b>mAP@0.75 trên tập Val:</b> 48.0%", bullet))
    story.append(Spacer(1, 8))

    story.append(Paragraph("C. Tốc độ suy luận (Inference Speed — GPU RTX 3050)", h3))
    speed_data = [
        ["Giai đoạn", "Thời gian", "Tốc độ"],
        ["Tiền xử lý (Preprocess)", "0.3 ms/ảnh", "—"],
        ["Suy luận (Inference)", "10.7 ms/ảnh", "~93.5 FPS"],
        ["Hậu xử lý NMS", "1.0 ms/ảnh", "—"],
        ["Tổng (End-to-End)", "~12.0 ms/ảnh", "~83 FPS"],
    ]
    story.append(make_table(speed_data[0], speed_data[1:], [180, 130, 130], "#059669"))
    story.append(Paragraph(
        "→ Tốc độ <b>83 FPS</b> vượt xa yêu cầu camera giám sát (25-30 FPS), "
        "đáp ứng giám sát đa luồng (multi-stream).", body_bold))

    # ======= BIỂU ĐỒ ĐÁNH GIÁ =======
    story.append(PageBreak())
    story.append(Paragraph("6.3. Đường cong đánh giá hiệu năng", h2))

    # PR + F1
    pr_img = img_path("PR_curve.png")
    f1_img = img_path("F1_curve.png")
    if os.path.exists(pr_img) and os.path.exists(f1_img):
        img_row = [
            [Image(pr_img, width=240, height=155), Image(f1_img, width=240, height=155)],
            [Paragraph("<i>Hình 6: Đường cong Precision-Recall</i>", caption),
             Paragraph("<i>Hình 7: Đường cong F1-Score theo Confidence</i>", caption)]
        ]
        t = Table(img_row, colWidths=[250, 250])
        t.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(t)
        story.append(Spacer(1, 6))

    # P + R curves
    p_img = img_path("P_curve.png")
    r_img = img_path("R_curve.png")
    if os.path.exists(p_img) and os.path.exists(r_img):
        img_row2 = [
            [Image(p_img, width=240, height=155), Image(r_img, width=240, height=155)],
            [Paragraph("<i>Hình 8: Đường cong Precision theo Confidence</i>", caption),
             Paragraph("<i>Hình 9: Đường cong Recall theo Confidence</i>", caption)]
        ]
        t2 = Table(img_row2, colWidths=[250, 250])
        t2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(t2)
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Phân tích:</b> Lớp Fire có đường PR cao và rộng (AP=87.1%). "
        "Đường F1 đạt đỉnh ở conf ≈ 0.3-0.4 — ngưỡng tối ưu cho cân bằng phát hiện sớm vs. báo động giả.", body))

    # ======= CONFUSION MATRIX =======
    story.append(PageBreak())
    story.append(Paragraph("6.4. Ma trận nhầm lẫn (Confusion Matrix)", h2))
    cm_img = img_path("confusion_matrix.png")
    cm_norm_img = img_path("confusion_matrix_normalized.png")
    if os.path.exists(cm_img) and os.path.exists(cm_norm_img):
        cm_row = [
            [Image(cm_img, width=240, height=190), Image(cm_norm_img, width=240, height=190)],
            [Paragraph("<i>Hình 10: Ma trận nhầm lẫn (raw)</i>", caption),
             Paragraph("<i>Hình 11: Ma trận nhầm lẫn chuẩn hóa</i>", caption)]
        ]
        t_cm = Table(cm_row, colWidths=[250, 250])
        t_cm.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(t_cm)
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Phân tích:</b> Fire có tỉ lệ phân loại đúng cao, ít nhầm lẫn. "
        "Smoke có tỉ lệ FN cao hơn (khói loãng → Background). Nhầm lẫn Fire↔Smoke tương đối thấp.", body))

    # ======= KẾT QUẢ TRỰC QUAN =======
    story.append(Paragraph("6.5. Kết quả trực quan — So sánh Ground Truth vs. Prediction", h2))
    val_label = img_path("val_batch0_labels.jpg")
    val_pred = img_path("val_batch0_pred.jpg")
    if os.path.exists(val_label) and os.path.exists(val_pred):
        vis_row = [
            [Image(val_label, width=240, height=175), Image(val_pred, width=240, height=175)],
            [Paragraph("<i>Hình 12: Ground Truth (nhãn thật)</i>", caption),
             Paragraph("<i>Hình 13: Prediction (YOLOv8s dự đoán)</i>", caption)]
        ]
        t_vis = Table(vis_row, colWidths=[250, 250])
        t_vis.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(t_vis)
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Mô hình phát hiện chính xác vị trí và kích thước ngọn lửa. Confidence score thường 0.5-0.9. "
        "Một số khói loãng/khói xa bị bỏ sót hoặc confidence thấp.", body))

    # ======= PHÂN TÍCH CHUYÊN SÂU =======
    story.append(PageBreak())
    story.append(Paragraph("6.6. Phân tích chuyên sâu", h2))

    story.append(Paragraph("Điểm mạnh nổi bật:", h3))
    for s in [
        "<b>Tổng quát hóa xuất sắc:</b> Test mAP@0.5 (77.4%) > Val mAP@0.5 (75.1%) → không overfitting.",
        "<b>Fire detection vượt trội:</b> P=86.1%, R=81.1%, mAP@0.5=87.1% — phát hiện ngọn lửa chính xác.",
        "<b>Tốc độ 83 FPS:</b> Gấp ~3 lần yêu cầu camera giám sát, cho phép giám sát 2-3 luồng camera đồng thời.",
        "<b>Hội tụ ổn định:</b> Loss giảm mượt mà, không dao động bất thường.",
    ]:
        story.append(Paragraph(f"• {s}", bullet))

    story.append(Paragraph("Hạn chế và thách thức:", h3))
    for w in [
        "<b>Class Imbalance:</b> Fire:Smoke = 5.4:1 → Smoke Recall chỉ 66.7%, bỏ sót ~33%.",
        "<b>Khói vô định hình:</b> Không biên dạng cố định, dễ hòa lẫn mây/sương mù.",
        "<b>mAP@0.5:0.95 hạn chế:</b> 47.0% tổng, 37.8% Smoke — localization chưa chặt ở IoU cao.",
    ]:
        story.append(Paragraph(f"• {w}", bullet))

    # ========== CHƯƠNG 7: KẾT LUẬN ==========
    story.append(PageBreak())
    story.append(Paragraph("CHƯƠNG 7: KẾT LUẬN", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))

    conclusion_data = [
        ["Mục tiêu", "Kết quả", "Đánh giá"],
        ["mAP@0.5 ≥ 75% trên Test", "77.4%", "✓ ĐẠT"],
        ["Tốc độ ≥ 30 FPS", "83 FPS", "✓ VƯỢT MỤC TIÊU"],
        ["Phát hiện Fire & Smoke", "Fire: 87.1%, Smoke: 67.7%", "✓ ĐẠT"],
        ["Không overfitting", "Test mAP > Val mAP", "✓ ĐẠT"],
    ]
    story.append(make_table(conclusion_data[0], conclusion_data[1:], [170, 190, 140], "#059669"))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Đóng góp chính:", h3))
    for c in [
        "Xây dựng thành công mô hình YOLOv8s fine-tuned cho fire detection.",
        "Chứng minh tính khả thi sử dụng one-stage detector trên GPU phổ thông.",
        "Phân tích toàn diện điểm mạnh và hạn chế, đặc biệt thách thức phát hiện khói.",
        "Đề xuất cấu hình triển khai: conf=0.25-0.35 (cảnh báo sớm), conf=0.50-0.60 (dập lửa tự động).",
    ]:
        story.append(Paragraph(f"• {c}", bullet))

    # ========== CHƯƠNG 8: HƯỚNG PHÁT TRIỂN ==========
    story.append(Paragraph("CHƯƠNG 8: HƯỚNG PHÁT TRIỂN", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))

    story.append(Paragraph("8.1. Cải thiện hiệu năng", h2))
    for item in [
        "Bổ sung 1,500-2,000 ảnh khói đa dạng để cân bằng dữ liệu.",
        "Tăng lên 70-80 epochs + Cosine Annealing LR → cải thiện 2-4% mAP.",
        "Thử nghiệm YOLOv8m/l, so sánh RT-DETR, YOLOv9, YOLOv10.",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Paragraph("8.2. Mở rộng chức năng", h2))
    for item in [
        "Video real-time: Camera IP/RTSP + tracking (DeepSORT, BoT-SORT).",
        "Semantic Segmentation: YOLOv8-seg phân đoạn chính xác vùng lửa/khói.",
        "Ước lượng mức độ nghiêm trọng: nhỏ / trung bình / lớn.",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Paragraph("8.3. Edge Computing & IoT", h2))
    for item in [
        "Xuất ONNX / TensorRT (Jetson) / TFLite (Android).",
        "Quantization INT8/FP16, Pruning, Knowledge Distillation.",
        "Push notification (Firebase), Sprinkler/Alarm relay, Dashboard web.",
        "Fusion camera hồng ngoại (thermal) + RGB + cảm biến nhiệt/khí.",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    # ========== CHƯƠNG 9: TÀI LIỆU THAM KHẢO ==========
    story.append(PageBreak())
    story.append(Paragraph("CHƯƠNG 9: TÀI LIỆU THAM KHẢO", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=8))

    refs = [
        "[1] Redmon, J. et al. (2016). \"You Only Look Once: Unified, Real-Time Object Detection.\" CVPR. arXiv: 1506.02640",
        "[2] Redmon, J. & Farhadi, A. (2017). \"YOLO9000: Better, Faster, Stronger.\" CVPR. arXiv: 1612.08242",
        "[3] Redmon, J. & Farhadi, A. (2018). \"YOLOv3: An Incremental Improvement.\" arXiv: 1804.02767",
        "[4] Bochkovskiy, A. et al. (2020). \"YOLOv4: Optimal Speed and Accuracy of Object Detection.\" arXiv: 2004.10934",
        "[5] Wang, C.-Y. et al. (2023). \"YOLOv7: Trainable bag-of-freebies.\" CVPR. arXiv: 2207.02696",
        "[6] Jocher, G. et al. (2023). \"Ultralytics YOLOv8.\" GitHub: github.com/ultralytics/ultralytics",
        "[7] Lin, T.-Y. et al. (2017). \"Feature Pyramid Networks for Object Detection.\" CVPR. arXiv: 1612.03144",
        "[8] Liu, S. et al. (2018). \"Path Aggregation Network for Instance Segmentation.\" CVPR. arXiv: 1803.01534",
        "[9] Zheng, Z. et al. (2020). \"Distance-IoU Loss: Faster Learning for Bounding Box Regression.\" AAAI. arXiv: 1911.08287",
        "[10] Lin, T.-Y. et al. (2017). \"Focal Loss for Dense Object Detection.\" ICCV. arXiv: 1708.02002",
        "[11] Li, P. & Zhao, W. (2020). \"Image Fire Detection Algorithms Based on CNN.\" Case Studies in Thermal Eng.",
        "[12] Saponara, S. et al. (2021). \"Real-time video fire/smoke detection based on CNN.\" J. Real-Time Image Processing.",
        "[13] Park, M. & Ko, B.C. (2020). \"Two-Step Real-Time Night-Time Fire Detection.\" Sensors, 20(8), 2202.",
        "[14] Abdusalomov, A.B. et al. (2023). \"Improved Forest Fire Detection Based on Detectron2.\" Sensors, 23(3), 1512.",
        "[15] Wang, D. et al. (2022). \"Real-time Smoke and Fire Detection Based on Improved YOLOv5.\" IEEE ICET.",
        "[16] Ultralytics YOLOv8 Documentation. https://docs.ultralytics.com/",
        "[17] PyTorch Documentation. https://pytorch.org/docs/stable/",
        "[18] Roboflow Universe — Fire and Smoke Datasets. https://universe.roboflow.com/",
        "[19] NVIDIA CUDA Toolkit Documentation. https://docs.nvidia.com/cuda/",
        "[20] Goodfellow, I. et al. (2016). Deep Learning. MIT Press.",
        "[21] Géron, A. (2022). Hands-On Machine Learning (3rd Ed). O'Reilly Media.",
    ]
    ref_style = ParagraphStyle('Ref', parent=body, fontSize=8.5, leading=13, spaceAfter=3, leftIndent=5)
    for ref in refs:
        story.append(Paragraph(ref, ref_style))

    # ============================================================
    # BUILD PDF
    # ============================================================
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"\n[OK] PDF da tao thanh cong: {os.path.abspath(output_file)}")
    print(f"     Kich thuoc: {os.path.getsize(output_file) / 1024:.1f} KB")


if __name__ == "__main__":
    create_full_report()

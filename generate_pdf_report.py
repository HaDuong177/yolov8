import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 1. Đăng ký font tiếng Việt từ Windows
pdfmetrics.registerFont(TTFont('Arial', r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', r'C:\Windows\Fonts\ariali.ttf'))

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
        self.setFont("Arial", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (trang > 1)
        if self._pageNumber > 1:
            self.drawString(54, 842 - 36, "Báo Cáo Đánh Giá Kỹ Thuật — YOLOv8s Fire & Smoke Detection")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 842 - 42, 595 - 54, 842 - 42)
            
        # Footer
        footer_text = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(595 - 54, 30, footer_text)
        self.drawString(54, 30, "Hệ Thống Nhận Diện Đám Cháy & Khói Tự Động")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 42, 595 - 54, 42)
        
        self.restoreState()

def create_report(output_filename="Bao_Cao_Danh_Gia_YOLOv8_Fire_Detection.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#1e293b"),
        alignment=1, # Center
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#64748b"),
        alignment=1, # Center
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Arial-Bold'
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        alignment=1
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Arial-Bold'
    )

    story = []

    # Tiêu đề
    story.append(Paragraph("BÁO CÁO ĐÁNH GIÁ CHI TIẾT HIỆU NĂNG MÔ HÌNH", title_style))
    story.append(Paragraph("Phát Hiện Đám Cháy & Khói Bằng Kiến Trúc YOLOv8s (Fire & Smoke Detection)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=12))

    # 1. Cấu hình thực nghiệm
    story.append(Paragraph("1. Cấu Hình Huấn Luyện & Phần Cứng", h1_style))
    config_data = [
        [Paragraph("Thông số", table_header_style), Paragraph("Giá trị thiết lập", table_header_style), Paragraph("Thông số", table_header_style), Paragraph("Giá trị thiết lập", table_header_style)],
        [Paragraph("Mô hình", table_cell_bold), Paragraph("YOLOv8s (Small)", table_cell_style), Paragraph("Kích thước ảnh", table_cell_bold), Paragraph("640 × 640", table_cell_style)],
        [Paragraph("Số Epochs", table_cell_bold), Paragraph("50", table_cell_style), Paragraph("Batch Size", table_cell_bold), Paragraph("16", table_cell_style)],
        [Paragraph("Phần cứng (GPU)", table_cell_bold), Paragraph("NVIDIA RTX 3050 (4GB)", table_cell_style), Paragraph("Thời gian train", table_cell_bold), Paragraph("~3.65 giờ (13,154s)", table_cell_style)],
        [Paragraph("Tổng tham số", table_cell_bold), Paragraph("11.12 triệu (28.4 GFLOPs)", table_cell_style), Paragraph("Tập dữ liệu", table_cell_bold), Paragraph("2 lớp: Fire & Smoke", table_cell_style)],
    ]
    t_config = Table(config_data, colWidths=[110, 137, 110, 137])
    t_config.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_config)
    story.append(Spacer(1, 10))

    # 2. Kết quả định lượng
    story.append(Paragraph("2. Kết Quả Định Lượng Chi Tiết", h1_style))
    story.append(Paragraph("A. Kết quả kiểm thử trên Tập Độc Lập (Test Set — 1,000 ảnh):", h2_style))
    
    test_data = [
        [Paragraph("Phân lớp", table_header_style), Paragraph("Số nhãn (Instances)", table_header_style), Paragraph("Precision (P)", table_header_style), Paragraph("Recall (R)", table_header_style), Paragraph("mAP@0.5", table_header_style), Paragraph("mAP@0.5:0.95", table_header_style)],
        [Paragraph("Toàn bộ (All)", table_cell_bold), Paragraph("2,177", table_cell_style), Paragraph("78.8%", table_cell_bold), Paragraph("73.9%", table_cell_bold), Paragraph("77.4%", table_cell_bold), Paragraph("47.0%", table_cell_bold)],
        [Paragraph("Lửa (Fire)", table_cell_style), Paragraph("1,839", table_cell_style), Paragraph("86.1%", table_cell_style), Paragraph("81.1%", table_cell_style), Paragraph("87.1%", table_cell_style), Paragraph("56.1%", table_cell_style)],
        [Paragraph("Khói (Smoke)", table_cell_style), Paragraph("338", table_cell_style), Paragraph("71.5%", table_cell_style), Paragraph("66.7%", table_cell_style), Paragraph("67.7%", table_cell_style), Paragraph("37.8%", table_cell_style)],
    ]
    t_test = Table(test_data, colWidths=[100, 95, 75, 75, 75, 74])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284c7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f0f9ff"), colors.white, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_test)
    story.append(Spacer(1, 8))

    story.append(Paragraph("B. Kết quả trên Tập Xác Thực (Validation Set — 2,000 ảnh):", h2_style))
    val_data = [
        [Paragraph("Phân lớp", table_header_style), Paragraph("Số nhãn (Instances)", table_header_style), Paragraph("Precision (P)", table_header_style), Paragraph("Recall (R)", table_header_style), Paragraph("mAP@0.5", table_header_style), Paragraph("mAP@0.5:0.95", table_header_style)],
        [Paragraph("Toàn bộ (All)", table_cell_bold), Paragraph("4,312", table_cell_style), Paragraph("80.1%", table_cell_bold), Paragraph("69.5%", table_cell_bold), Paragraph("75.1%", table_cell_bold), Paragraph("45.1%", table_cell_bold)],
        [Paragraph("Lửa (Fire)", table_cell_style), Paragraph("3,627", table_cell_style), Paragraph("87.5%", table_cell_style), Paragraph("79.2%", table_cell_style), Paragraph("85.5%", table_cell_style), Paragraph("55.0%", table_cell_style)],
        [Paragraph("Khói (Smoke)", table_cell_style), Paragraph("685", table_cell_style), Paragraph("72.7%", table_cell_style), Paragraph("59.9%", table_cell_style), Paragraph("64.8%", table_cell_style), Paragraph("35.3%", table_cell_style)],
    ]
    t_val = Table(val_data, colWidths=[100, 95, 75, 75, 75, 74])
    t_val.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#475569")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_val)
    story.append(Spacer(1, 8))

    story.append(Paragraph("C. Tốc độ thực thi suy luận (Inference Speed trên GPU RTX 3050):", h2_style))
    story.append(Paragraph("• <b>Tiền xử lý (Preprocess):</b> 0.3 ms | <b>Suy luận (Inference):</b> 10.7 ms (~93.5 FPS) | <b>Hậu xử lý (NMS):</b> 1.0 ms", bullet_style))
    story.append(Paragraph("• <b>Tổng độ trễ: ~12.0 ms/ảnh (~83 FPS)</b>. Hoàn toàn đáp ứng tiêu chuẩn xử lý thời gian thực (Real-time) trên các hệ thống giám sát camera (25–60 FPS).", bullet_style))
    story.append(Spacer(1, 10))

    # Page Break sang trang 2: Phân tích & Hình ảnh trực quan
    story.append(PageBreak())

    story.append(Paragraph("3. Phân Tích & Đánh Giá Chuyên Sâu", h1_style))
    
    story.append(Paragraph("1. Khả năng tổng quát hóa tuyệt vời (Không Overfitting):", h2_style))
    story.append(Paragraph("Mô hình đạt kết quả trên tập Test (mAP@0.5 = 77.4%, mAP50-95 = 47.0%) thậm chí nhỉnh hơn tập Validation (mAP@0.5 = 75.1%, mAP50-95 = 45.1%). Điều này chứng minh mô hình học các đặc trưng khái quát hóa thực tế của đám cháy rất tốt, không bị hiện tượng học vẹt (overfitting).", body_style))

    story.append(Paragraph("2. Hiệu năng nhận diện Đám Cháy (Fire) vượt trội:", h2_style))
    story.append(Paragraph("Lớp Lửa đạt độ chính xác Precision 86.1%, Recall 81.1% và mAP@0.5 lên tới 87.1% trên tập Test. Các đám cháy có màu sắc, độ sáng tương phản cao được nhận diện tức thì với độ tin cậy rất cao, giảm thiểu tối đa tình trạng bỏ sót nguy cơ.", body_style))

    story.append(Paragraph("3. Hiện tượng mất cân bằng dữ liệu & Thách thức nhận diện Khói (Smoke):", h2_style))
    story.append(Paragraph("Số lượng đối tượng Lửa trong tập dữ liệu nhiều gấp <b>hơn 5.4 lần</b> so với Khói (1,839 Fire vs 338 Smoke trên tập Test). Điều này trực tiếp khiến mAP@0.5 của Khói dừng ở mức 67.7% và Recall là 66.7%. Ngoài ra, khói là vật thể vô định hình (amorphous boundary), dễ bị loãng và nhầm lẫn với mây, sương mù hoặc phông nền sáng.", body_style))

    story.append(Paragraph("4. Độ hội tụ và ổn định huấn luyện:", h2_style))
    story.append(Paragraph("Hàm mất mát train/cls_loss giảm sâu từ 2.30 xuống 0.62, val/cls_loss giảm từ 2.56 xuống 0.77. Quá trình học qua 50 epochs duy trì đà giảm đều đặn, không có phân kỳ hay dao động bất thường.", body_style))
    story.append(Spacer(1, 10))

    # Biểu đồ kết quả
    story.append(Paragraph("4. Đồ Thị Huấn Luyện & Ma Trận Nhầm Lẫn", h1_style))
    
    res_img_path = r"runs_fire_detection\train_yolov8s\results.png"
    cm_img_path = r"runs_fire_detection\train_yolov8s\confusion_matrix.png"
    
    if os.path.exists(res_img_path) and os.path.exists(cm_img_path):
        img_table_data = [
            [
                Image(res_img_path, width=240, height=140),
                Image(cm_img_path, width=240, height=140)
            ],
            [
                Paragraph("<b>Hình 1:</b> Quá trình hội tụ Loss và Metrics qua 50 Epochs", table_cell_style),
                Paragraph("<b>Hình 2:</b> Ma trận nhầm lẫn (Confusion Matrix)", table_cell_style)
            ]
        ]
        t_imgs = Table(img_table_data, colWidths=[247, 247])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t_imgs)
        story.append(Spacer(1, 10))

    # Page Break sang trang 3: Kiến nghị thực tế & Kết luận
    story.append(PageBreak())

    story.append(Paragraph("5. Đồ Thị PR Curve & Phân Tích Đường Cong Hiệu Năng", h1_style))
    pr_img_path = r"runs_fire_detection\train_yolov8s\PR_curve.png"
    f1_img_path = r"runs_fire_detection\train_yolov8s\F1_curve.png"
    if os.path.exists(pr_img_path) and os.path.exists(f1_img_path):
        img2_table_data = [
            [
                Image(pr_img_path, width=240, height=145),
                Image(f1_img_path, width=240, height=145)
            ],
            [
                Paragraph("<b>Hình 3:</b> Precision-Recall Curve (PR Curve)", table_cell_style),
                Paragraph("<b>Hình 4:</b> F1-Confidence Curve", table_cell_style)
            ]
        ]
        t_imgs2 = Table(img2_table_data, colWidths=[247, 247])
        t_imgs2.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t_imgs2)
        story.append(Spacer(1, 12))

    story.append(Paragraph("6. Kiến Nghị Tối Ưu Hóa & Triển Khai Thực Tế", h1_style))
    story.append(Paragraph("1. <b>Cấu hình ngưỡng tin cậy (Confidence Threshold) theo bài toán:</b>", bullet_style))
    story.append(Paragraph("• <i>Hệ thống cảnh báo sớm (Early Detection):</i> Đặt conf = 0.25 - 0.35 nhằm tăng tối đa Recall, ưu tiên phát hiện sớm khói và tàn lửa dù ở cự ly xa.", bullet_style))
    story.append(Paragraph("• <i>Hệ thống tự động kích hoạt vòi phun / chuông báo:</i> Đặt conf = 0.50 - 0.60 nhằm đảm bảo chỉ hành động khi độ tin cậy chắc chắn, triệt tiêu báo động giả.", bullet_style))
    
    story.append(Paragraph("2. <b>Cân bằng dữ liệu khói (Data Augmentation):</b>", bullet_style))
    story.append(Paragraph("Bổ sung thêm khoảng 1,500 - 2,000 ảnh đặc thù về khói (khói công nghiệp, khói rơm rạ, khói cháy nhà ban ngày và ban đêm) để cân bằng tỉ lệ 1:1 với lớp lửa, từ đó nâng mAP của khói lên ngang bằng lớp lửa (~85%+).", bullet_style))

    story.append(Paragraph("3. <b>Tăng số chu kỳ huấn luyện (Epochs):</b>", bullet_style))
    story.append(Paragraph("Hàm mất mát ở epoch 50 vẫn đang có độ dốc giảm nhẹ. Có thể tăng lên 70 - 80 epochs kết hợp Cosine Learning Rate Schedule để đạt cực trị tối ưu.", bullet_style))

    story.append(Paragraph("4. <b>Chuẩn hóa định dạng nhãn (Label Sanitization):</b>", bullet_style))
    story.append(Paragraph("Khắc phục cảnh báo nhãn hỗn hợp Detection - Segmentation để dữ liệu thuần dạng Bounding Box, tránh phân mảnh bộ nhớ khi nạp batch.", bullet_style))

    story.append(Spacer(1, 15))
    story.append(Paragraph("7. Kết Luận Chung", h1_style))
    story.append(Paragraph("Mô hình <b>YOLOv8s</b> sau 50 epochs đạt chất lượng rất tốt với <b>mAP@0.5 đạt 77.4% trên tập Test</b> và <b>tốc độ xử lý ~83 FPS trên GPU RTX 3050</b>. Mô hình hoàn toàn đủ điều kiện kỹ thuật để đóng gói thành dịch vụ (API/Edge Inference) phục vụ bài toán giám sát an toàn cháy nổ trong các nhà xưởng, khu dân cư hoặc rừng.", body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF report generated successfully:", os.path.abspath(output_filename))

if __name__ == "__main__":
    create_report()

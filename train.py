import torch
from ultralytics import YOLO

def main():
    # Kiểm tra GPU
    print("=" * 50)
    print("KIỂM TRA CẤU HÌNH PHẦN CỨNG:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("CẢNH BÁO: Chưa nhận GPU! Đang chạy trên CPU.")
    print("=" * 50)

    # Khởi tạo mô hình YOLOv8
    # yolov8s.pt: Mô hình cỡ nhỏ (cân bằng tốt giữa tốc độ và độ chính xác)
    # yolov8n.pt: Mô hình siêu nhẹ
    model = YOLO("yolov8s.pt")

    # Bắt đầu huấn luyện
    # Đối với GPU RTX 3050 (4GB VRAM), batch=16 hoặc 8 là tối ưu
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,                   # Sử dụng GPU 0 (RTX 3050)
        workers=2,                  # Tối ưu cho Windows
        project="runs_fire_detection",
        name="train_yolov8s",
        save=True
    )

    # Đánh giá mô hình trên tập test
    print("\n--- ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---")
    metrics = model.val(data="data.yaml", split="test", device=0)
    print("Đánh giá hoàn tất!")

if __name__ == "__main__":
    main()

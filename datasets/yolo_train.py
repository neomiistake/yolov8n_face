from torch.xpu import device
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO('yolov8n.pt')  # 使用预训练的 YOLOv8n 模型

# 开始训练
model.train(
    data='badminton.yaml',  # 使用你的 YAML 文件
    epochs=50,              # 设置训练迭代次数
    batch=16,               # 设置批次大小
    imgsz=640,               # 设置图片尺寸
)
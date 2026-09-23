import os
from ultralytics import YOLO

print("エクスポート開始...")
model = YOLO("yolov8n.pt")

# 416固定サイズでINT8量子化エクスポート
output_path = model.export(
    format="openvino",
    int8=True,
    data="coco8.yaml",
    imgsz=416
)

print(f"エクスポート完了: {output_path}")
print("フォルダの存在確認:", os.path.exists(output_path))
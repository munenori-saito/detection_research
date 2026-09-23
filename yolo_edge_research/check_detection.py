import cv2
from ultralytics import YOLO

# 416 INT8モデルの読み込み
model = YOLO("yolov8n_int8_openvino_model")

cap = cv2.VideoCapture("benchmark_traffic.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 検証結果の保存設定
out = cv2.VideoWriter(
    "output_416_int8.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print("検証動画を出力中... (約100フレーム処理します)")
frame_count = 0

while cap.isOpened() and frame_count < 100:
    ret, frame = cap.read()
    if not ret:
        break

    # 416pxで推論
    results = model(frame, imgsz=416, verbose=False)
    
    # 検出ボックスを描画
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()
print("完了: output_416_int8.mp4 が作成されました。再生して奥の車が検出されているか確認してください。")
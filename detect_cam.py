import cv2
import time
from ultralytics import YOLO

# 1. モデルの読み込み
model = YOLO("best.pt")

# 2. カメラの起動
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("エラー: カメラを認識できませんでした。")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 推論
    results = model(frame, imgsz=416, conf=0.5, verbose=False)

    # 4. 描画
    annotated_frame = results[0].plot()

    # 5. FPS表示
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.imshow("Detection Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
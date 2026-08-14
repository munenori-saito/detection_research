import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0) #0番は内蔵カメラ

print("Qで終了")

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました。")
        break

    results = model(frame, stream=True)

    # 認識結果（バウンディングボックスとラベル）を描画
    for result in results:
        annotated_frame = result.plot()

    # 認識結果が描画された映像を画面に表示
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

#YOLOのバッチ取得したいわつ
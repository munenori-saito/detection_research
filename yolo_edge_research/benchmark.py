import cv2
import time
import numpy as np
from ultralytics import YOLO

def run_benchmark(video_path, model_path, imgsz=640, num_frames=300):
    """
    動画を入力として、モデルの推論FPSを計測する関数。
    """
    print(f"--- ベンチマーク開始 ---")
    print(f"モデル: {model_path} (入力サイズ: {imgsz})")
    print(f"テスト動画: {video_path}")
    
    # 1. モデルの読み込み
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return

    # 2. 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした。")
        return

    inference_times = []
    frame_count = 0

    print("推論中... (最初の数フレームはウォームアップとして除外します)")

    while cap.isOpened() and frame_count < num_frames + 10:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 3. 推論時間の計測（純粋なモデルの処理時間のみを測る）
        start_time = time.perf_counter()
        
        # 描画(plot)などは行わず、推論のみを実行
        results = model(frame, imgsz=imgsz, verbose=False)
        
        end_time = time.perf_counter()
        
        # 最初の10フレームは初期化のオーバーヘッドがあるため計測から除外（ウォームアップ）
        if frame_count >= 10:
            inference_times.append((end_time - start_time) * 1000) # ミリ秒に変換
            
        frame_count += 1

    cap.release()

    # 4. 結果の計算と出力
    if not inference_times:
        print("計測可能なフレームがありませんでした。")
        return

    avg_inference_ms = np.mean(inference_times)
    avg_fps = 1000.0 / avg_inference_ms

    print("\n--- ベンチマーク結果 ---")
    print(f"計測フレーム数: {len(inference_times)}")
    print(f"平均推論時間: {avg_inference_ms:.2f} ms / frame")
    print(f"平均FPS: {avg_fps:.2f} FPS")
    print("------------------------\n")

if __name__ == "__main__":
    # 自身の環境に合わせてパスを変更してください
    VIDEO_FILE = "benchmark_traffic.mp4"
    
    # まずは基準となる素のPyTorchモデルで計測（自動ダウンロードされます）
    MODEL_FILE = "yolov8n_int8_openvino_model" 
    
    run_benchmark(VIDEO_FILE, MODEL_FILE, imgsz=416)
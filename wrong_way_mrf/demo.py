"""
demo.py
-------
時空間MRF 逆走検知デモ - メインスクリプト

使い方:
    # 基本実行（動画ファイルから出力動画を生成）
    python demo.py --input ../video/benchmark_traffic.mp4 --output output.mp4

    # 正常走行方向を手動で指定（右向き=0°、下向き=90°）
    python demo.py --input ../video/benchmark_traffic.mp4 --normal-direction 0

    # リアルタイム表示のみ（ファイル保存なし）
    python demo.py --input ../video/benchmark_traffic.mp4 --no-save

    # 処理するフレーム数を制限（テスト用）
    python demo.py --input ../video/benchmark_traffic.mp4 --max-frames 200
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 同ディレクトリ内のモジュールを import
sys.path.insert(0, str(Path(__file__).parent))
from tracker import SimpleTracker
from wrong_way_detector import WrongWayDetector


# ---------------------------------------------------------------------------
# 可視化ユーティリティ
# ---------------------------------------------------------------------------

def draw_tracks(
    frame: np.ndarray,
    tracks,
    probs: dict,
    detector,
    wrongway_threshold: float = 0.6,
) -> np.ndarray:
    """
    追跡車両のバウンディングボックスと逆走確率を描画する。

    色分け:
      - 緑    : 正常走行
      - 黄    : 逆走確率 0.4〜0.6（要注意）
      - 赤    : 逆走確率 0.6 以上
      - 点滅赤: is_wrong_way() が True（連続判定済み）
    """
    overlay = frame.copy()

    for trk in tracks:
        vid = trk.track_id
        prob = probs.get(vid, 0.0)
        is_confirmed = detector.is_wrong_way(vid, min_consecutive=3)

        # 色の決定
        if is_confirmed:
            color = (0, 0, 255)   # 赤（確定逆走）
            thickness = 3
        elif prob >= wrongway_threshold:
            color = (0, 100, 255) # 橙（高確率）
            thickness = 2
        elif prob >= 0.4:
            color = (0, 220, 255) # 黄（疑い）
            thickness = 2
        else:
            color = (0, 200, 0)   # 緑（正常）
            thickness = 1

        x1, y1, x2, y2 = trk.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # ラベル（ID + 確率）
        label = f"ID:{vid} {prob:.2f}"
        label_y = max(y1 - 6, 15)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, label_y - th - 2), (x1 + tw + 2, label_y + 2), color, -1)
        cv2.putText(overlay, label, (x1 + 1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 速度ベクトル矢印
        vel = trk.smoothed_velocity()
        cx, cy = int(trk.center[0]), int(trk.center[1])
        arrow_scale = 3.0
        ex = int(cx + vel[0] * arrow_scale)
        ey = int(cy + vel[1] * arrow_scale)
        cv2.arrowedLine(overlay, (cx, cy), (ex, ey), color, 2, tipLength=0.3)

        # 確定逆走の場合は "WRONG WAY!" アラート
        if is_confirmed:
            alert_x = max(x1, 0)
            alert_y = max(y1 - 25, 20)
            cv2.putText(overlay, "WRONG WAY!", (alert_x, alert_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return overlay


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps_actual: float,
    normal_direction: float | None,
    n_total: int,
    n_wrongway: int,
    is_warmup: bool,
) -> np.ndarray:
    """HUD（ヘッドアップディスプレイ）情報をフレーム左上に描画する"""
    h, w = frame.shape[:2]

    # 半透明の背景パネル
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (320, 140), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.5, frame, 0.5, 0, frame)

    lines = [
        f"Frame: {frame_idx}",
        f"FPS: {fps_actual:.1f}",
        f"Normal dir: {normal_direction:.1f} deg" if normal_direction is not None else "Normal dir: estimating...",
        f"Vehicles: {n_total}  WrongWay: {n_wrongway}",
        "[WARMUP]" if is_warmup else "[RUNNING]",
    ]
    for i, line in enumerate(lines):
        color = (0, 180, 255) if i == 4 and is_warmup else (200, 255, 200)
        cv2.putText(frame, line, (15, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # 正常走行方向インジケータ（右上に羅針盤風）
    if normal_direction is not None:
        cx, cy, r = w - 60, 60, 40
        cv2.circle(frame, (cx, cy), r, (100, 100, 100), 1)
        angle_rad = np.radians(normal_direction)
        ex = int(cx + r * 0.8 * np.cos(angle_rad))
        ey = int(cy + r * 0.8 * np.sin(angle_rad))
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 100), 2, tipLength=0.3)
        cv2.putText(frame, "N", (cx - 5, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)

    return frame


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """凡例を右下に描画する"""
    h, w = frame.shape[:2]
    items = [
        ((0, 200, 0),   "Normal"),
        ((0, 220, 255), "Suspicious (>0.4)"),
        ((0, 100, 255), "High risk (>0.6)"),
        ((0, 0, 255),   "WRONG WAY (confirmed)"),
    ]
    for i, (color, label) in enumerate(items):
        y = h - 20 - i * 22
        cv2.rectangle(frame, (w - 200, y - 12), (w - 180, y + 2), color, -1)
        cv2.putText(frame, label, (w - 175, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def run_demo(args):
    # ---- モデルロード ----
    print("[INFO] YOLOv8 モデルをロード中...")
    try:
        from ultralytics import YOLO
        model_path = args.model
        model = YOLO(model_path)
        print(f"[INFO] モデルロード完了: {model_path}")
    except ImportError:
        print("[ERROR] ultralytics がインストールされていません。pip install ultralytics を実行してください。")
        sys.exit(1)

    # ---- 動画オープン ----
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"[ERROR] 動画ファイルを開けません: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 入力動画: {width}x{height} @ {orig_fps:.1f}fps, {total_frames}フレーム")

    # ---- 出力動画の準備 ----
    writer = None
    if args.output and not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, orig_fps, (width, height))
        print(f"[INFO] 出力先: {args.output}")

    # ---- トラッカー・検出器の初期化 ----
    tracker = SimpleTracker(
        iou_threshold=0.3,
        max_miss_frames=5,
        min_age_to_report=3,
        conf_threshold=args.conf,
    )

    detector = WrongWayDetector(
        window_size=8,
        temporal_weight=4.0,
        spatial_weight=2.0,
        spatial_distance_thresh=200.0,
        bp_iterations=10,
        wrongway_threshold=args.threshold,
        angle_sigma_deg=30.0,
        normal_direction_deg=args.normal_direction,
    )

    # ---- フレームループ ----
    frame_idx = 0
    max_frames = args.max_frames if args.max_frames > 0 else float("inf")
    fps_timer = time.time()
    fps_actual = orig_fps

    print("[INFO] 処理を開始します... (q キーで終了)")

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS 計測
        now = time.time()
        if frame_idx % 30 == 0 and frame_idx > 0:
            fps_actual = 30.0 / (now - fps_timer)
            fps_timer = now

        # ---- YOLOv8 推論 ----
        results = model(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
            verbose=False,
        )
        detections = SimpleTracker.parse_yolo_results(results)

        # ---- トラッキング更新 ----
        tracks = tracker.update(detections)

        # ---- MRF 逆走判定 ----
        probs = detector.update(frame_idx, tracks)

        # ---- 統計 ----
        n_wrongway = sum(
            1 for trk in tracks if detector.is_wrong_way(trk.track_id, min_consecutive=3)
        )

        # ---- 可視化 ----
        vis = frame.copy()
        vis = draw_tracks(vis, tracks, probs, detector, args.threshold)
        vis = draw_hud(
            vis,
            frame_idx=frame_idx,
            fps_actual=fps_actual,
            normal_direction=detector.normal_direction,
            n_total=len(tracks),
            n_wrongway=n_wrongway,
            is_warmup=not detector.direction_estimator.is_ready,
        )
        vis = draw_legend(vis)

        # ---- 出力 ----
        if writer:
            writer.write(vis)

        if not args.no_display:
            cv2.imshow("Wrong-Way Detection (MRF)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] ユーザーにより終了")
                break

        # ---- ログ ----
        if frame_idx % 60 == 0:
            n_dir = f"{detector.normal_direction:.1f}°" if detector.normal_direction else "推定中"
            print(f"  Frame {frame_idx:5d} | 車両数: {len(tracks):3d} | 逆走: {n_wrongway} | 正常方向: {n_dir}")

        frame_idx += 1

    # ---- 後処理 ----
    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] 出力動画を保存しました: {args.output}")
    cv2.destroyAllWindows()
    print(f"[INFO] 処理完了。合計 {frame_idx} フレームを処理しました。")


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="時空間MRF 逆走検知デモ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("../video/benchmark_traffic.mp4"),
        help="入力動画ファイルのパス",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output_wrongway.mp4"),
        help="出力動画ファイルのパス",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="../yolo_edge_research/yolov8n.pt",
        help="YOLOv8 モデルファイルのパス（.pt または OpenVINO モデルディレクトリ）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO 推論時の入力サイズ",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="YOLO 検出の信頼度閾値",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="逆走判定の確率閾値（MRF 出力）",
    )
    parser.add_argument(
        "--normal-direction",
        type=float,
        default=None,
        help="正常走行方向の角度（度, -180〜180）。None の場合は自動推定。例: 右向き=0, 下向き=90",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="処理するフレーム数の上限（0=無制限）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="動画ファイルへの保存をスキップする",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="リアルタイム表示をスキップする（headless 実行）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args)

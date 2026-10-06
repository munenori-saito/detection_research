"""
tracker.py
----------
YOLOv8 + Simple IoU トラッカー

外部の ByteTrack / SORT に依存せず、IoU ベースのシンプルな
多物体トラッキングを numpy だけで実装する。

主な機能:
  - YOLOv8 で車両クラス（car, truck, bus, motorcycle）を検出
  - フレーム間で IoU マッチングにより同一車両に一貫した ID を付与
  - 各車両の速度ベクトル（移動方向）を履歴から算出
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """1台の車両の追跡状態"""
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixels
    center: np.ndarray        # [cx, cy]
    velocity: np.ndarray      # [vx, vy] pixels/frame（移動量）
    age: int = 1              # 追跡継続フレーム数
    miss_count: int = 0       # 連続でマッチングに失敗したフレーム数
    # 過去数フレームの中心座標履歴 (deque)
    center_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def update(self, bbox: np.ndarray):
        """新しい検出結果でトラック状態を更新する"""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        new_center = np.array([cx, cy])

        # 速度 = 今フレームの中心 - 前フレームの中心
        if len(self.center_history) > 0:
            self.velocity = new_center - self.center_history[-1]
        else:
            self.velocity = np.array([0.0, 0.0])

        self.bbox = bbox.copy()
        self.center = new_center
        self.center_history.append(new_center.copy())
        self.age += 1
        self.miss_count = 0

    def predict(self):
        """マッチングに失敗したフレームで等速モデルで位置を予測する"""
        self.bbox = self.bbox + np.array([
            self.velocity[0], self.velocity[1],
            self.velocity[0], self.velocity[1]
        ])
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.center = np.array([cx, cy])
        self.miss_count += 1

    def smoothed_velocity(self, n_frames: int = 5) -> np.ndarray:
        """
        過去 n_frames 間の平均速度ベクトルを返す（方向推定に使用）。
        履歴が足りない場合は現在の velocity を返す。
        """
        history = list(self.center_history)
        if len(history) < 2:
            return self.velocity.copy()
        # 最大 n_frames フレーム前との差分の平均
        n = min(n_frames, len(history) - 1)
        diffs = []
        for i in range(1, n + 1):
            diffs.append(history[-i] - history[-i - 1] if i + 1 <= len(history) else history[-1] - history[-2])
        return np.mean(diffs, axis=0)

    def direction_angle_deg(self, n_frames: int = 5) -> Optional[float]:
        """
        移動方向を角度（度, -180〜180）で返す。
        十分な履歴がない場合は None。
        """
        vel = self.smoothed_velocity(n_frames)
        speed = np.linalg.norm(vel)
        if speed < 0.5:  # ほぼ静止とみなす閾値
            return None
        angle_rad = np.arctan2(vel[1], vel[0])
        return float(np.degrees(angle_rad))


# ---------------------------------------------------------------------------
# IoU ユーティリティ
# ---------------------------------------------------------------------------

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """2つのバウンディングボックス（[x1,y1,x2,y2]）の IoU を計算する"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area < 1e-6:
        return 0.0
    return float(inter_area / union_area)


def iou_matrix(tracks: List[TrackState], detections: List[np.ndarray]) -> np.ndarray:
    """トラックリスト × 検出リストの IoU 行列 shape=(len(tracks), len(detections))"""
    mat = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            mat[i, j] = compute_iou(trk.bbox, det)
    return mat


def hungarian_match(cost_matrix: np.ndarray, threshold: float = 0.3) -> List[Tuple[int, int]]:
    """
    コスト行列からグリーディーマッチングを行う（IoU を最大化）。
    scipy が使える場合は linear_sum_assignment を使い、なければグリーディーにフォールバック。
    threshold 未満の IoU のペアはマッチングしない。

    Returns
    -------
    matches: [(track_idx, det_idx), ...]
    """
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= threshold:
                matches.append((int(r), int(c)))
        return matches
    except ImportError:
        # scipy なしのグリーディーマッチング
        matches = []
        used_rows = set()
        used_cols = set()
        flat_indices = np.argsort(-cost_matrix.flatten())
        for idx in flat_indices:
            r = idx // cost_matrix.shape[1]
            c = idx % cost_matrix.shape[1]
            if r in used_rows or c in used_cols:
                continue
            if cost_matrix[r, c] < threshold:
                break
            matches.append((int(r), int(c)))
            used_rows.add(r)
            used_cols.add(c)
        return matches


# ---------------------------------------------------------------------------
# SimpleTracker 本体
# ---------------------------------------------------------------------------

class SimpleTracker:
    """
    YOLOv8 検出結果を受け取り、IoU ベースでトラッキングを行うクラス。

    使い方:
        tracker = SimpleTracker()
        for frame in video:
            detections = yolo_detect(frame)   # List[np.ndarray] shape=(N,4)
            tracks = tracker.update(detections)
            for trk in tracks:
                print(trk.track_id, trk.direction_angle_deg())
    """

    # YOLOv8 の車両クラスID（COCO）
    VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car=2, motorcycle=3, bus=5, truck=7

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_miss_frames: int = 5,
        min_age_to_report: int = 3,
        conf_threshold: float = 0.4,
    ):
        self.iou_threshold = iou_threshold
        self.max_miss_frames = max_miss_frames
        self.min_age_to_report = min_age_to_report
        self.conf_threshold = conf_threshold

        self._next_id: int = 1
        self._active_tracks: Dict[int, TrackState] = {}

    # ------------------------------------------------------------------
    # YOLOv8 結果のパース
    # ------------------------------------------------------------------

    @staticmethod
    def parse_yolo_results(results) -> List[np.ndarray]:
        """
        Ultralytics YOLO の results オブジェクトから
        車両クラスの bbox リスト [x1,y1,x2,y2] を返す。

        Parameters
        ----------
        results : ultralytics YOLO results (results[0])
        """
        bboxes = []
        if results is None or len(results) == 0:
            return bboxes

        r = results[0]
        if r.boxes is None:
            return bboxes

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()   # shape=(N,4)
        cls  = boxes.cls.cpu().numpy()    # shape=(N,)
        conf = boxes.conf.cpu().numpy()   # shape=(N,)

        for i in range(len(xyxy)):
            if int(cls[i]) in SimpleTracker.VEHICLE_CLASS_IDS and conf[i] >= 0.4:
                bboxes.append(xyxy[i].astype(np.float32))

        return bboxes

    # ------------------------------------------------------------------
    # トラッキング更新
    # ------------------------------------------------------------------

    def update(self, detections: List[np.ndarray]) -> List[TrackState]:
        """
        1フレーム分の検出結果でトラックを更新し、アクティブなトラック一覧を返す。

        Parameters
        ----------
        detections : List of [x1,y1,x2,y2] arrays

        Returns
        -------
        tracks : 現在アクティブで min_age_to_report 以上継続しているトラック一覧
        """
        active_list = list(self._active_tracks.values())

        if len(active_list) == 0:
            # 全て新規トラック
            for det in detections:
                self._create_track(det)
        elif len(detections) == 0:
            # 全トラックをミス扱い
            for trk in active_list:
                trk.predict()
        else:
            iou_mat = iou_matrix(active_list, detections)
            matches = hungarian_match(iou_mat, threshold=self.iou_threshold)

            matched_trk_idx = {m[0] for m in matches}
            matched_det_idx = {m[1] for m in matches}

            # マッチしたトラックを更新
            for trk_idx, det_idx in matches:
                active_list[trk_idx].update(detections[det_idx])

            # アンマッチのトラックは予測で継続
            for i, trk in enumerate(active_list):
                if i not in matched_trk_idx:
                    trk.predict()

            # アンマッチの検出は新規トラックとして生成
            for j, det in enumerate(detections):
                if j not in matched_det_idx:
                    self._create_track(det)

        # miss が多すぎるトラックを削除
        to_delete = [tid for tid, trk in self._active_tracks.items()
                     if trk.miss_count > self.max_miss_frames]
        for tid in to_delete:
            del self._active_tracks[tid]

        # 報告対象（十分に追跡できているもの）を返す
        return [trk for trk in self._active_tracks.values()
                if trk.age >= self.min_age_to_report]

    def _create_track(self, bbox: np.ndarray) -> TrackState:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        trk = TrackState(
            track_id=self._next_id,
            bbox=bbox.copy(),
            center=np.array([cx, cy]),
            velocity=np.array([0.0, 0.0]),
        )
        trk.center_history.append(trk.center.copy())
        self._active_tracks[self._next_id] = trk
        self._next_id += 1
        return trk

    def reset(self):
        self._active_tracks.clear()
        self._next_id = 1

    @property
    def active_tracks(self) -> Dict[int, TrackState]:
        return self._active_tracks

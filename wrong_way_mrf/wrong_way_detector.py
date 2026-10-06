"""
wrong_way_detector.py
---------------------
逆走判定モジュール

役割:
  1. カメラ映像から「正常走行方向」を自動推定する
  2. 各車両の移動方向と正常走行方向を比較して観測ポテンシャルを生成する
  3. SpatioTemporalMRF を使ったグラフを構築・推論し逆走確率を算出する
  4. スライディングウィンドウで時空間グラフを管理する（メモリ効率）
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from mrf_core import SpatioTemporalMRF, observation_potential_from_angle
from tracker import TrackState


# ---------------------------------------------------------------------------
# 正常走行方向の推定
# ---------------------------------------------------------------------------

class NormalDirectionEstimator:
    """
    画面内の多数の車両速度ベクトルの最頻方向を「正常走行方向」として推定する。

    アルゴリズム:
      - 各フレームで全追跡車両の方向角を収集
      - ヒストグラム（36ビン、10°刻み）の最頻方向を正常方向とする
      - EMAで時間的に平滑化する
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        min_vehicles: int = 3,
        histogram_bins: int = 36,
        warmup_frames: int = 30,
    ):
        self.ema_alpha = ema_alpha
        self.min_vehicles = min_vehicles
        self.histogram_bins = histogram_bins
        self.warmup_frames = warmup_frames

        self._estimated_direction: Optional[float] = None
        self._frame_count: int = 0
        self._angle_buffer: deque = deque(maxlen=200)  # 蓄積バッファ

    def update(self, tracks: List[TrackState]) -> Optional[float]:
        """
        追跡車両リストを受け取り、正常走行方向の推定値を更新・返す。

        Returns
        -------
        方向角（度, -180〜180）または None（推定未完了）
        """
        self._frame_count += 1
        valid_angles = []
        for trk in tracks:
            angle = trk.direction_angle_deg()
            if angle is not None:
                valid_angles.append(angle)

        if len(valid_angles) >= self.min_vehicles:
            self._angle_buffer.extend(valid_angles)

        # ウォームアップ期間が過ぎたら推定
        if self._frame_count >= self.warmup_frames and len(self._angle_buffer) >= self.min_vehicles:
            dominant = self._dominant_direction(list(self._angle_buffer))
            if dominant is not None:
                if self._estimated_direction is None:
                    self._estimated_direction = dominant
                else:
                    # EMA で滑らかに更新
                    # 角度のEMAは差分が ±180 を超えないように処理
                    diff = dominant - self._estimated_direction
                    diff = (diff + 180) % 360 - 180  # -180〜180 に正規化
                    self._estimated_direction += self.ema_alpha * diff
                    self._estimated_direction = (self._estimated_direction + 180) % 360 - 180

        return self._estimated_direction

    def _dominant_direction(self, angles: List[float]) -> Optional[float]:
        """角度リストの最頻方向（ヒストグラムの最大ビン）を返す"""
        if not angles:
            return None
        # -180〜180 を 0〜360 に変換してヒストグラム化
        angles_360 = [(a + 360) % 360 for a in angles]
        hist, bin_edges = np.histogram(angles_360, bins=self.histogram_bins, range=(0, 360))
        peak_bin = int(np.argmax(hist))
        dominant_deg = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
        # 0〜360 を -180〜180 に戻す
        dominant_deg = (dominant_deg + 180) % 360 - 180
        return float(dominant_deg)

    def set_manual_direction(self, angle_deg: float):
        """正常走行方向を手動で設定する（既知の場合に使用）"""
        self._estimated_direction = angle_deg

    @property
    def estimated_direction(self) -> Optional[float]:
        return self._estimated_direction

    @property
    def is_ready(self) -> bool:
        return self._estimated_direction is not None


# ---------------------------------------------------------------------------
# 角度差の計算
# ---------------------------------------------------------------------------

def angle_difference(angle_a: float, angle_b: float) -> float:
    """
    2つの方向角（度）の最小差を返す（0〜180）。
    例: 170° と -170° の差は 20°
    """
    diff = abs(angle_a - angle_b)
    if diff > 180:
        diff = 360 - diff
    return float(diff)


# ---------------------------------------------------------------------------
# 逆走判定器本体
# ---------------------------------------------------------------------------

class WrongWayDetector:
    """
    時空間MRFを用いた逆走判定器。

    スライディングウィンドウでフレームを管理し、
    window_size フレーム分のグラフを構築して LBP 推論を行う。

    Usage:
        detector = WrongWayDetector()
        for frame_idx, tracks in enumerate(per_frame_tracks):
            results = detector.update(frame_idx, tracks)
            # results: {track_id: wrong_way_probability}
    """

    def __init__(
        self,
        window_size: int = 8,
        temporal_weight: float = 4.0,
        spatial_weight: float = 2.0,
        spatial_distance_thresh: float = 200.0,
        bp_iterations: int = 10,
        wrongway_threshold: float = 0.6,
        angle_sigma_deg: float = 30.0,
        normal_direction_deg: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        window_size            : 時空間グラフのスライディングウィンドウ幅（フレーム数）
        temporal_weight        : 時間エッジのポテンシャル強度
        spatial_weight         : 空間エッジのポテンシャル強度
        spatial_distance_thresh: 空間エッジを張る距離閾値（ピクセル）
        bp_iterations          : LBP の反復回数
        wrongway_threshold     : 逆走判定の確率閾値
        angle_sigma_deg        : 方向角→ポテンシャル変換のガウシアン幅
        normal_direction_deg   : 正常走行方向（None の場合は自動推定）
        """
        self.window_size = window_size
        self.wrongway_threshold = wrongway_threshold
        self.angle_sigma_deg = angle_sigma_deg

        self._mrf = SpatioTemporalMRF(
            temporal_weight=temporal_weight,
            spatial_weight=spatial_weight,
            spatial_distance_thresh=spatial_distance_thresh,
            bp_iterations=bp_iterations,
        )

        self._direction_estimator = NormalDirectionEstimator()
        if normal_direction_deg is not None:
            self._direction_estimator.set_manual_direction(normal_direction_deg)

        # スライディングウィンドウ: {frame_idx: {vehicle_id: TrackState}}
        self._window: deque = deque(maxlen=window_size)
        self._frame_indices: deque = deque(maxlen=window_size)

        # 最新の逆走確率キャッシュ
        self._latest_probs: Dict[int, float] = {}
        # 逆走確認カウンタ（連続フレームで逆走と判定された回数）
        self._wrongway_count: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # メインのフレーム更新
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        tracks: List[TrackState],
    ) -> Dict[int, float]:
        """
        1フレーム分の追跡結果を受け取り、各車両の逆走確率を返す。

        Parameters
        ----------
        frame_idx : 現在のフレーム番号
        tracks    : SimpleTracker から得られたトラックリスト

        Returns
        -------
        probs : {vehicle_id: wrong_way_probability (0〜1)}
        """
        # 正常走行方向の更新
        normal_dir = self._direction_estimator.update(tracks)

        # フレームデータをウィンドウに追加
        frame_data: Dict[int, TrackState] = {trk.track_id: trk for trk in tracks}
        self._window.append(frame_data)
        self._frame_indices.append(frame_idx)

        # 方向推定が未完了なら簡易判定で返す
        if not self._direction_estimator.is_ready:
            return self._fallback_detection(tracks)

        # MRFグラフを再構築して推論
        probs = self._run_mrf_inference(normal_dir)

        # 逆走カウンタ更新（MRF推論後）
        all_vids = {trk.track_id for trk in tracks}
        for vid in all_vids:
            p = probs.get(vid, 0.0)
            if p >= self.wrongway_threshold:
                self._wrongway_count[vid] += 1
            else:
                # 正常に戻ったらカウントを減らす（急に0にしない）
                self._wrongway_count[vid] = max(0, self._wrongway_count[vid] - 1)

        self._latest_probs = probs
        return probs

    # ------------------------------------------------------------------
    # MRF グラフ構築 + 推論
    # ------------------------------------------------------------------

    def _run_mrf_inference(self, normal_direction_deg: float) -> Dict[int, float]:
        """ウィンドウ内の全データでMRFグラフを構築してBPを実行する"""
        self._mrf.clear()

        frames_list = list(self._frame_indices)
        window_list = list(self._window)

        # ノードを追加
        for f_pos, (frame_idx, frame_data) in enumerate(zip(frames_list, window_list)):
            for vid, trk in frame_data.items():
                obs_pot = self._compute_obs_potential(trk, normal_direction_deg)
                self._mrf.add_node(vid, frame_idx, obs_pot)

        # 時間エッジを追加
        for f_pos in range(1, len(frames_list)):
            prev_frame = frames_list[f_pos - 1]
            curr_frame = frames_list[f_pos]
            prev_data = window_list[f_pos - 1]
            curr_data = window_list[f_pos]
            # 両フレームに存在する車両に時間エッジ
            common_vids = set(prev_data.keys()) & set(curr_data.keys())
            for vid in common_vids:
                self._mrf.build_temporal_edges(vid, prev_frame, curr_frame)

        # 空間エッジを追加（最新フレームのみ）
        if len(frames_list) > 0:
            latest_frame = frames_list[-1]
            latest_data = window_list[-1]
            positions = {
                vid: (trk.center[0], trk.center[1])
                for vid, trk in latest_data.items()
            }
            self._mrf.build_spatial_edges(latest_frame, positions)

        # BP 推論
        beliefs = self._mrf.run_belief_propagation()

        # 最新フレームの各車両の逆走確率を返す
        probs: Dict[int, float] = {}
        if len(frames_list) > 0:
            latest_frame = frames_list[-1]
            latest_data = window_list[-1]
            for vid in latest_data.keys():
                p = self._mrf.get_wrongway_prob(vid, latest_frame, beliefs)
                probs[vid] = p

        return probs

    # ------------------------------------------------------------------
    # 観測ポテンシャル計算
    # ------------------------------------------------------------------

    def _compute_obs_potential(
        self, trk: TrackState, normal_direction_deg: float
    ) -> np.ndarray:
        """
        車両の移動方向と正常走行方向の差から観測ポテンシャルを計算する。
        方向が不明（静止中など）の場合は uniform ポテンシャルを返す。
        """
        angle = trk.direction_angle_deg()
        if angle is None:
            return np.array([0.5, 0.5])

        diff = angle_difference(angle, normal_direction_deg)
        return observation_potential_from_angle(
            angle_diff_deg=diff,
            sigma_deg=self.angle_sigma_deg,
        )

    # ------------------------------------------------------------------
    # 方向推定前のフォールバック（閾値ベース）
    # ------------------------------------------------------------------

    def _fallback_detection(self, tracks: List[TrackState]) -> Dict[int, float]:
        """
        方向推定ウォームアップ前の簡易判定。
        全車両の中央値方向から大きく外れる車両を逆走候補とする。
        """
        angles = []
        for trk in tracks:
            a = trk.direction_angle_deg()
            if a is not None:
                angles.append((trk.track_id, a))

        if len(angles) < 2:
            return {trk.track_id: 0.0 for trk in tracks}

        angle_values = [a for _, a in angles]
        median_angle = float(np.median(angle_values))

        probs: Dict[int, float] = {}
        for vid, angle in angles:
            diff = angle_difference(angle, median_angle)
            pot = observation_potential_from_angle(diff, sigma_deg=self.angle_sigma_deg)
            probs[vid] = float(pot[1])

        # 方向不明の車両は 0
        known_vids = {vid for vid, _ in angles}
        for trk in tracks:
            if trk.track_id not in known_vids:
                probs[trk.track_id] = 0.0

        return probs

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def is_wrong_way(self, vehicle_id: int, min_consecutive: int = 3) -> bool:
        """
        指定車両が逆走しているかどうかをブーリアンで返す。
        min_consecutive フレーム以上連続して逆走確率が閾値を超えている場合のみ True。
        """
        return self._wrongway_count.get(vehicle_id, 0) >= min_consecutive

    def get_probability(self, vehicle_id: int) -> float:
        """最新フレームでの逆走確率を返す"""
        return self._latest_probs.get(vehicle_id, 0.0)

    @property
    def normal_direction(self) -> Optional[float]:
        return self._direction_estimator.estimated_direction

    @property
    def direction_estimator(self) -> NormalDirectionEstimator:
        return self._direction_estimator

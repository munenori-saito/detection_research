"""
mrf_core.py
-----------
時空間 MRF（Markov Random Field）コアモジュール

グラフ構造:
  ・ノード  : 各車両 × 各フレームにおける「逆走状態」（0=正常, 1=逆走）
  ・時間エッジ: 同一車両の連続フレーム間（時間的一貫性）
  ・空間エッジ: 同フレーム内の空間的に隣接する車両間（周囲の交通流との一貫性）

推論:
  Loopy Belief Propagation（LBP）で各ノードの周辺確率を近似計算する。
  外部ライブラリに依存せずに numpy だけで実装。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# データ型定義
# ---------------------------------------------------------------------------

@dataclass
class MRFNode:
    """MRF の1ノード（車両ID × フレーム番号）"""
    vehicle_id: int
    frame_idx: int
    # 観測ポテンシャル: [P(正常 | 観測), P(逆走 | 観測)]
    observation_potential: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    # LBP メッセージバッファ: key = 隣接ノードID, value = shape(2,) array
    messages: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        return f"v{self.vehicle_id}_f{self.frame_idx}"


@dataclass
class MRFEdge:
    """MRF の1エッジ"""
    node_a_id: str
    node_b_id: str
    # ペアワイズポテンシャル shape=(2,2)
    # pairwise[i,j] = Ψ(状態i, 状態j) の値（非正規化）
    pairwise_potential: np.ndarray = field(default_factory=lambda: np.eye(2))
    edge_type: str = "temporal"  # "temporal" or "spatial"


# ---------------------------------------------------------------------------
# ポテンシャル生成ヘルパー
# ---------------------------------------------------------------------------

def make_temporal_potential(consistency_weight: float = 4.0) -> np.ndarray:
    """
    時間エッジ用ペアワイズポテンシャル。
    同じ状態が連続する場合のスコアを高くし、急激な逆走→正常の変化を抑制する。

    consistency_weight が大きいほど状態が時間的に安定する（スムージング強）。
    """
    pot = np.array([[consistency_weight, 1.0],
                    [1.0, consistency_weight]])
    return pot


def make_spatial_potential(cooperation_weight: float = 2.0) -> np.ndarray:
    """
    空間エッジ用ペアワイズポテンシャル。
    隣接車両が同じ方向に走っている場合のスコアを高くする。
    ただし逆走車が周囲に引きずられすぎないように temporal より弱め。
    """
    pot = np.array([[cooperation_weight, 1.0],
                    [1.0, cooperation_weight]])
    return pot


def observation_potential_from_angle(
    angle_diff_deg: float,
    normal_direction_deg: float = 0.0,
    sigma_deg: float = 30.0,
) -> np.ndarray:
    """
    車両の移動方向角度差から観測ポテンシャルを計算する。

    Parameters
    ----------
    angle_diff_deg   : 推定進行方向と正常方向の角度差（絶対値, 0〜180度）
    normal_direction_deg : 正常走行方向（参照値, 通常0として正規化済みで渡す）
    sigma_deg        : ガウシアン幅（大きいほど曖昧な判定）

    Returns
    -------
    np.ndarray shape=(2,)  [P_normal, P_wrongway]（正規化済み）
    """
    # 正常方向への近さをガウシアンでスコア化
    score_normal = np.exp(-0.5 * (angle_diff_deg / sigma_deg) ** 2)
    # 逆走側（180度差に近いほど高スコア）
    score_wrong = np.exp(-0.5 * ((180.0 - angle_diff_deg) / sigma_deg) ** 2)

    total = score_normal + score_wrong + 1e-9
    return np.array([score_normal / total, score_wrong / total])


# ---------------------------------------------------------------------------
# MRF グラフ
# ---------------------------------------------------------------------------

class SpatioTemporalMRF:
    """
    時空間 MRF グラフ本体。
    フレームごとに車両を登録し、エッジを自動生成して LBP で推論する。
    """

    def __init__(
        self,
        temporal_weight: float = 4.0,
        spatial_weight: float = 2.0,
        spatial_distance_thresh: float = 150.0,  # ピクセル
        bp_iterations: int = 10,
    ):
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self.spatial_distance_thresh = spatial_distance_thresh
        self.bp_iterations = bp_iterations

        self._nodes: Dict[str, MRFNode] = {}
        self._edges: List[MRFEdge] = []
        # 隣接リスト: node_id -> [隣接 node_id, ...]
        self._adjacency: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # ノード操作
    # ------------------------------------------------------------------

    def add_node(self, vehicle_id: int, frame_idx: int, obs_potential: np.ndarray) -> MRFNode:
        node = MRFNode(
            vehicle_id=vehicle_id,
            frame_idx=frame_idx,
            observation_potential=obs_potential.copy(),
        )
        self._nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, [])
        return node

    def _add_edge(self, node_a_id: str, node_b_id: str, pairwise: np.ndarray, etype: str):
        edge = MRFEdge(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            pairwise_potential=pairwise.copy(),
            edge_type=etype,
        )
        self._edges.append(edge)
        self._adjacency[node_a_id].append(node_b_id)
        self._adjacency[node_b_id].append(node_a_id)

    # ------------------------------------------------------------------
    # エッジ自動生成
    # ------------------------------------------------------------------

    def build_temporal_edges(self, vehicle_id: int, frame_a: int, frame_b: int):
        """同一車両の連続フレーム間に時間エッジを張る"""
        id_a = f"v{vehicle_id}_f{frame_a}"
        id_b = f"v{vehicle_id}_f{frame_b}"
        if id_a in self._nodes and id_b in self._nodes:
            pot = make_temporal_potential(self.temporal_weight)
            self._add_edge(id_a, id_b, pot, "temporal")

    def build_spatial_edges(self, frame_idx: int, vehicle_positions: Dict[int, Tuple[float, float]]):
        """
        同フレーム内で distance < spatial_distance_thresh の車両ペアに空間エッジを張る。

        Parameters
        ----------
        frame_idx        : 対象フレーム
        vehicle_positions: {vehicle_id: (cx, cy)} の辞書
        """
        ids = list(vehicle_positions.keys())
        pot = make_spatial_potential(self.spatial_weight)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                vid_i, vid_j = ids[i], ids[j]
                pos_i = np.array(vehicle_positions[vid_i])
                pos_j = np.array(vehicle_positions[vid_j])
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < self.spatial_distance_thresh:
                    id_i = f"v{vid_i}_f{frame_idx}"
                    id_j = f"v{vid_j}_f{frame_idx}"
                    if id_i in self._nodes and id_j in self._nodes:
                        self._add_edge(id_i, id_j, pot, "spatial")

    # ------------------------------------------------------------------
    # Belief Propagation（Loopy BP）
    # ------------------------------------------------------------------

    def _initialize_messages(self):
        """全メッセージを uniform に初期化"""
        for node in self._nodes.values():
            node.messages = {}
            for neighbor_id in self._adjacency[node.node_id]:
                node.messages[neighbor_id] = np.array([1.0, 1.0])

    def _get_pairwise(self, node_a_id: str, node_b_id: str) -> np.ndarray:
        """エッジリストから対応するペアワイズポテンシャルを返す"""
        for edge in self._edges:
            if (edge.node_a_id == node_a_id and edge.node_b_id == node_b_id) or \
               (edge.node_a_id == node_b_id and edge.node_b_id == node_a_id):
                return edge.pairwise_potential
        return np.ones((2, 2))

    def run_belief_propagation(self) -> Dict[str, np.ndarray]:
        """
        Loopy Belief Propagation を実行し、各ノードの周辺確率を返す。

        Returns
        -------
        beliefs: {node_id: np.ndarray shape=(2,)}  [P(正常), P(逆走)]
        """
        self._initialize_messages()

        # ノードIDリストを固定（更新順序を安定させる）
        node_ids = list(self._nodes.keys())

        for _iter in range(self.bp_iterations):
            new_messages: Dict[Tuple[str, str], np.ndarray] = {}

            for node_id in node_ids:
                node = self._nodes[node_id]
                for neighbor_id in self._adjacency[node_id]:
                    # node_id → neighbor_id へのメッセージを計算
                    pairwise = self._get_pairwise(node_id, neighbor_id)

                    # node の belief（neighbor からのメッセージを除いた積）
                    belief = node.observation_potential.copy()
                    for other_neighbor_id, msg in node.messages.items():
                        if other_neighbor_id != neighbor_id:
                            belief = belief * msg

                    # メッセージ計算: m_{node→neighbor}(x_neighbor) = sum_{x_node} Ψ * belief
                    msg_new = np.zeros(2)
                    for state_neighbor in range(2):
                        val = 0.0
                        for state_node in range(2):
                            val += pairwise[state_node, state_neighbor] * belief[state_node]
                        msg_new[state_neighbor] = val

                    # 正規化（数値安定性）
                    msg_sum = msg_new.sum()
                    if msg_sum > 1e-12:
                        msg_new /= msg_sum

                    new_messages[(node_id, neighbor_id)] = msg_new

            # メッセージを一括更新（synchronous BP）
            for (src_id, dst_id), msg in new_messages.items():
                self._nodes[dst_id].messages[src_id] = msg

        # 各ノードの belief を計算
        beliefs: Dict[str, np.ndarray] = {}
        for node_id, node in self._nodes.items():
            b = node.observation_potential.copy()
            for msg in node.messages.values():
                b = b * msg
            b_sum = b.sum()
            if b_sum > 1e-12:
                b /= b_sum
            beliefs[node_id] = b

        return beliefs

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def get_wrongway_prob(self, vehicle_id: int, frame_idx: int, beliefs: Dict[str, np.ndarray]) -> float:
        """指定車両・フレームの逆走確率を返す"""
        node_id = f"v{vehicle_id}_f{frame_idx}"
        if node_id in beliefs:
            return float(beliefs[node_id][1])
        return 0.0

    def clear(self):
        """グラフをリセット"""
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

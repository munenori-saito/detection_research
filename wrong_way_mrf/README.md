# 時空間MRF 逆走検知デモ

YOLOv8 による車両検出と **時空間マルコフ確率場（Spatio-Temporal MRF）** を組み合わせた逆走車検知デモシステムです。

---

## システム概要

### なぜ MRF を使うのか

単純に「この1フレームでの移動方向が逆向きかどうか」だけで判定すると、次の問題が起きます。

- **1フレームの誤検出に弱い** → 車線変更・旋回中に誤報が出やすい
- **孤立した判定** → 周囲の交通流との矛盾を考慮できない

時空間MRFを使うと、各車両の判定を「時間的な前後フレーム」と「空間的な隣接車両」の両方と結びつけて確率的に推論するため、より安定した検知が可能になります。

### グラフ構造

```
フレーム t-2     フレーム t-1     フレーム t
  [車A]  ──時間──  [車A]  ──時間──  [車A]
    │                │                │
  空間             空間             空間
    │                │                │
  [車B]  ──時間──  [車B]  ──時間──  [車B]
```

- **ノード** : 各車両 × 各フレームの「状態」（正常 / 逆走）
- **時間エッジ** : 同一車両の連続フレーム間 → 時間的な一貫性を強化
- **空間エッジ** : 同フレーム内の近接車両間 → 周囲の交通流との整合性

### アルゴリズムの流れ

```
入力動画
  │
  ▼
YOLOv8（車両クラスのみ検出: car / truck / bus / motorcycle）
  │
  ▼
SimpleTracker（IoU ベーストラッキング）
  │  各車両に一貫した ID を付与、速度ベクトルを計算
  ▼
NormalDirectionEstimator（正常走行方向の自動推定）
  │  最初の30フレームは全車両の方向ヒストグラムから推定
  ▼
WrongWayDetector（時空間MRF + Loopy Belief Propagation）
  │  スライディングウィンドウ（8フレーム）でグラフを構築
  │  LBP で各ノードの逆走確率を推論
  ▼
可視化 + 出力動画
```

---

## ファイル構成

```
wrong_way_mrf/
├── mrf_core.py           # MRF グラフ・LBP コア実装
├── tracker.py            # YOLOv8 + IoU トラッカー
├── wrong_way_detector.py # 逆走判定ロジック（MRF 統合）
├── demo.py               # メインデモスクリプト（CLI）
├── requirements.txt      # 依存ライブラリ
└── README.md             # このファイル
```

---

## セットアップ

### 1. 依存ライブラリのインストール

```bash
cd wrong_way_mrf
pip install -r requirements.txt
```

### 2. YOLOv8 モデルの確認

`yolo_edge_research/yolov8n.pt` が既に存在しています（再ダウンロード不要）。

---

## 実行方法

### 基本実行（デフォルト設定）

```bash
cd wrong_way_mrf
python demo.py
```

デフォルトでは `../video/benchmark_traffic.mp4` を処理し、`output_wrongway.mp4` を出力します。

### オプション指定の例

```bash
# 入力・出力ファイルを明示的に指定
python demo.py --input ../video/benchmark_traffic.mp4 --output result.mp4

# 正常走行方向を手動指定（右向き = 0°、下向き = 90°、左向き = ±180°）
python demo.py --normal-direction 0

# リアルタイム表示なしでバッチ処理（サーバー環境向け）
python demo.py --no-display

# テスト用に最初の 300 フレームのみ処理
python demo.py --max-frames 300

# OpenVINO INT8 モデルを使用（高速化）
python demo.py --model ../yolo_edge_research/yolov8n_int8_openvino_model

# 逆走判定の感度を調整（デフォルト: 0.6）
python demo.py --threshold 0.5
```

### 全オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--input` / `-i` | `../video/benchmark_traffic.mp4` | 入力動画ファイル |
| `--output` / `-o` | `output_wrongway.mp4` | 出力動画ファイル |
| `--model` / `-m` | `../yolo_edge_research/yolov8n.pt` | YOLOv8 モデルパス |
| `--imgsz` | `640` | YOLO 推論サイズ（px） |
| `--conf` | `0.4` | YOLO 検出信頼度閾値 |
| `--threshold` | `0.6` | MRF 逆走確率の判定閾値 |
| `--normal-direction` | `None`（自動推定） | 正常走行方向（度） |
| `--max-frames` | `0`（無制限） | 処理フレーム数の上限 |
| `--no-save` | False | 動画保存をスキップ |
| `--no-display` | False | リアルタイム表示をスキップ |

---

## 出力映像の見方

| 色 | 意味 |
|---|---|
| 緑の枠 | 正常走行（逆走確率 < 0.4） |
| 黄の枠 | 要注意（逆走確率 0.4〜0.6） |
| 橙の枠 | 高リスク（逆走確率 0.6 以上） |
| 赤の枠 + WRONG WAY! | 逆走確定（3フレーム以上連続で高確率） |
| 矢印 | 車両の移動方向ベクトル |
| 右上コンパス | 推定された正常走行方向 |

---

## パラメータのチューニング指針

### 誤報が多い場合

- `--threshold` を上げる（例: 0.7）
- `temporal_weight` を上げる（`wrong_way_detector.py` 内）
- `min_consecutive` を上げる（`demo.py` の `is_wrong_way` 呼び出し箇所）

### 検出が遅い・見逃す場合

- `--threshold` を下げる（例: 0.5）
- `window_size` を増やす（`WrongWayDetector` の初期化）
- `--normal-direction` で正確な方向を手動指定

### 処理速度が遅い場合

- `--model` に OpenVINO INT8 モデルを指定
- `--imgsz 416` に落とす
- `--no-display` で表示をオフにする

---

## 技術的な補足

### Loopy Belief Propagation について

厳密な推論が保証される Tree-BP とは異なり、閉路を含むグラフ（Loopy MRF）では LBP は近似推論です。ただし実用上は多くのケースで十分な精度が得られることが知られており、今回のような車両の時空間グラフでも有効です。

### 正常走行方向の自動推定

最初の 30 フレーム（ウォームアップ期間）は、全車両の移動方向のヒストグラムから最頻方向を正常走行方向とします。この間は MRF の代わりに中央値ベースの簡易判定を行います。

カメラが動く場合や交差点などでは手動指定（`--normal-direction`）のほうが安定します。

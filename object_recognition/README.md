# object_recognition — 物体認識学習用コード集

NumPy のみで実装した、画像分類のフルパイプライン。
**Preprocessor × FeatureExtractor × Classifier** の各機能ごとに
複数のアルゴリズムを用意し、戦略パターンで切り替えられる。

## ディレクトリ構成

```
object_recognition/
├── data.py        # 合成図形データ (circle / square / triangle)
├── preprocess.py  # 前処理 4 種
├── feature.py     # 特徴抽出 4 種
├── classifier.py  # 多クラス分類器 4 種
├── metrics.py     # 評価指標 (混同行列付き)
├── pipeline.py    # 統合 API
├── main.py        # 全 64 通り比較
└── README.md
```

## 各機能のアルゴリズム選択肢

### Preprocessor (4 種)

| 名前 | クラス | 効果 |
|------|--------|------|
| `noop`     | `NoOp`              | 変換しない |
| `minmax`   | `MinMaxScaler`      | 全画素を [0,1] に線形変換 |
| `standard` | `StandardScaler`    | 平均 0 / 分散 1 に標準化 |
| `hist`     | `HistogramEqualizer`| 各画像のヒストグラム均等化 (照明変化に強い) |

### FeatureExtractor (4 種)

| 名前 | クラス | 計算内容 |
|------|--------|----------|
| `raw`  | `RawPixels` | 画素値を flatten したベースライン |
| `hog`  | `HOG`       | 勾配方向ヒストグラム、cell→block 階層、L2-Hys 正規化 |
| `lbp`  | `LBP`       | 各画素の 8 近傍を比較した 8bit パターンの 256bin ヒストグラム |
| `edge` | `EdgeHist`  | Sobel エッジの方向ヒストグラム (HOG の簡易版) |

### Classifier (4 種, すべて多クラス対応)

| 名前 | クラス | アルゴリズム |
|------|--------|--------------|
| `knn`     | `KNNClassifier`     | k 近傍 + コサイン距離 + 距離重み付き投票 |
| `softmax` | `SoftmaxRegression` | 多項ロジスティック回帰 + L2 正則化 |
| `svm`     | `OvRLinearSVM`      | One-vs-Rest 線形 SVM (Pegasos 平均化 SGD) |
| `mlp`     | `MLPClassifier`     | 2 層 MLP + Adam + He 初期化 |

## 重要な数式

### HOG (Histogram of Oriented Gradients)
```
gy = ∂I/∂y,  gx = ∂I/∂x
magnitude   = √(gx² + gy²)
orientation = atan2(gy, gx)            # 符号なしなら mod π
セル: 各 cell × cell 領域で方向別の重み付きヒストグラム
ブロック: block × block セルを連結し L2-Hys 正規化
   v ← v / √(‖v‖² + ε²)  → clip(v, 0.2) → 再正規化
```

### LBP (Local Binary Pattern)
```
LBP(c) = Σ_{k=0..7} [ I(neighbor_k) ≥ I(c) ] · 2^k
特徴量: LBP 値の 0..255 ヒストグラム (正規化)
```

### Softmax Regression (多項 LR)
```
P(y=c|x) = exp(w_c·x + b_c) / Σ_k exp(w_k·x + b_k)
L = -1/N Σ log P(y_i|x_i) + 0.5λ‖W‖²
```
log-sum-exp トリックで overflow を回避: `Z ← Z - max(Z)`

### MLP + Adam
```
He 初期化: W ~ N(0, sqrt(2/n_in))
Adam:  m ← β₁m + (1-β₁)g
       v ← β₂v + (1-β₂)g²
       θ ← θ - α · m̂ / (√v̂ + ε)
```

## 使い方

```python
from data       import make_shapes, train_test_split
from preprocess import get_preprocessor
from feature    import get_feature
from classifier import get_classifier
from pipeline   import ObjectRecognitionPipeline
from metrics    import classification_report

ds = make_shapes(n_per_class=200, size=24)
train, test = train_test_split(ds)

pipe = ObjectRecognitionPipeline(
    preprocessor = get_preprocessor("standard"),
    feature      = get_feature("hog"),
    classifier   = get_classifier("softmax"),
)
pipe.fit(train.images, train.labels)
preds = pipe.predict(test.images)
print(classification_report(test.labels, preds, train.label_names))
```

## 全組み合わせ比較

```bash
python main.py
```

64 通り (4×4×4) を評価し、精度順にソートして出力する。

## 精度を最優先する設計選択

| 箇所 | 採用した工夫 | 理由 |
|------|-------------|------|
| HOG | L2-Hys 正規化 (clip→再正規化) | 局所コントラスト変化に頑健 |
| Softmax | log-sum-exp + L2 正則化 + 早期停止 | overflow 回避と過学習抑制 |
| SVM | Pegasos の平均化 SGD + 学習率 1/(λt) | 平均化で汎化性能向上 |
| MLP | He 初期化 + Adam | ReLU と相性が良く収束が速く安定 |
| kNN | コサイン距離 + 距離重み付き投票 | 高次元特徴で L2 距離より精度が高い |
| 全分類器 | フルバッチ最適化 / 平均化で安定収束 | 速度より精度を優先 |

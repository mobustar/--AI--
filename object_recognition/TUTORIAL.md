# 物体認識 完全チュートリアル

> **このディレクトリで学べること**  
> 画像から「何が写っているか」を当てる全工程を数学レベルで理解する。  
> 前処理 → 特徴抽出 → 分類 の 3 ステップがなぜこの順序で必要なのかを体で覚える。

---

## 全体マップ

```
画像配列 (N, H, W) — 各画素は 0.0〜1.0 の float
    │
    ▼  [Step 1] preprocess.py
正規化済み画像  (N, H, W)
    │
    ▼  [Step 2] feature.py
特徴ベクトル   (N, F)  ← HOG/LBP/DCT/EdgeHist
    │
    ▼  [Step 3] classifier.py
予測ラベル     (N,)   ← "circle" / "square" / ...
```

---

## Step 1 — 画像前処理  `preprocess.py`

### なぜ前処理が必要か

生の画像は「撮影環境」「照明」「センサの違い」で画素値がバラバラになる。  
前処理で値の**スケールを揃える**ことで後段モデルの収束が安定する。

### 1-1. MinMax 正規化

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$$

最も素直な方法。ただし外れ値の強い影響を受ける。

### 1-2. 標準化 (StandardScaler) ← 推奨

$$x' = \frac{x - \mu}{\sigma}$$

> $\mu$: 訓練画像全体の平均  $\sigma$: 標準偏差

出力は平均 0 / 標準偏差 1 の分布になる。  
- SVM・ロジスティック回帰は特に標準化の恩恵を受けやすい  
- 訓練データの $\mu, \sigma$ をテストにも適用する（データリーク防止）

### 1-3. ヒストグラム均等化

累積分布関数 (CDF) を使って輝度の偏りを補正する。

$$\text{eq}(i) = \text{CDF}(i) = \frac{1}{N} \sum_{j=0}^{i} h(j)$$

> $h(j)$: 輝度値 $j$ の画素数  $N$: 全画素数

照明が暗くつぶれた画像でも形状を識別しやすくする効果がある。

---

## Step 2 — 特徴抽出  `feature.py`

これが認識精度を最も左右するステップ。「何を特徴とするか」でモデルの賢さが変わる。

### 精度ランキング (図形認識タスク)

```
HOG > EdgeHist > DCT > RawPixels > LBP
```

---

### 2-1. Raw Pixels (ベースライン)

画素値をそのまま flatten するだけ。 $H \times W$ 次元のベクトルになる。

```
24×24 画像 → 576 次元
```

シンプルだが、照明変化・位置ずれ・ノイズに弱い。

---

### 2-2. HOG — Histogram of Oriented Gradients ★最重要

**Dalal & Triggs (2005)** が人物検出のために提案した特徴量。形状に非常に強い。

#### 理論: なぜ勾配が形状を捉えるか

物体の「形」は**輝度の急激な変化 (エッジ)** として現れる。  
HOG はエッジの**向きと強さの分布**を記述する。

#### 数式: 4 ステップ

**Step 1: 勾配を計算する (中央差分)**

$$g_y[i,j] = I[i+1,j] - I[i-1,j]$$
$$g_x[i,j] = I[i,j+1] - I[i,j-1]$$

**Step 2: 強度と方向を計算する**

$$m[i,j] = \sqrt{g_x^2 + g_y^2} \quad \text{(勾配の大きさ)}$$

$$\theta[i,j] = \arctan\left(\frac{g_y}{g_x}\right) \mod \pi \quad \text{(符号なし: 0〜π)}$$

> $\theta$ を 0〜π に折りたたむ理由: エッジは 2 方向に現れるが意味は同じ  
> (左から右のエッジ = 右から左のエッジ → 同じビンに入れる)

**Step 3: セル内でヒストグラムを作る**

画像を cell_size×cell_size のセルに分割。  
各セルの画素について、`magnitude` を重みとして `θ` のビンに加算:

```
ビン 0 (0°)    ビン 1 (20°)   ...  ビン 8 (160°)
  ↑              ↑                    ↑
Σ m[i,j]       Σ m[i,j]      ...   Σ m[i,j]
(θ≈0の画素)    (θ≈20の画素)        (θ≈160の画素)
```

→ 各セルが **n_bins=9** 次元のヒストグラムになる

**Step 4: ブロック正規化 (L2-Hys)**

block_size×block_size のセルをまとめて L2 正規化:

$$v' = \frac{v}{\sqrt{\|v\|^2 + \varepsilon^2}}$$

その後 0.2 でクリッピングして再正規化 (Hys = Hysteresis):

$$v'' = \min(v', 0.2), \quad v''' = \frac{v''}{\sqrt{\|v''\|^2 + \varepsilon^2}}$$

照明変化でスケールが変わっても正規化で吸収できる。

#### 実装との対応

```python
# feature.py の HOG.extract() より
gy, gx = self._gradients(img)
mag = np.sqrt(gx ** 2 + gy ** 2)
ang = np.arctan2(gy, gx) % np.pi    # 符号なし方向
hists = self._cell_histograms(mag, ang)
feat = self._block_normalize(hists)
```

#### 特徴次元数の計算

```
画像: 24×24 / cell_size=4 / block_size=2 / n_bins=9

セル数: 6×6 = 36
ブロック数 (重複あり): (6-2+1) × (6-2+1) = 5×5 = 25
ブロックあたり: 2×2×9 = 36 次元
HOG 次元数: 25 × 36 = 900
```

---

### 2-3. LBP — Local Binary Pattern

テクスチャ（質感）の記述に特化した手法。

#### 数式

各画素について 8 近傍を時計回りに比較し、中心値以上なら 1、未満なら 0 として 8 bit パターンを生成:

$$\text{LBP}[i,j] = \sum_{k=0}^{7} s(I_k - I_c) \cdot 2^k$$

> $I_c$: 中心画素値  $I_k$: k 番目の近傍画素値  
> $s(x) = 1$ if $x \geq 0$, else $s(x) = 0$

画像全体の LBP 値の**ヒストグラム** (0〜255 の 256 ビン) が特徴ベクトルになる。

図形認識より **材質・テクスチャ識別** に向いているため、このタスクでは精度が低め。

---

### 2-4. EdgeHist — エッジ方向ヒストグラム

HOG の簡易版。Sobel フィルタでエッジを検出し、向きの分布をヒストグラムにする。

#### Sobel フィルタ (畳み込みカーネル)

$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

$$g_x = I * S_x, \quad g_y = I * S_y \quad (\text{* は畳み込み})$$

HOG との違い: セル分割とブロック正規化を行わない → 位置情報が失われる代わりに高速で安定。

---

### 2-5. DCT — 離散コサイン変換

JPEG 圧縮と同じ変換。画像の「周波数成分」に変換し、低周波成分を特徴にする。

#### 数式 (DCT-II)

$$F[k_y, k_x] = \sum_{n_y=0}^{H-1} \sum_{n_x=0}^{W-1} I[n_y, n_x] \cdot d_H[k_y, n_y] \cdot d_W[k_x, n_x]$$

$$d[k, n] = \sqrt{\frac{2}{N}} \cos\left(\frac{\pi k (2n+1)}{2N}\right), \quad d[0, n] = \sqrt{\frac{1}{N}}$$

**ジグザグスキャン**: 低周波成分 (左上) から優先的に n 個を取り出す。

```
DC成分(直流)
↓
[F00  F01  F02  ...]
[F10  F11  F12  ...]   ジグザグ: F00→F01→F10→F20→F11→F02...
[F20  F21  F22  ...]
...
```

→ 少ない係数で画像の主要情報を表現できる。形状より明暗・テクスチャ向け。

---

## Step 3 — 分類器  `classifier.py`

### 3-1. KNN (k近傍法) + コサイン距離

最もシンプルな分類器。「似ている学習例の多数決」で予測する。

$$\text{cosine\_dist}(a, b) = 1 - \frac{a \cdot b}{\|a\| \cdot \|b\|}$$

ユークリッド距離との違い:
- ユークリッド: ベクトルの大きさ（長さ）を考慮
- コサイン: ベクトルの方向だけを比較 → HOG/LBP のような高次元スパース特徴に向く

---

### 3-2. Softmax 回帰 (多項ロジスティック回帰)

二値のロジスティック回帰を多クラスに拡張。

$$P(y=c \mid x) = \frac{\exp(w_c \cdot x + b_c)}{\sum_{k=1}^{K} \exp(w_k \cdot x + b_k)}$$

#### log-sum-exp トリック (数値安定化)

分母の exp が overflow するのを防ぐ:

$$\text{softmax}(z_c) = \frac{e^{z_c - z_{\max}}}{\sum_k e^{z_k - z_{\max}}}$$

$z_{\max}$ を引いても確率値は変わらないが、指数が 0 以下になり overflow しない。

#### 損失と勾配

$$L = -\frac{1}{N} \sum_i \log P(y_i \mid x_i) + \frac{\lambda}{2} \|W\|^2$$

$$\frac{\partial L}{\partial W} = \frac{1}{N} X^\top (P - Y) + \lambda W$$

> $Y$: one-hot 行列  $P - Y$: 予測確率と正解の誤差

---

### 3-3. OvR Linear SVM (One-vs-Rest)

K クラス問題を K 個の二値 SVM に分解する。

```
K=3 クラス (A, B, C) のとき:
  SVM_A: "A か A 以外か"
  SVM_B: "B か B 以外か"
  SVM_C: "C か C 以外か"

予測: decision_function が最大のクラスを選ぶ
```

各二値 SVM は Pegasos 風 SGD で学習 (学習率 $\eta_t = 1/(\lambda t)$)。

---

### 3-4. MLP (2 層ニューラルネット)

$$X \xrightarrow{\text{Linear}} Z_1 \xrightarrow{\text{ReLU}} H_1 \xrightarrow{\text{Linear}} Z_2 \xrightarrow{\text{Softmax}} P$$

#### He 初期化 (ReLU 前提)

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

> ReLU は半分の入力をゼロにするため、分散を 2 倍にして勾配消失を防ぐ

#### Adam オプティマイザ

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一次モーメント)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二次モーメント)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(バイアス補正)}$$
$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

> $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$ がデフォルト  
> - $\hat{m}$: 勾配の方向をなめらかに追跡  
> - $\hat{v}$: 各次元ごとに学習率を自動調整 → 次元によって重みの更新速度が変わる

---

### 3-5. Gaussian Naive Bayes

各特徴がクラスごとに**ガウス分布**に従うと仮定:

$$P(x_j \mid y=c) = \frac{1}{\sqrt{2\pi \sigma_{cj}^2}} \exp\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$

対数事後確率 (クラスごとに計算して argmax):

$$\log P(y=c \mid x) \propto \log P(c) - \frac{1}{2} \sum_j \left[\log(2\pi\sigma_{cj}^2) + \frac{(x_j - \mu_{cj})^2}{\sigma_{cj}^2}\right]$$

学習: クラスごとに $\mu_{cj}$ (平均) と $\sigma_{cj}^2$ (分散) を計算するだけ。超高速。

---

### 3-6. LDA — 線形判別分析

クラスが**同じ分散行列**を共有するガウス分布に従うと仮定し、線形判別関数を導く。

$$\delta_c(x) = x^\top \Sigma_W^{-1} \mu_c - \frac{1}{2} \mu_c^\top \Sigma_W^{-1} \mu_c + \log P(c)$$

$$\hat{y} = \arg\max_c\ \delta_c(x)$$

> $\Sigma_W$: クラス内散布行列 $= \sum_c \sum_{x \in C_c} (x - \mu_c)(x - \mu_c)^\top$  
> $\mu_c$: クラス $c$ の平均ベクトル

GNB との違い:
- GNB: 各クラスが異なる分散を持つ (特徴間の相関を無視)
- LDA: 全クラスが同じ $\Sigma_W$ を共有 (クラス間の関係を考慮)

---

## Step 4 — 実行してみる

### 全組み合わせ比較

```bash
cd object_recognition
python main.py
```

```
preproc   feature   classifier     accuracy
------------------------------------------
standard  edge      softmax           0.942
standard  hog       mlp               0.935
standard  hog       lda               0.921
...
```

### 1 つの構成だけ試す

```python
from data       import make_shapes, train_test_split
from preprocess import get_preprocessor
from feature    import get_feature
from classifier import get_classifier
from pipeline   import ObjectRecognitionPipeline

ds = make_shapes(n_per_class=200, size=24, noise=0.08, seed=0)
train, test = train_test_split(ds, ratio=0.8, seed=1)

pipe = ObjectRecognitionPipeline(
    preprocessor = get_preprocessor("standard"),
    feature      = get_feature("hog"),
    classifier   = get_classifier("softmax"),
)
pipe.fit(train.images, train.labels)
preds = pipe.predict(test.images)
```

---

## Step 5 — 実験課題

### 課題 1: HOG のセルサイズを変える

```python
feat = get_feature("hog", cell_size=2)  # 細かく分割
feat = get_feature("hog", cell_size=8)  # 粗く分割
```

小さいセルサイズ → 位置情報が細かい / ノイズに弱い  
大きいセルサイズ → 位置に頑健 / 細部を失う

### 課題 2: ノイズの影響を調べる

```python
ds_clean = make_shapes(noise=0.0)   # ノイズなし
ds_noisy = make_shapes(noise=0.2)   # ノイズ多め
```

どの前処理+特徴の組み合わせがノイズに最も頑健か比較する。

### 課題 3: HOG の勾配を可視化する

```python
import numpy as np
from feature import HOG

hog = HOG()
img = ds.images[0]
gy, gx = hog._gradients(img)
mag = np.sqrt(gx**2 + gy**2)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3)
axes[0].imshow(img, cmap='gray')
axes[1].imshow(np.abs(gx), cmap='hot')
axes[2].imshow(mag, cmap='hot')
plt.show()
```

### 課題 4: DCT 係数の数を変える

```python
feat_few  = get_feature("dct", n_components=8)   # 低周波のみ
feat_many = get_feature("dct", n_components=64)  # 多めの係数
```

係数が多すぎると高周波ノイズも取り込む → 精度が下がる場合がある。

---

## まとめ

| コンポーネント | ファイル | 核心概念 |
|--------------|---------|---------|
| MinMax/Standard | `preprocess.py` | スケール統一 / 標準化 |
| HistEQ | `preprocess.py` | CDF による輝度分布の均等化 |
| HOG | `feature.py` | 勾配の方向ヒストグラム + L2-Hys 正規化 |
| LBP | `feature.py` | 近傍比較による 8bit パターン分布 |
| DCT | `feature.py` | 周波数変換 + ジグザグ低周波抽出 |
| KNN | `classifier.py` | コサイン距離で k 近傍多数決 |
| Softmax | `classifier.py` | 多クラス確率最大化 (log-sum-exp 安定化) |
| MLP | `classifier.py` | ReLU + Adam + He 初期化 |
| GNB | `classifier.py` | ガウス分布の対数事後確率 |
| LDA | `classifier.py` | クラス内散布行列の逆行列による線形判別 |

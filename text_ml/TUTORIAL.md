# テキスト分類 完全チュートリアル

> **このディレクトリで学べること**  
> 「文書を機械に読ませる」ための全工程を、数式レベルで理解する。  
> トークナイズ → ベクトル化 → 分類 の 3 ステップがどう繋がるかを手を動かしながら習得する。

---

## 全体マップ

```
テキスト (生文字列)
    │
    ▼  [Step 1] tokenizer.py
トークン列  ["美味", "味し", "しい", ...]
    │
    ▼  [Step 2] vectorizer.py
数値ベクトル  [0.12, 0.0, 0.53, ...]   shape: (N文書, V語彙)
    │
    ▼  [Step 3] classifier.py
予測ラベル   "スポーツ" / "政治" / ...
```

---

## Step 1 — トークナイズ  `tokenizer.py`

### 目的

「ねこがさかなをたべた」という文字列を、機械が比較できる**単位(トークン)**に分解する。  
英語は空白で自然に分かれるが、日本語は空白がないため工夫が必要。

### 手法一覧

| 名前 | コード上の名前 | 説明 | 日本語適性 |
|------|---------------|------|-----------|
| 空白区切り | `whitespace` | split() するだけ | ✗ |
| 文字 N-gram | `char_ngram` | 連続 n 文字を抽出 | ◎ |
| 正規表現 | `regex` | 文字種の塊を抽出 | ○ |
| 1 文字 | `char` | 1 文字ずつ | △ |

### 文字 N-gram の仕組み (最重要)

入力テキスト `"美味しいラーメン"` に **n=2** を適用すると:

```
"美味しいラーメン"
→ ["美味", "味し", "しい", "いラ", "ラー", "ーメ", "メン"]
```

n=2 と n=3 の両方を使う (`ns=(2,3)`) と更に多様な特徴が取れる。  
なぜ精度が高いか — **部分文字列の共通性**を捉えられるから:
- 「美味しい」と「美味」は "美味" というトークンを共有 → 類似文書として識別しやすい

### 実際に動かしてみる

```python
from tokenizer import get_tokenizer

tok = get_tokenizer("char_ngram", ns=(2, 3))
print(tok.tokenize("機械学習はとても面白い"))
# → ['機械', '械学', '学習', '習は', 'はと', 'とて', 'ても', '面白', '白い',
#    '機械学', '械学習', '学習は', ...]
```

---

## Step 2 — ベクトル化  `vectorizer.py`

### なぜベクトルが必要か

分類器は数値しか扱えない。「単語の出現パターン」を数値の配列に変換する。

---

### 2-1. Bag of Words (Count Vectorizer)

最もシンプルな手法。各単語の**出現回数**を次元として持つ。

```
語彙 = {"機械": 0, "学習": 1, "楽しい": 2, "難しい": 3, ...}

文書A "機械学習は楽しい" → [1, 1, 1, 0, ...]
文書B "機械は難しい"     → [1, 0, 0, 1, ...]
```

**問題点**: 「は」「の」のような助詞が高頻度で意味を持たない次元を汚染する。

---

### 2-2. TF-IDF (最重要・最も精度が出る)

「よく出るが珍しい語ほど重要」という直感を数式化した手法。

#### 数式

$$\text{TF}(t, d) = \text{文書 } d \text{ における単語 } t \text{ の出現回数}$$

> `sublinear_tf=True` の場合: $\text{TF}(t,d) = 1 + \log(\text{count}(t,d))$  
> → 大量出現による過大評価を抑える

$$\text{IDF}(t) = \log\frac{1 + N}{1 + \text{DF}(t)} + 1$$

> $N$: 全文書数  $\text{DF}(t)$: 単語 $t$ を含む文書数  
> → 多くの文書に出る語ほど小さくなる（"の", "は" などの重みを下げる）

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

最後に **L2 正規化** (コサイン距離のため):

$$\vec{x}_d = \frac{\vec{x}_d}{\|\vec{x}_d\|_2}$$

#### 直感的な例

| 単語 | 文書A での TF | DF (全体) | IDF | TF-IDF |
|------|-------------|----------|-----|--------|
| 機械学習 | 3 | 10 | 高い | **大きい** |
| の | 5 | 1000 | 低い | **小さい** |

→ 特徴的な語が強調され、ノイズが抑制される。

---

### 2-3. BM25 (TF-IDF の改良版)

TF-IDF の 2 つの弱点を改善:
1. **TF の飽和問題** — 同じ語が 10 回出ても 3 回出ても意味は同じくらい
2. **文書長の不公平** — 長い文書は短い文書より自然に TF が大きい

#### 数式

$$\text{IDF}_{\text{BM25}}(t) = \log\left(\frac{N - \text{DF}(t) + 0.5}{\text{DF}(t) + 0.5} + 1\right)$$

$$\text{TF}_{\text{BM25}}(t, d) = \frac{\text{TF}(t,d) \cdot (k_1 + 1)}{\text{TF}(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

> $k_1 = 1.5$: TF の飽和速度。大きいほど飽和が遅い  
> $b = 0.75$: 文書長正規化の強さ。0 なら正規化なし、1 なら完全正規化  
> $|d|$: 文書のトークン数、$\text{avgdl}$: 平均文書長

**グラフで理解する飽和効果**:

```
スコア
↑
k1+1 ─────────────────────── BM25 (飽和する)
      /
     /    _____________________  TF (線形に増え続ける)
    /  _/
   / /
──────────────────────────→ TF の値
```

---

### 2-4. Hashing Vectorizer

語彙を保存しない代わりに、各トークンをハッシュ関数で固定次元に写像する。

```
"機械" → hash("機械") % n_features → 次元 4217 に +1
"学習" → hash("学習") % n_features → 次元 891  に +1
```

大規模データやオンライン学習向け。ただし語彙の確認ができないためデバッグは難しい。

---

## Step 3 — 分類器  `classifier.py`

### 3-1. 多項ナイーブベイズ (MultinomialNB)

#### 数式

$$P(c \mid d) \propto P(c) \cdot \prod_i P(t_i \mid c)^{x_i}$$

> $P(c)$: クラス事前確率  $P(t_i \mid c)$: クラス $c$ での単語 $t_i$ の生成確率  
> $x_i$: 文書 $d$ における単語 $t_i$ の出現回数

対数に変換して積→和にする（数値安定化）:

$$\log P(c \mid d) = \log P(c) + \sum_i x_i \cdot \log P(t_i \mid c)$$

#### Laplace スムージング (ゼロ除算回避)

学習データに出ていない語の確率がゼロになるのを防ぐ:

$$P(t_i \mid c) = \frac{\text{count}(t_i, c) + \alpha}{\sum_j \text{count}(t_j, c) + \alpha \cdot |V|}$$

$\alpha=1$ がデフォルト。分子に $\alpha$ を足すことで、未見語でも確率 > 0 になる。

#### 実装との対応

```python
# classifier.py の MultinomialNB.fit() より
count = X_c.sum(axis=0) + self.alpha      # Laplace smoothing
self.feature_log_prob_[i] = np.log(count) - np.log(count.sum())  # log P(t|c)
```

---

### 3-2. ロジスティック回帰 (LogisticRegression)

#### モデル

$$P(y=1 \mid x) = \sigma(w \cdot x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### 損失関数 (交差エントロピー + L2 正則化)

$$L(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[y_i \log \sigma(z_i) + (1-y_i) \log(1-\sigma(z_i))\right] + \frac{\lambda}{2} \|w\|^2$$

#### 勾配降下法による更新

$$\frac{\partial L}{\partial w} = \frac{1}{N} X^\top (\sigma(z) - y) + \lambda w$$

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

> $\eta$: 学習率  $\lambda$: 正則化係数（大きいほど重みが小さくなり過学習を防ぐ）

#### シグモイド関数の数値安定化

$z$ が大きいとき $e^{-z} \approx 0$ で問題なし。$z$ が小さいとき $e^{-z}$ が overflow する。

```python
# classifier.py の _sigmoid() より (条件分岐で安定化)
if z >= 0:
    return 1.0 / (1.0 + exp(-z))      # 安全な経路
else:
    ez = exp(z)
    return ez / (1.0 + ez)            # 等価だが overflow しない
```

---

### 3-3. 線形 SVM (LinearSVM)

#### モデルと損失関数

$$L(w, b) = \frac{1}{N} \sum_i \max(0,\ 1 - y_i (w \cdot x_i + b)) + \frac{\lambda}{2} \|w\|^2$$

> $y_i \in \{-1, +1\}$ に変換  
> $\max(0, \cdot)$ が**ヒンジ損失** — マージン内にある点にのみペナルティ

#### なぜ高次元テキストに強いか

- 語彙数 = 特徴次元 → 数千〜数万次元が普通
- L2 正則化 + ヒンジ損失 → スパース高次元でも汎化が良い
- ロジスティック回帰と精度は同等だが、マージン最大化により決定境界がより頑健

#### Pegasos 風 SGD + 平均化

```python
# 学習率を t に反比例して減衰
eta = 1.0 / (self.reg * t)
# ヒンジ損失の勾配更新
if margin < 1:
    w = (1 - eta * reg) * w + eta * y_pm[i] * x[i]
# 全ステップの平均で汎化向上
w_final = w_avg / t
```

---

### 3-4. Ridge Classifier (閉形式解)

$$\hat{\theta} = (X^\top X + \alpha I)^{-1} X^\top y$$

反復計算なしで一発解が出る。小〜中規模データで最速の学習。

双対形式 ($n < d$ の場合に効率的):

$$K = \tilde{X}\tilde{X}^\top + \alpha I \in \mathbb{R}^{n \times n}, \quad
  \theta = \tilde{X}^\top K^{-1} y$$

> $\tilde{X}$ はバイアス項を追加した行列。  
> 特徴次元より文書数が少ない場合に $n \times n$ の逆行列計算で済む。

---

## Step 4 — パイプラインをつなげる  `pipeline.py`

```
train.texts ──Tokenizer──▶ tokens ──Vectorizer──▶ X_train ──Classifier──▶ モデル
test.texts  ──Tokenizer──▶ tokens ──Vectorizer──▶ X_test  ──predict()──▶ preds
```

**重要**: `Vectorizer.fit()` は**訓練データのみ**で実行し、テストデータには `transform()` のみを使う。  
（テストの語彙情報を訓練に漏洩させない = データリーク防止）

---

## Step 5 — 実行してみる

### 全組み合わせ比較

```bash
cd text_ml
python main.py
```

出力例:
```
tokenizer   vectorizer  classifier     accuracy
----------------------------------------------
char_ngram  tfidf       svm               0.923
char_ngram  tfidf       logreg            0.916
regex       bm25        svm               0.908
...
```

### 推奨構成のみ試す

```python
from data       import load_dataset
from tokenizer  import get_tokenizer
from vectorizer import get_vectorizer
from classifier import get_classifier
from pipeline   import TextClassificationPipeline

train, test = load_dataset(seed=42)

pipe = TextClassificationPipeline(
    tokenizer  = get_tokenizer("char_ngram", ns=(2, 3)),
    vectorizer = get_vectorizer("tfidf"),
    classifier = get_classifier("svm"),
)
pipe.fit(train.texts, train.labels)
preds = pipe.predict(test.texts)
```

---

## Step 6 — 実験課題

### 課題 1: IDF の効果を確認する

```python
# TF-IDF と BoW を比較する
from vectorizer import get_vectorizer
from tokenizer  import get_tokenizer

tok = get_tokenizer("char_ngram")
train, test = __import__("data").load_dataset()
tokens = [tok.tokenize(t) for t in train.texts]

vec_count = get_vectorizer("count").fit(tokens)
vec_tfidf = get_vectorizer("tfidf").fit(tokens)

# 同じ文書でもベクトルの中身が違う → IDF の影響を観察
x_count = vec_count.transform(tokens[:1])
x_tfidf = vec_tfidf.transform(tokens[:1])
```

### 課題 2: n の大きさと精度の関係

`ns=(1,)` `ns=(2,)` `ns=(3,)` `ns=(2,3)` で精度を比較する。  
なぜ 2-gram と 3-gram の組み合わせが良いか考察してみよう。

### 課題 3: 正則化の強さを変える

`LogisticRegression(reg=1e-1)` vs `reg=1e-3` vs `reg=1e-5`  
- 正則化が強い → 重みが小さく / 弱い → 過学習の可能性

### 課題 4: 未知語を含む文書を予測してみる

```python
texts = ["全く関係ない新語を含む文章テスト"]
preds = pipe.predict(texts)
```

HashingVectorizer と TfIdfVectorizer でどう振る舞いが変わるかを確認する。

---

## まとめ

| コンポーネント | ファイル | 核心数式 |
|--------------|---------|---------|
| トークナイズ | `tokenizer.py` | 文字列 → トークン列 |
| BoW | `vectorizer.py` | $x_i = \text{count}(t_i, d)$ |
| TF-IDF | `vectorizer.py` | $x_i = \text{TF}(t_i,d) \cdot \text{IDF}(t_i)$ (L2 正規化) |
| BM25 | `vectorizer.py` | TF 飽和 + 文書長正規化 |
| NaiveBayes | `classifier.py` | $\log P(c\|d) = \log P(c) + \sum x_i \log P(t_i\|c)$ |
| LogReg | `classifier.py` | $\sigma(Xw+b)$ のクロスエントロピー最小化 |
| SVM | `classifier.py` | ヒンジ損失 + マージン最大化 |
| Ridge | `classifier.py` | $(X^\top X + \alpha I)^{-1} X^\top y$ |

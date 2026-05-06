# text_ml

NumPy **のみ**で実装したテキスト分類のフルパイプライン。
日本語レビュー（肯定/否定）を、トークナイザ → ベクトル化 → 分類器 の組み合わせで分類する。

---

## パイプライン概要

```
入力テキスト（日本語文字列）
        │
        ▼
┌────────────────────┐
│     Tokenizer      │  テキスト → トークン列 (4 種)
│ whitespace /       │
│ char_ngram /       │
│ regex / char       │
└─────────┬──────────┘
          │  ["美味", "味し", "しい", ...]
          ▼
┌────────────────────┐
│     Vectorizer     │  トークン列 → 数値ベクトル (3 種)
│ count / tfidf /    │
│ hashing            │
└─────────┬──────────┘
          │  [0.0, 0.52, 0.0, 0.31, ...]  (固定長ベクトル)
          ▼
┌────────────────────┐
│     Classifier     │  ベクトル → クラス予測 (4 種)
│ nb / logreg /      │
│ svm / knn          │
└─────────┬──────────┘
          │
          ▼
     予測クラス
    肯定 / 否定
```

各コンポーネントは**戦略パターン**で実装されており、名前を指定するだけで差し替えられる。

---

## ディレクトリ構成

```
text_ml/
├── data.py          # 日本語レビューのトイデータセット（肯定/否定 各 30 件）
├── tokenizer.py     # トークナイザ 4 種
├── vectorizer.py    # ベクトル化器 3 種
├── classifier.py    # 分類器 4 種
├── metrics.py       # 評価指標（Precision / Recall / F1 等）
├── pipeline.py      # 上記を統合する TextClassificationPipeline
└── main.py          # 全 48 通り比較 + 推奨構成の詳細評価
```

---

## データセット（data.py）

| 項目 | 値 |
|------|-----|
| クラス | 否定 (0) / 肯定 (1) |
| データ数 | 各クラス 30 件（計 60 件）|
| 内容 | 日本語レストランレビュー |
| 訓練/評価分割 | 80:20（seed=42）|

```
肯定: "この料理はとても美味しかったです"
否定: "料理が冷めていて美味しくなかった"
```

---

## Tokenizer（tokenizer.py）— 4 種

テキストを機械学習が扱える「トークン列」に分解する。

**共通前処理**（全トークナイザに適用）:

```
normalize(text) = lowercase(NFKC(text))

NFKC 正規化の例:
  "Ａ"（全角）→ "a"（半角小文字）
  "ｱｲｳ"（半角カナ）→ "アイウ"（全角カナ）
```

| 名前 | クラス | 日本語精度 |
|------|--------|:--------:|
| `whitespace` | `WhitespaceTokenizer` | △ |
| `char_ngram` | `CharNGramTokenizer` | ◎ |
| `regex` | `RegexTokenizer` | ○ |
| `char` | `CharTokenizer` | △ |

---

### WhitespaceTokenizer（空白区切り）

**定義**:

```
tokenize(text) = normalize(text).split()

split() は連続する空白・タブ・改行を 1 つの区切りとして扱う。
```

**いつ使うか**: 英語など空白が単語境界の言語。日本語には不適。

---

### CharNGramTokenizer（文字 N-gram）

**定義**:

```
【前処理】
  text' = re.sub(r"\s+", "", normalize(text))   （空白を除去）

【N-gram 抽出】
  ns = (n_1, n_2, ...) （使用する N の集合、デフォルト (2, 3)）

  tokens = []
  for n in ns:
    if len(text') < n: continue
    for i in 0, 1, ..., len(text')−n:
      tokens.append(text'[i : i+n])

例（text'="美味しい", ns=(2,3)）:
  n=2: ["美味", "味し", "しい"]
  n=3: ["美味し", "味しい"]
  → ["美味", "味し", "しい", "美味し", "味しい"]
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `ns` | `(2, 3)` | 使用する N の集合 |

**利点**: 形態素解析不要。日本語に有効。**推奨**。

---

### RegexTokenizer（正規表現トークナイザ）

**定義**:

```
PATTERN = re.compile(
    r"[一-鿿]+"   |   # 漢字（CJK Unified Ideographs）
    r"[぀-ゟ]+"   |   # ひらがな（Hiragana block）
    r"[゠-ヿ]+"   |   # カタカナ（Katakana block）
    r"[a-zA-Z0-9]+"           # ASCII 英数字
)

tokenize(text) = PATTERN.findall(normalize(text))

文字種の境界を単語境界と見なして分割する。
```

**利点**: 形態素解析に近い分割を専用ライブラリなしで実現できる。

---

### CharTokenizer（文字単位）

**定義**:

```
tokenize(text) = [c for c in normalize(text) if not c.isspace()]

空白文字（スペース・タブ・改行）を除いた各文字を 1 トークンとして返す。
```

**利点**: 語彙が最小（文字数のみ）。未知語が存在しない。
**欠点**: 1 文字では意味を捉えにくい。

> **ファクトリ関数**: `get_tokenizer("char_ngram", ns=(2, 3))`

---

## Vectorizer（vectorizer.py）— 3 種

トークン列を固定長の数値ベクトルに変換する。

| 名前 | クラス | 精度傾向 |
|------|--------|:--------:|
| `count` | `CountVectorizer` | ○ |
| `tfidf` | `TfIdfVectorizer` | ◎ **最高精度** |
| `hashing` | `HashingVectorizer` | △ 未知語に強い |

---

### CountVectorizer（Bag of Words）

**定義**:

```
【学習】
  V = {t : DF(t) ≥ min_df  かつ  DF(t) ≤ max_df × N}

  DF(t) = |{d ∈ D_train : t ∈ d}|   （t が出現した文書数）
  N     = |D_train|                  （全文書数）

  語彙辞書: vocab = {t: idx}   idx = 0, 1, ..., |V|−1

【変換】
  x_d ∈ ℝ^{|V|}

  x_d[vocab[t]] = |{token ∈ d : token = t}|   （出現回数）
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `min_df` | 1 | 語彙に含める最小文書頻度 |
| `max_df` | 1.0 | 語彙に含める最大文書頻度（比率）|

---

### TfIdfVectorizer（TF-IDF）

**定義**:

```
【学習】
  同じ語彙フィルタ（min_df, max_df）で V と vocab を構築する。

  IDF(t) = log( (1 + N) / (1 + DF(t)) ) + 1      （smooth IDF）
  → DF=N（全文書に出現）のとき IDF = log(1) + 1 = 1.0  （最小値）
  → DF=1 のとき IDF = log((1+N)/2) + 1            （大きい値）

【変換】
  TF'(t, d) = 1 + log(count(t, d))   if count(t, d) > 0   （sublinear TF）
            = 0                       otherwise

  z_d[vocab[t]] = TF'(t, d) × IDF(t)

  L2 正規化:
    x_d = z_d / ‖z_d‖₂     （‖z_d‖₂ = 0 のとき分母を 1 とする）
```

TF-IDF の値の大小:

```
"です"（DF=58/60）→ IDF ≈ 1.03  （低い: ほぼ全文書に出現）
"美味"（DF=15/60）→ IDF ≈ 2.34  （中程度）
"絶品"（DF=3/60） → IDF ≈ 3.72  （高い: 稀な語 = 識別力が強い）
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `min_df` | 1 | 最小文書頻度 |
| `sublinear_tf` | `True` | `1+log(TF)` を使うかどうか |

**いつ使うか**: **一般的なテキスト分類で最推奨**。

---

### HashingVectorizer（ハッシュ写像）

**定義**:

```
語彙辞書を作らず、ハッシュ関数で直接インデックスに変換する。

【ハッシュ関数】
  h(t) = int.from_bytes(MD5(t.encode("utf-8"))[:8], "little")

【変換】
  x_d ∈ ℝ^{n_features}    （初期値 0）

  for t in tokens_d:
    idx  = h(t) mod n_features
    sign = +1  if (h(t) >> 1) mod 2 == 0  else −1   （符号トリック）
    x_d[idx] += sign

符号トリックの効果:
  衝突 (t₁ ≠ t₂ だが idx(t₁) = idx(t₂)) が発生しても
  sign が +1/−1 とランダムなので期待値 E[sign(t₁)×sign(t₂)] = 0 → 部分的に相殺
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `n_features` | 4096 | ハッシュ空間の次元数 |
| `signed` | `True` | 符号トリックを使うかどうか |

**注意**: Naive Bayes は非負値を前提とするため、符号トリック（負値が生じる）との相性が悪い。

> **ファクトリ関数**: `get_vectorizer("tfidf", min_df=1)`

---

## Classifier（classifier.py）— 4 種

ベクトル化された文書を 2 クラス（肯定/否定）に分類する。

---

### MultinomialNB（多項ナイーブベイズ）

**定義**:

```
訓練データ: {(x_i, y_i)}   x_i ∈ ℝ^d（非負整数カウント前提）、y_i ∈ {0, 1}

【学習】
  クラス事前確率:
    log P(c) = log(N_c / N)     N_c = クラス c のサンプル数

  クラス条件付き特徴確率（Laplace smoothing）:
    count(t, c) = Σ_{i: y_i=c} x_i[t]   （クラス c での特徴 t の総出現数）

    log P(t | c) = log( count(t, c) + α )
                − log( Σ_{t'} count(t', c)  +  α × |V| )

    α: Laplace smoothing 係数（デフォルト 1.0）

【予測（対数確率の和）】
  score(c | x) = log P(c)  +  Σ_t x[t] · log P(t | c)

  ŷ = argmax_{c ∈ {0,1}} score(c | x)

  負値保護: x ← max(x, 0)  （Hashing の符号トリックへの防御）
```

**利点**: 学習が非常に速い（1 パス）。小データでも高精度。
**欠点**: 条件付き独立性仮定が強い。CountVectorizer との相性が最良。

---

### LogisticRegression（ロジスティック回帰）

**定義**:

```
パラメータ: w ∈ ℝ^d, b ∈ ℝ

【シグモイド関数（数値安定版）】
  σ(z) = 1 / (1 + exp(−z))   if z ≥ 0
       = exp(z) / (1 + exp(z)) if z < 0

【モデル】
  P(y=1 | x) = σ(w · x + b)

【損失関数（交差エントロピー + L2 正則化）】
  L(w, b) = −(1/N) Σ_{i=1}^{N} [ y_i log σ(z_i) + (1−y_i) log(1−σ(z_i)) ]
            + (λ/2) ‖w‖²

  数値安定版の交差エントロピー:
    CE(y, z) = log(1+exp(z)) − y·z
             = log1p(exp(z))   if z < 0       （exp(z) は小さい）
               z + log1p(exp(−z))  if z ≥ 0  （exp(−z) は小さい）

【勾配（フルバッチ）】
  ∂L/∂w = (1/N) Xᵀ (σ(z) − y)  +  λw
  ∂L/∂b = (1/N) Σ_i (σ(z_i) − y_i)

【更新】
  w ← w − lr · ∂L/∂w
  b ← b − lr · ∂L/∂b

【早期停止】
  |L_t − L_{t-1}| < tol のとき停止

【予測】
  ŷ = 𝟙[ σ(w · x + b) ≥ 0.5 ]
    = 𝟙[ w · x + b ≥ 0 ]
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `lr` | 0.5 | 学習率 |
| `reg` | 1e-3 | L2 正則化係数 λ |
| `epochs` | 1000 | 最大エポック数 |
| `tol` | 1e-6 | 早期停止閾値 |

**利点**: 確率値を出力できる。解釈性が高い（重みの符号が特徴の向きを示す）。

---

### LinearSVM（線形 SVM）

**定義**:

```
パラメータ: w ∈ ℝ^d, b ∈ ℝ

ラベル変換: y_pm ∈ {−1, +1}
  y_pm[i] = +1  if y_i = 1（肯定）
           = −1  if y_i = 0（否定）

【損失関数（ヒンジ損失 + L2 正則化）】
  L(w, b) = (1/N) Σ_{i=1}^{N} max(0, 1 − y_pm[i] · (w · x_i + b))
            + (λ/2) ‖w‖²

【平均化 SGD（Pegasos 風）】
  各エポック内でサンプルをランダムにシャッフルして 1 件ずつ処理。

  累積ステップ数 t ← t + 1
  学習率: η_t = 1 / (λ · t)

  margin_i = y_pm[i] · (w · x_i + b)

  if margin_i < 1:          （ヒンジ損失の劣微分が非ゼロ）
    w ← (1 − η_t λ) w  +  η_t y_pm[i] x_i
    b ← b  +  η_t y_pm[i]
  else:
    w ← (1 − η_t λ) w      （正則化のみ）

  累積:
    w_avg ← w_avg + w
    b_avg ← b_avg + b

【最終パラメータ（平均化）】
  ŵ = w_avg / T,  b̂ = b_avg / T   （T = 総ステップ数 epochs × N）

【予測】
  ŷ = 𝟙[ ŵ · x + b̂ ≥ 0 ]
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `reg` | 1e-3 | 正則化係数 λ |
| `epochs` | 50 | SGD エポック数 |

**利点**: テキストのような高次元疎ベクトルで高精度。**推奨**。

---

### KNNClassifier（k 近傍法）

**定義**:

```
訓練データ: {(x_i, y_i)}_{i=1}^{N}   x_i ∈ ℝ^d, y_i ∈ {0, 1}

【L2 正規化（fit 時に事前計算）】
  x̂_i = x_i / ‖x_i‖₂     （‖x_i‖₂ = 0 のとき分母を 1 とする）

【コサイン距離】
  dist_cos(x, x_i) = 1 − (x̂ · x̂_i)
  （x̂ = x / ‖x‖₂）

  内積行列 S = X̂_test @ X̂_trainᵀ から一括計算。

【k 近傍の取得】
  N_k(x) = k 個の最近傍インデックスの集合（numpy.argpartition で高速化）

【距離重み付き予測】
  w_i = 1 / (dist_cos(x, x_i) + ε)   ε=1e-9

  score_c = Σ_{i ∈ N_k(x), y_i = c} w_i   c ∈ {0, 1}

  ŷ = argmax_{c ∈ {0,1}} score_c
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `k` | 5 | 近傍数 |
| `weights` | `"distance"` | `"uniform"` は重みなし多数決 |

**利点**: 学習不要。非線形境界を自然に表現できる。
**欠点**: 推論時に全訓練データとの距離計算が必要で遅い。

> **ファクトリ関数**: `get_classifier("svm")`

---

## 評価指標（metrics.py）

**定義**:

```
混同行列: C ∈ ℤ^{2×2}   C[i,j] = 真ラベル i を j と予測した数

TP = C[1,1],  FP = C[0,1],  FN = C[1,0],  TN = C[0,0]

Precision_c = TP_c / (TP_c + FP_c + ε)
Recall_c    = TP_c / (TP_c + FN_c + ε)
F1_c        = 2 · Precision_c · Recall_c / (Precision_c + Recall_c + ε)

Macro-F1    = (F1_0 + F1_1) / 2

Accuracy    = (TP + TN) / (TP + FP + FN + TN)
```

---

## 使い方

### 基本

```python
from data       import load_dataset
from tokenizer  import get_tokenizer
from vectorizer import get_vectorizer
from classifier import get_classifier
from pipeline   import TextClassificationPipeline
from metrics    import classification_report

train, test = load_dataset(seed=42)

pipe = TextClassificationPipeline(
    tokenizer  = get_tokenizer("char_ngram", ns=(2, 3)),
    vectorizer = get_vectorizer("tfidf"),
    classifier = get_classifier("svm"),
)
pipe.fit(train.texts, train.labels)
preds = pipe.predict(test.texts)
print(classification_report(test.labels, preds, train.label_names))
```

### 全 48 通り比較

```bash
python main.py
```

---

## 推奨構成

```
char_ngram + tfidf + svm   → 最高精度クラス

理由:
  char_ngram : 形態素解析不要で日本語を確実にトークン化
  tfidf      : 識別力の高い語を重み付け、文書長の違いを吸収
  svm        : 高次元疎ベクトルに対して線形 SVM が最も有効
```

---

## 精度向上のための設計選択

| 箇所 | 工夫 | 理由 |
|------|------|------|
| 全トークナイザ | NFKC 正規化 + 小文字化 | 全半角ゆれ・大文字小文字を吸収 |
| TF-IDF | sublinear TF + smooth IDF + L2 正規化 | 高頻度語の過大評価を抑え、文書長を等価に扱う |
| Naive Bayes | 対数計算 + Laplace smoothing + clip(0,∞) | アンダーフロー・未知語・負値混入に頑健 |
| LogReg | フルバッチ GD + L2 正則化 + 早期停止 + 数値安定 sigmoid | 収束が滑らか |
| SVM | 平均化 SGD + η=1/(λt) 減衰 | 平均化で汎化性能向上 |
| kNN | コサイン距離 + 距離重み付き投票 | テキスト疎ベクトルでは L2 より精度が高い |

---

## 拡張アイデア

| 層 | 発展的な選択肢 |
|----|--------------|
| Tokenizer | MeCab / Janome / SentencePiece / BPE |
| Vectorizer | Word2Vec / fastText / SVD（LSA）|
| Classifier | 多クラス Softmax / MLP / Transformer |
| 評価 | K-fold 交差検証 / ROC-AUC / 学習曲線 |

---

## 依存ライブラリ

```
numpy のみ（外部ライブラリ不要）
Python 3.x（unicodedata, re, hashlib は標準ライブラリ）
```

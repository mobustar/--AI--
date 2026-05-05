# text_ml — テキスト分類学習用コード集

NumPy のみで実装した、テキスト分類のフルパイプライン。
**Tokenizer × Vectorizer × Classifier** の各機能ごとに複数のアルゴリズムを用意し、
戦略パターン (Strategy Pattern) で自由に切り替えられる構造になっている。

## ディレクトリ構成

```
text_ml/
├── data.py          # 日本語レビューのトイデータ (肯定/否定 30 件ずつ)
├── tokenizer.py     # トークナイザ 4 種
├── vectorizer.py    # ベクトル化器 3 種
├── classifier.py    # 分類器 4 種
├── metrics.py       # 評価指標
├── pipeline.py      # 上記を統合する高レベル API
├── main.py          # 全組み合わせを比較するエントリポイント
└── README.md
```

## 各機能のアルゴリズム選択肢

### Tokenizer (4 種)

| 名前 | クラス | 特徴 | 想定用途 |
|------|--------|------|----------|
| `whitespace` | `WhitespaceTokenizer` | 空白で分割 | 英語など単語境界が明示的な言語 |
| `char_ngram` | `CharNGramTokenizer` | 文字 N-gram (n=2,3 推奨) | **日本語に頑健** |
| `regex`      | `RegexTokenizer` | 文字種(漢字/かな/カナ/英数)で分割 | 形態素解析器が無い環境 |
| `char`       | `CharTokenizer` | 1 文字ずつ | 語彙最小・RNN/Transformer 入力 |

### Vectorizer (3 種)

| 名前 | クラス | 数式・特徴 |
|------|--------|------------|
| `count`   | `CountVectorizer`   | 単純な出現回数 (Bag of Words) |
| `tfidf`   | `TfIdfVectorizer`   | `tf·idf`, IDF=`log((1+N)/(1+df))+1`, sublinear TF, L2 正規化 |
| `hashing` | `HashingVectorizer` | ハッシュで固定次元化 (語彙保持なし、未知語に強い) |

### Classifier (4 種)

| 名前 | クラス | アルゴリズム | 強み |
|------|--------|--------------|------|
| `nb`     | `MultinomialNB`     | 多項ナイーブベイズ + Laplace smoothing | 小データでも安定 |
| `logreg` | `LogisticRegression`| L2 正則化 + フルバッチ勾配降下 + 早期停止 | バランス良 |
| `svm`    | `LinearSVM`         | ヒンジ損失 + L2 + 平均化 SGD (Pegasos) | **高次元疎データに強い** |
| `knn`    | `KNNClassifier`     | コサイン距離 + 距離重み付き多数決 | 学習不要、解釈しやすい |

## 重要な数式

### TF-IDF
```
TF(t,d)  = 文書 d 中の語 t の出現回数
IDF(t)   = log( (1+N) / (1+DF(t)) ) + 1
TF-IDF   = (1+log(TF)) · IDF                  ← sublinear_tf=True
最後に L2 正規化: x ← x / ‖x‖₂
```

### Multinomial Naive Bayes
```
P(c|d) ∝ P(c) · ∏ᵢ P(tᵢ|c)^{xᵢ}
log P(c|d) ∝ log P(c) + Σᵢ xᵢ · log P(tᵢ|c)
P(tᵢ|c) = (count(tᵢ,c) + α) / (Σⱼ count(tⱼ,c) + α·|V|)   ← Laplace
```

### Logistic Regression (二値, L2 正則化)
```
L = -1/N Σ [y log σ(z) + (1-y) log(1-σ(z))] + 0.5λ‖w‖²
∂L/∂w = 1/N Xᵀ (σ(z) - y) + λw
```

### Linear SVM (Pegasos 風)
```
L = 1/N Σ max(0, 1 - yᵢ(w·xᵢ + b)) + 0.5λ‖w‖²
学習率: ηₜ = 1/(λt)   ← 減衰させて安定収束
```

### k-NN (コサイン距離)
```
sim(x, xᵢ) = (x · xᵢ) / (‖x‖ ‖xᵢ‖)
dist        = 1 - sim
重み付き投票: weight(i) = 1 / (dist(x,xᵢ) + ε)
```

## 使い方

```python
from data       import load_dataset
from tokenizer  import get_tokenizer
from vectorizer import get_vectorizer
from classifier import get_classifier
from pipeline   import TextClassificationPipeline
from metrics    import classification_report

train, test = load_dataset()

pipe = TextClassificationPipeline(
    tokenizer  = get_tokenizer("char_ngram", ns=(2, 3)),
    vectorizer = get_vectorizer("tfidf"),
    classifier = get_classifier("svm"),
)
pipe.fit(train.texts, train.labels)
preds = pipe.predict(test.texts)
print(classification_report(test.labels, preds, train.label_names))
```

## 全組み合わせ比較

```bash
python main.py
```

48 通り (4×3×4) の組み合わせを評価し、精度順にソートして出力する。

## 精度を最優先する設計選択

| 箇所 | 採用した工夫 | 理由 |
|------|-------------|------|
| Naive Bayes | 全て対数計算、`α=1` Laplace smoothing、入力 `clip(0,∞)` | 未知語・アンダーフロー・負値混入に堅牢 |
| Logistic Regression | フルバッチ GD + L2 + 早期停止 + 数値安定 sigmoid | ミニバッチより収束が滑らか |
| SVM | 平均化 SGD (Pegasos) + 学習率 `1/(λt)` 減衰 | 平均化で汎化性能向上 |
| TF-IDF | sublinear TF (`1+log TF`) + smooth IDF + L2 正規化 | 過大な TF を抑え長文/短文を等価に扱う |
| k-NN | コサイン類似度 + 距離重み付き投票 | テキスト疎ベクトルでは L2 距離より精度高い |
| 共通 | `unicodedata.NFKC` 正規化 + 小文字化 | 全半角ゆれ・大文字小文字を吸収 |

## 拡張アイデア

- **Tokenizer**: BPE / SentencePiece / MeCab + Janome
- **Vectorizer**: Word2Vec / fastText / SVD で次元圧縮 (LSA)
- **Classifier**: Multinomial Logistic Regression (多クラス) / MLP / Transformer
- **評価**: K-fold 交差検証, ROC-AUC, 学習曲線

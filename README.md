# AI 学習ディレクトリ

NumPy のみで実装した機械学習の 3 つのモジュール。

| ディレクトリ | 内容 |
|---|---|
| `text_ml/` | テキスト分類 (トークナイザ → TF-IDF → 分類器) |
| `reinforcement_learning/` | 強化学習 (GridWorld × Q-learning ほか) |
| `object_recognition/` | 物体認識 (HOG 特徴量 → 分類器) |

---

## セットアップ

```bash
# 1. 仮想環境を作成
python3 -m venv .venv

# 2. 仮想環境を有効化
source .venv/bin/activate

# 3. 依存ライブラリをインストール (numpy のみ)
pip install -r requirements.txt
```

---

## 実行方法

仮想環境を有効化した状態で、各ディレクトリに移動して `main.py` を実行する。

### テキスト分類

```bash
cd text_ml
python main.py
```

```
tokenizer   vectorizer  classifier    accuracy
----------------------------------------------
char_ngram  tfidf       svm              0.923
char_ngram  tfidf       logreg           0.916
...
```

### 強化学習

```bash
cd reinforcement_learning
python main.py
```

```
=== Env: gridworld ===
policy      algo          mean     ± std
------------------------------------------
decay_eps   dyna_q        0.832     0.021
...
```

### 物体認識

```bash
cd object_recognition
python main.py
```

```
preproc   feature   classifier    accuracy
------------------------------------------
standard  edge      softmax          0.942
...
```

---

## 仮想環境の終了

```bash
deactivate
```

---

## チュートリアル

各ディレクトリの `TUTORIAL.md` に、アルゴリズムの数式・仕組み・実験課題をまとめている。  
コードブロックは numpy と標準ライブラリのみで動作するため、そのままターミナルで実行できる。

| ファイル | 内容 |
|---|---|
| `text_ml/TUTORIAL.md` | トークナイズ・TF-IDF・各分類器の数式解説 |
| `reinforcement_learning/TUTORIAL.md` | MDP・ベルマン方程式・各アルゴリズムの解説 |
| `object_recognition/TUTORIAL.md` | HOG・LBP・DCT・各分類器の数式解説 |

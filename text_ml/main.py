"""
==============================================================
  main.py | 全アルゴリズムの組み合わせ比較
==============================================================
Tokenizer × Vectorizer × Classifier の組み合わせを評価し、
精度を比較する。

実行:
    python main.py
"""

from itertools import product

from data       import load_dataset
from tokenizer  import get_tokenizer
from vectorizer import get_vectorizer
from classifier import get_classifier
from pipeline   import TextClassificationPipeline
from metrics    import accuracy, classification_report


# ─── 比較する戦略の組み合わせ ────────────────────────────────
TOKENIZERS  = ["whitespace", "char_ngram", "regex", "char"]
VECTORIZERS = ["count", "tfidf", "hashing", "bm25"]
CLASSIFIERS = ["nb", "logreg", "svm", "knn", "perceptron", "ridge"]


def build_pipeline(tok_name: str, vec_name: str, clf_name: str):
    """各戦略名から TextClassificationPipeline を構築する"""
    # トークナイザ: char_ngram は n=(2,3) を既定値に
    if tok_name == "char_ngram":
        tokenizer = get_tokenizer(tok_name, ns=(2, 3))
    else:
        tokenizer = get_tokenizer(tok_name)

    # ベクトル化器: hashing は次元を抑える
    if vec_name == "hashing":
        vectorizer = get_vectorizer(vec_name, n_features=2**12)
    else:
        vectorizer = get_vectorizer(vec_name, min_df=1)

    # 分類器: NB は非負カウントを前提とするため tfidf/hashing と相性が悪い
    classifier = get_classifier(clf_name)

    return TextClassificationPipeline(tokenizer, vectorizer, classifier)


def run_all_combinations():
    train, test = load_dataset(seed=42)

    print(f"訓練データ: {len(train.texts)} 件")
    print(f"評価データ: {len(test.texts)} 件")
    print()

    results = []
    for tok, vec, clf in product(TOKENIZERS, VECTORIZERS, CLASSIFIERS):
        # NB は負値を持てない → hashing は signed=False にする等の対処が必要
        # 簡単のため NB × hashing は精度低下を許容してそのまま実行
        try:
            pipe = build_pipeline(tok, vec, clf)
            pipe.fit(train.texts, train.labels)
            preds = pipe.predict(test.texts)
            acc   = accuracy(test.labels, preds)
            results.append((tok, vec, clf, acc))
        except Exception as e:
            results.append((tok, vec, clf, f"ERR: {type(e).__name__}"))

    # 精度の高い順に表示
    print(f"{'tokenizer':<12}{'vectorizer':<12}{'classifier':<12}{'accuracy':>10}")
    print("-" * 46)
    valid = [r for r in results if isinstance(r[3], float)]
    errs  = [r for r in results if not isinstance(r[3], float)]
    for tok, vec, clf, acc in sorted(valid, key=lambda r: -r[3]):
        print(f"{tok:<12}{vec:<12}{clf:<12}{acc:>10.3f}")
    if errs:
        print("\n[エラーになった組み合わせ]")
        for tok, vec, clf, msg in errs:
            print(f"  {tok}/{vec}/{clf}: {msg}")


def run_best_pipeline_detail():
    """精度重視のおすすめ構成で詳細レポートを表示"""
    train, test = load_dataset(seed=42)

    pipe = build_pipeline(
        tok_name="char_ngram",   # 日本語に頑健
        vec_name="tfidf",        # 重み付けで精度向上
        clf_name="svm",          # 高次元疎データに強い
    )
    pipe.fit(train.texts, train.labels)
    preds = pipe.predict(test.texts)

    print("\n=== 推奨構成: char_ngram + tfidf + svm ===")
    print(classification_report(test.labels, preds, train.label_names))

    # サンプル予測をいくつか表示
    print("\n[予測サンプル]")
    for text, true, pred in zip(test.texts[:5], test.labels[:5], preds[:5]):
        mark = "○" if true == pred else "×"
        print(f"  {mark} 真={train.label_names[true]} / 予={train.label_names[pred]} : {text}")


if __name__ == "__main__":
    run_all_combinations()
    run_best_pipeline_detail()

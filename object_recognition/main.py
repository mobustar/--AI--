"""
==============================================================
  main.py | 全アルゴリズム組み合わせ比較
==============================================================
Preprocessor × Feature × Classifier の組み合わせを比較する。

実行:
    python main.py
"""

from itertools import product

from data       import make_shapes, train_test_split
from preprocess import get_preprocessor
from feature    import get_feature
from classifier import get_classifier
from pipeline   import ObjectRecognitionPipeline
from metrics    import accuracy, classification_report


PREPROCESSORS = ["noop", "minmax", "standard", "hist"]
FEATURES      = ["raw", "hog", "lbp", "edge", "dct"]
CLASSIFIERS   = ["knn", "softmax", "svm", "mlp", "gnb", "lda"]


def build_pipeline(prep_name: str, feat_name: str, clf_name: str):
    return ObjectRecognitionPipeline(
        preprocessor = get_preprocessor(prep_name),
        feature      = get_feature(feat_name),
        classifier   = get_classifier(clf_name),
    )


def run_all_combinations():
    ds = make_shapes(n_per_class=200, size=24, noise=0.08, seed=0)
    train, test = train_test_split(ds, ratio=0.8, seed=1)

    print(f"訓練データ: {len(train.labels)} 枚 / 評価データ: {len(test.labels)} 枚")
    print(f"画像サイズ: {train.images.shape[1:]} / クラス: {train.label_names}")
    print()

    results = []
    for p, f, c in product(PREPROCESSORS, FEATURES, CLASSIFIERS):
        try:
            pipe = build_pipeline(p, f, c)
            pipe.fit(train.images, train.labels)
            preds = pipe.predict(test.images)
            acc   = accuracy(test.labels, preds)
            results.append((p, f, c, acc))
        except Exception as e:
            results.append((p, f, c, f"ERR: {type(e).__name__}"))

    valid = [r for r in results if isinstance(r[3], float)]
    errs  = [r for r in results if not isinstance(r[3], float)]

    print(f"{'preproc':<10}{'feature':<10}{'classifier':<12}{'accuracy':>10}")
    print("-" * 42)
    for p, f, c, acc in sorted(valid, key=lambda r: -r[3]):
        print(f"{p:<10}{f:<10}{c:<12}{acc:>10.3f}")
    if errs:
        print("\n[エラー]")
        for p, f, c, msg in errs:
            print(f"  {p}/{f}/{c}: {msg}")


def run_recommended():
    """精度重視のおすすめ構成 (実測で最も高精度な組み合わせ)"""
    ds = make_shapes(n_per_class=200, size=24, noise=0.08, seed=0)
    train, test = train_test_split(ds, ratio=0.8, seed=1)

    pipe = build_pipeline("standard", "edge", "softmax")
    pipe.fit(train.images, train.labels)
    preds = pipe.predict(test.images)

    print("\n=== 推奨構成: standard + edge + softmax ===")
    print(classification_report(test.labels, preds, train.label_names))


if __name__ == "__main__":
    run_all_combinations()
    run_recommended()

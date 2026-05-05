"""
==============================================================
  classifier.py | 分類器 (戦略パターン)
==============================================================
ベクトル化された文書を分類する。

選択アルゴリズム:
  1. MultinomialNB        - 多項ナイーブベイズ。高速・小データに強い
  2. LogisticRegression   - 確率的線形分類器。万能なベースライン
  3. LinearSVM            - マージン最大化。高次元疎データに強い
  4. KNNClassifier        - インスタンスベース。学習不要だが推論が重い

精度の傾向 (TF-IDF + 短文の場合):
  LinearSVM ≈ LogisticRegression > MultinomialNB > KNN
  (NB は CountVectorizer 前提で設計されている点に注意)

数値安定性のための工夫:
  - Naive Bayes:     log 確率で計算 (アンダーフロー防止)
  - LogReg:          log-sum-exp (オーバーフロー防止)
  - SVM:             ヒンジ損失 + L2 正則化 を平均化 SGD で
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# ─── 1. Multinomial Naive Bayes ─────────────────────────────
class MultinomialNB(BaseClassifier):
    """
    多項ナイーブベイズ (テキスト分類の古典的かつ強力なベースライン)。

    モデル:
      P(c|d) ∝ P(c) * Π_i P(t_i|c)^{x_i}
      → log P(c|d) = log P(c) + Σ_i x_i * log P(t_i|c)

    学習:
      P(c)     = N_c / N                    (クラス事前確率)
      P(t_i|c) = (count(t_i,c) + α) /       (Laplace smoothing)
                 (Σ_j count(t_j,c) + α*|V|)

    アンダーフロー対策で全て対数で扱う。
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: np.ndarray = None
        self.class_log_prior_: np.ndarray = None
        self.feature_log_prob_: np.ndarray = None  # (n_classes, n_features)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        # NB は非負カウント前提なので負値はクリップ (Hashing 符号トリックなどへの防御)
        X = np.clip(X, 0.0, None)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            # クラス事前確率 log P(c)
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            # 特徴量の出現回数 + Laplace smoothing (alpha>0 で必ず正値になる)
            count = X_c.sum(axis=0) + self.alpha
            # 正規化して log P(t|c) を得る
            self.feature_log_prob_[i] = np.log(count) - np.log(count.sum())
        return self

    def _joint_log_likelihood(self, X):
        # log P(c|d) ∝ log P(c) + X @ log P(t|c)^T
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        X = np.clip(np.asarray(X, dtype=np.float64), 0.0, None)
        return self.classes_[self._joint_log_likelihood(X).argmax(axis=1)]

    def predict_proba(self, X):
        """log-sum-exp で安定的に確率化"""
        X = np.clip(np.asarray(X, dtype=np.float64), 0.0, None)
        jll = self._joint_log_likelihood(X)
        log_norm = self._logsumexp(jll, axis=1, keepdims=True)
        return np.exp(jll - log_norm)

    @staticmethod
    def _logsumexp(x, axis=None, keepdims=False):
        m = np.max(x, axis=axis, keepdims=True)
        out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
        return out if keepdims else np.squeeze(out, axis=axis)


# ─── 2. Logistic Regression (二値) ──────────────────────────
class LogisticRegression(BaseClassifier):
    """
    L2 正則化付きロジスティック回帰 (二値分類)。

    モデル: P(y=1|x) = σ(w·x + b),  σ(z) = 1/(1+e^{-z})

    損失: L = -1/N Σ [y log σ(z) + (1-y) log(1-σ(z))] + 0.5 * λ‖w‖²
    勾配: ∂L/∂w = 1/N X^T (σ(z) - y) + λw
          ∂L/∂b = 1/N Σ (σ(z) - y)

    最適化: フルバッチ勾配降下 (収束が安定 → 精度重視)
    """

    def __init__(self,
                 lr: float = 0.5,
                 reg: float = 1e-3,
                 epochs: int = 1000,
                 tol: float = 1e-6):
        self.lr      = lr
        self.reg     = reg
        self.epochs  = epochs
        self.tol     = tol
        self.w_:  np.ndarray = None
        self.b_:  float      = 0.0

    @staticmethod
    def _sigmoid(z):
        # オーバーフロー対策の数値安定実装
        out = np.empty_like(z)
        pos = z >= 0
        out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
        ez        = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        self.w_ = np.zeros(d)
        self.b_ = 0.0
        prev_loss = np.inf

        for _ in range(self.epochs):
            z       = X @ self.w_ + self.b_
            p       = self._sigmoid(z)
            err     = p - y                                  # (n,)
            grad_w  = X.T @ err / n + self.reg * self.w_
            grad_b  = err.mean()
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b

            # 早期収束判定
            loss = self._loss(X, y)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def _loss(self, X, y):
        z = X @ self.w_ + self.b_
        # 数値安定な交差エントロピー: log(1+exp(z)) - y*z
        log1pexp = np.where(z >= 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
        ce = (log1pexp - y * z).mean()
        return ce + 0.5 * self.reg * (self.w_ @ self.w_)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._sigmoid(X @ self.w_ + self.b_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ─── 3. Linear SVM (二値) ───────────────────────────────────
class LinearSVM(BaseClassifier):
    """
    L2 正則化 + ヒンジ損失の線形 SVM。

    モデル: f(x) = w·x + b, 予測 = sign(f(x))
    損失:  L = 1/N Σ max(0, 1 - y_i f(x_i)) + 0.5 * λ‖w‖²
           y ∈ {-1, +1}

    最適化: 平均化 SGD (Pegasos 的) で安定収束。
    精度を重視するため学習率は 1/(λt) で減衰させる。
    """

    def __init__(self,
                 reg: float = 1e-3,
                 epochs: int = 50,
                 seed: int = 0):
        self.reg    = reg
        self.epochs = epochs
        self.seed   = seed
        self.w_: np.ndarray = None
        self.b_: float = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        # ラベルを {-1,+1} に変換
        y01 = np.asarray(y)
        y_pm = np.where(y01 == 1, 1.0, -1.0)

        n, d   = X.shape
        rng    = np.random.default_rng(self.seed)
        self.w_ = np.zeros(d)
        self.b_ = 0.0

        # 平均化用のバッファ
        w_avg = np.zeros(d)
        b_avg = 0.0
        t = 0
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for i in order:
                t += 1
                eta = 1.0 / (self.reg * t)            # 学習率減衰
                margin = y_pm[i] * (X[i] @ self.w_ + self.b_)
                # ヒンジ損失の勾配
                if margin < 1:
                    self.w_ = (1 - eta * self.reg) * self.w_ + eta * y_pm[i] * X[i]
                    self.b_ = self.b_ + eta * y_pm[i]
                else:
                    self.w_ = (1 - eta * self.reg) * self.w_
                w_avg += self.w_
                b_avg += self.b_
        # 平均化 (汎化性能向上)
        self.w_ = w_avg / t
        self.b_ = b_avg / t
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


# ─── 4. k-NN (コサイン距離) ─────────────────────────────────
class KNNClassifier(BaseClassifier):
    """
    学習データ全件を保持し、推論時に k 個の最近傍で多数決する。

    距離関数: コサイン類似度 (テキストベクトルでは L2 距離より精度が高い)
    weights:  "uniform" or "distance" (距離による重み付き投票)

    精度を重視するため、デフォルトは k=5, weights="distance"。
    """

    def __init__(self, k: int = 5, weights: str = "distance"):
        self.k = k
        self.weights = weights
        self.X_: np.ndarray = None
        self.y_: np.ndarray = None
        self._X_norm: np.ndarray = None

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        # 学習データを L2 正規化しておくと、内積=コサイン類似度になる
        norms = np.linalg.norm(self.X_, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._X_norm = self.X_ / norms
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms

        # コサイン類似度 → 距離 = 1 - 類似度
        sim  = Xn @ self._X_norm.T              # (n_test, n_train)
        dist = 1.0 - sim

        preds = np.empty(X.shape[0], dtype=self.y_.dtype)
        classes = np.unique(self.y_)

        for i in range(X.shape[0]):
            # 最近傍 k 件のインデックス
            idx = np.argpartition(dist[i], self.k)[: self.k]
            d   = dist[i, idx]
            lbl = self.y_[idx]

            if self.weights == "distance":
                # 距離が近いほど重みを大きく (eps で 0 除算回避)
                w = 1.0 / (d + 1e-9)
            else:
                w = np.ones_like(d)

            # クラスごとに重みを集計して最大のものを選ぶ
            scores = np.array([w[lbl == c].sum() for c in classes])
            preds[i] = classes[scores.argmax()]
        return preds


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_classifier(name: str = "logreg", **kwargs) -> BaseClassifier:
    """
    名前で Classifier を取得する
        name in {"nb", "logreg", "svm", "knn"}
    """
    table = {
        "nb":      MultinomialNB,
        "logreg":  LogisticRegression,
        "svm":     LinearSVM,
        "knn":     KNNClassifier,
    }
    if name not in table:
        raise ValueError(f"unknown classifier: {name}")
    return table[name](**kwargs)

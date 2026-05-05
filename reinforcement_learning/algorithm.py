"""
==============================================================
  algorithm.py | 強化学習アルゴリズム (戦略パターン)
==============================================================
Q テーブルベースの 6 つのアルゴリズム。すべて共通インタフェース:
    algo.train(env, policy, n_episodes) → rewards
    algo.greedy_action(state) → 学習後の貪欲方策

選択アルゴリズム:
  1. MonteCarlo       - エピソード終了後に G_t で更新
  2. SARSA            - On-policy TD(0)
  3. QLearning        - Off-policy TD(0)
  4. ExpectedSARSA    - 次状態の期待値で更新 (分散低減)
  5. DoubleQLearning  - 最大化バイアスを除去 (確率的環境で精度向上)
  6. DynaQ            - モデル学習 + プランニングで サンプル効率向上

精度の傾向:
  決定的:     Q-Learning ≈ SARSA ≈ ExpectedSARSA
  確率的:     DoubleQLearning > ExpectedSARSA > Q-Learning
  少データ:   DynaQ が最良

数式まとめ:
  TD ターゲット (それぞれ):
    SARSA:           r + γ Q(s', a')              ※ a' は実際にとる行動
    Q-learning:      r + γ max_a' Q(s', a')
    Expected SARSA:  r + γ Σ_a' π(a'|s') Q(s', a')
    Double Q:        r + γ Q_B(s', argmax_a' Q_A(s', a'))
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List
import numpy as np

from env    import BaseEnv
from policy import BasePolicy


class BaseAlgorithm(ABC):
    """共通基底"""

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.95):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.Q = np.zeros((n_states, n_actions))

    @abstractmethod
    def train(self, env: BaseEnv, policy: BasePolicy, n_episodes: int) -> List[float]:
        ...

    def greedy_action(self, state: int) -> int:
        return int(self.Q[state].argmax())


# ─── 1. Monte Carlo (Every-Visit) ───────────────────────────
class MonteCarlo(BaseAlgorithm):
    """
    エピソード終了後にリターン G_t = Σ γ^k r_{t+k+1} を計算し
    各 (s,a) ペアに対して Q を G_t に向けて更新する。
        Q(s,a) ← Q(s,a) + α [G_t - Q(s,a)]
    """

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            episode = []                 # [(s, a, r), ...]
            state = env.reset()
            done  = False
            while not done:
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)
                episode.append((state, a, r))
                state = ns

            # 後ろから累積リターン G を計算して更新
            G = 0.0
            for s, a, r in reversed(episode):
                G = r + self.gamma * G
                self.Q[s, a] += self.alpha * (G - self.Q[s, a])

            rewards.append(sum(t[2] for t in episode))
            policy.on_episode_end()
        return rewards


# ─── 2. SARSA (On-policy TD) ────────────────────────────────
class SARSA(BaseAlgorithm):
    """
    on-policy TD(0):
        Q(s,a) ← Q(s,a) + α [r + γ Q(s', a') - Q(s,a)]
    a' は方策が実際に選ぶ行動。安全な方策を学ぶ傾向。
    """

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            action = policy.select(self.Q, state)
            done  = False
            total = 0.0
            while not done:
                ns, r, done = env.step(action)
                na = 0 if done else policy.select(self.Q, ns)
                target = r + (0.0 if done else self.gamma * self.Q[ns, na])
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])
                state, action = ns, na
                total += r
            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── 3. Q-Learning (Off-policy TD) ──────────────────────────
class QLearning(BaseAlgorithm):
    """
    off-policy TD(0):
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
    最大化バイアスがあるため、確率的環境では Q を過大評価することがある。
    """

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done  = False
            total = 0.0
            while not done:
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)
                target = r + (0.0 if done else self.gamma * self.Q[ns].max())
                self.Q[state, a] += self.alpha * (target - self.Q[state, a])
                state = ns
                total += r
            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── 4. Expected SARSA ──────────────────────────────────────
class ExpectedSARSA(BaseAlgorithm):
    """
    SARSA の分散を抑える派生:
        Q(s,a) ← Q(s,a) + α [r + γ Σ_a' π(a'|s') Q(s',a') - Q(s,a)]

    π は ε-greedy を仮定して期待値を計算する。
    SARSA より低分散で精度が高くなる傾向。
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        super().__init__(n_states, n_actions, alpha, gamma)
        self.epsilon = epsilon

    def _expected_q(self, state):
        """ε-greedy 方策に対する Q の期待値"""
        q = self.Q[state]
        n = len(q)
        max_a = q.argmax()
        # 同点を考慮した greedy 確率
        max_actions = np.flatnonzero(q == q.max())
        n_max = len(max_actions)
        prob = np.full(n, self.epsilon / n)         # ランダム成分
        prob[max_actions] += (1 - self.epsilon) / n_max
        return float((prob * q).sum())

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            total = 0.0
            while not done:
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)
                target = r + (0.0 if done else self.gamma * self._expected_q(ns))
                self.Q[state, a] += self.alpha * (target - self.Q[state, a])
                state = ns
                total += r
            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── 5. Double Q-Learning ───────────────────────────────────
class DoubleQLearning(BaseAlgorithm):
    """
    Q_A, Q_B の 2 つのテーブルを交互に更新し、最大化バイアスを除去する。
    どちらを更新するかは 50% で選択。
        Q_A(s,a) ← Q_A(s,a) + α [r + γ Q_B(s', argmax_a' Q_A(s',a')) - Q_A(s,a)]

    確率的環境で Q-Learning より明確に良い精度を出す。
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, seed=0):
        # 親の Q 初期化を流用しつつ、追加で QA / QB を持つ
        super().__init__(n_states, n_actions, alpha, gamma)
        self.QA = np.zeros((n_states, n_actions))
        self.QB = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)

    def _sync_Q(self):
        """policy / greedy_action 用に平均値を Q に反映する"""
        self.Q = (self.QA + self.QB) / 2.0

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            total = 0.0
            while not done:
                self._sync_Q()                 # 行動選択時は最新の平均を使う
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)
                if self.rng.random() < 0.5:
                    # QA を更新: 最大化は QA で、評価は QB で
                    best_a = int(self.QA[ns].argmax())
                    target = r + (0.0 if done else self.gamma * self.QB[ns, best_a])
                    self.QA[state, a] += self.alpha * (target - self.QA[state, a])
                else:
                    best_a = int(self.QB[ns].argmax())
                    target = r + (0.0 if done else self.gamma * self.QA[ns, best_a])
                    self.QB[state, a] += self.alpha * (target - self.QB[state, a])
                state = ns
                total += r
            rewards.append(total)
            policy.on_episode_end()
        self._sync_Q()                          # 学習終了後も同期しておく
        return rewards


# ─── 6. Dyna-Q (model-based + model-free) ───────────────────
class DynaQ(BaseAlgorithm):
    """
    Dyna-Q (Sutton 1990):
      実環境からのサンプルで Q を更新するだけでなく、
      観測した遷移を「モデル」として保存し、それを使って
      n_planning 回の追加 Q 更新 (planning) を行う。

      → サンプル効率が大幅に向上。
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 n_planning: int = 20, seed: int = 0):
        super().__init__(n_states, n_actions, alpha, gamma)
        self.n_planning = n_planning
        self.rng = np.random.default_rng(seed)
        # 決定的モデル: model[(s,a)] = (r, s')
        self.model: dict = {}
        self.observed_pairs: list = []

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            total = 0.0
            while not done:
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)

                # 1) 直接的 Q 学習更新
                target = r + (0.0 if done else self.gamma * self.Q[ns].max())
                self.Q[state, a] += self.alpha * (target - self.Q[state, a])

                # 2) モデル保存
                if (state, a) not in self.model:
                    self.observed_pairs.append((state, a))
                self.model[(state, a)] = (r, ns, done)

                # 3) プランニング (n_planning 回)
                if self.observed_pairs:
                    for _p in range(self.n_planning):
                        idx = int(self.rng.integers(len(self.observed_pairs)))
                        ps, pa = self.observed_pairs[idx]
                        pr, pns, pdone = self.model[(ps, pa)]
                        ptarget = pr + (0.0 if pdone else self.gamma * self.Q[pns].max())
                        self.Q[ps, pa] += self.alpha * (ptarget - self.Q[ps, pa])

                state = ns
                total += r
            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_algorithm(name: str, n_states: int, n_actions: int,
                  **kwargs) -> BaseAlgorithm:
    table = {
        "mc":       MonteCarlo,
        "sarsa":    SARSA,
        "q":        QLearning,
        "esarsa":   ExpectedSARSA,
        "double_q": DoubleQLearning,
        "dyna_q":   DynaQ,
    }
    if name not in table:
        raise ValueError(f"unknown algorithm: {name}")
    return table[name](n_states=n_states, n_actions=n_actions, **kwargs)

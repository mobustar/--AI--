"""
==============================================================
  trainer.py | 強化学習トレーナー
==============================================================
Env × Policy × Algorithm を組み合わせて学習・評価する。
精度評価のため複数 seed で実行して平均化することも支援する。
"""

from typing import List, Tuple
import numpy as np

from env       import BaseEnv
from policy    import BasePolicy
from algorithm import BaseAlgorithm


def evaluate_greedy(env: BaseEnv, algo: BaseAlgorithm,
                    n_episodes: int = 50) -> float:
    """
    学習後の貪欲方策で平均報酬を測る。
    確率的環境では複数試行の平均が必要。
    """
    total = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = algo.greedy_action(state)
            state, r, done = env.step(a)
            ep_r += r
        total += ep_r
    return total / n_episodes


def train_and_evaluate(env_factory, policy_factory, algo_factory,
                        n_episodes: int = 500,
                        n_eval_episodes: int = 50,
                        n_seeds: int = 3) -> Tuple[float, float, List[List[float]]]:
    """
    異なる seed で n_seeds 回学習を行い、評価平均と標準偏差を返す。

    Args:
        env_factory:    seed → BaseEnv  を返す callable
        policy_factory: seed → BasePolicy
        algo_factory:   (n_states, n_actions, seed) → BaseAlgorithm

    Returns:
        mean_eval: 平均評価報酬
        std_eval:  評価報酬の標準偏差
        all_train_curves: 各 seed の訓練報酬曲線
    """
    eval_rewards = []
    train_curves = []
    for seed in range(n_seeds):
        env  = env_factory(seed)
        algo = algo_factory(env.n_states, env.n_actions, seed)
        pol  = policy_factory(seed)
        train_curve = algo.train(env, pol, n_episodes)
        train_curves.append(train_curve)

        # 評価用に環境をリセットして貪欲方策で報酬計測
        eval_env = env_factory(seed + 10000)
        eval_r   = evaluate_greedy(eval_env, algo, n_eval_episodes)
        eval_rewards.append(eval_r)

    return float(np.mean(eval_rewards)), float(np.std(eval_rewards)), train_curves

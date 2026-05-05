"""
==============================================================
  main.py | 強化学習: 全組み合わせの比較実行
==============================================================
Env × Policy × Algorithm の組み合わせを 3 seed 平均で評価する。

実行:
    python main.py
"""

from itertools import product

from env       import get_env
from policy    import get_policy
from algorithm import get_algorithm
from trainer   import train_and_evaluate


# ─── 比較する戦略 ───────────────────────────────────────────
ENVS       = ["gridworld", "cliffwalk", "stochastic"]
POLICIES   = ["eps", "decay_eps", "boltzmann", "ucb"]
ALGORITHMS = ["mc", "sarsa", "q", "esarsa", "double_q", "dyna_q"]


def make_factories(env_name, pol_name, algo_name):
    """seed を受け取って各部品を生成する factory を返す"""

    def env_factory(seed):
        # GridWorld/CliffWalk は決定的なので seed 不要、
        # StochasticGrid だけが seed を要する
        if env_name == "stochastic":
            return get_env(env_name, seed=seed)
        return get_env(env_name)

    def policy_factory(seed):
        if pol_name == "eps":
            return get_policy(pol_name, epsilon=0.1, seed=seed)
        if pol_name == "decay_eps":
            return get_policy(pol_name, epsilon=1.0, decay=0.995,
                              epsilon_min=0.05, seed=seed)
        if pol_name == "boltzmann":
            return get_policy(pol_name, tau=0.5, seed=seed)
        if pol_name == "ucb":
            return get_policy(pol_name, c=1.4, seed=seed)
        raise ValueError(pol_name)

    def algo_factory(n_states, n_actions, seed):
        # アルゴリズム固有の引数を整理
        common = dict(alpha=0.1, gamma=0.95)
        if algo_name == "esarsa":
            return get_algorithm(algo_name, n_states, n_actions,
                                 **common, epsilon=0.1)
        if algo_name == "double_q":
            return get_algorithm(algo_name, n_states, n_actions,
                                 **common, seed=seed)
        if algo_name == "dyna_q":
            return get_algorithm(algo_name, n_states, n_actions,
                                 **common, n_planning=10, seed=seed)
        return get_algorithm(algo_name, n_states, n_actions, **common)

    return env_factory, policy_factory, algo_factory


def run_all_combinations(n_episodes: int = 400, n_seeds: int = 3):
    print(f"訓練エピソード: {n_episodes} / seed 数: {n_seeds}\n")

    rows = []
    for env_name, pol_name, algo_name in product(ENVS, POLICIES, ALGORITHMS):
        envF, polF, algoF = make_factories(env_name, pol_name, algo_name)
        try:
            mean, std, _ = train_and_evaluate(
                envF, polF, algoF,
                n_episodes=n_episodes,
                n_eval_episodes=30,
                n_seeds=n_seeds,
            )
            rows.append((env_name, pol_name, algo_name, mean, std))
        except Exception as e:
            rows.append((env_name, pol_name, algo_name, None, type(e).__name__))

    # 環境ごとに結果をまとめて表示
    for env_name in ENVS:
        print(f"=== Env: {env_name} ===")
        print(f"{'policy':<12}{'algo':<10}{'mean':>10}{'± std':>10}")
        print("-" * 42)
        env_rows = [r for r in rows if r[0] == env_name and isinstance(r[3], float)]
        for _, pol, algo, mean, std in sorted(env_rows, key=lambda r: -r[3]):
            print(f"{pol:<12}{algo:<10}{mean:>10.3f}{std:>10.3f}")
        print()


def run_recommended():
    """環境ごとの推奨構成を詳細評価 (報酬最大化を最優先)"""
    recommendations = [
        # 決定的・短経路: Dyna-Q がサンプル効率と精度のバランスが良い
        ("gridworld",  "decay_eps", "dyna_q"),
        # 崖回避が必要な環境: Q-learning が最短経路 (-13) を学ぶ
        # SARSA は安全な迂回経路を学ぶ (-15~-17) ので別の最適化基準
        ("cliffwalk",  "decay_eps", "q"),
        # 確率的: Double Q-Learning は最大化バイアスを除去できる
        ("stochastic", "decay_eps", "double_q"),
    ]
    print("\n=== 推奨構成 (各環境ごとに報酬最大化を優先) ===")
    for env_name, pol_name, algo_name in recommendations:
        envF, polF, algoF = make_factories(env_name, pol_name, algo_name)
        mean, std, _ = train_and_evaluate(envF, polF, algoF,
                                           n_episodes=500,
                                           n_eval_episodes=50,
                                           n_seeds=5)
        print(f"  {env_name:<12} | {pol_name:<10} | {algo_name:<10}"
              f" | mean={mean:7.3f} ± {std:.3f}")


if __name__ == "__main__":
    run_all_combinations(n_episodes=400, n_seeds=3)
    run_recommended()

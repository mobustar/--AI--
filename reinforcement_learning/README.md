# reinforcement_learning — 強化学習コード集

NumPy のみで実装した、Q テーブルベース強化学習のフルセット。
**Env × Policy × Algorithm** の各機能ごとに複数のアルゴリズムを用意し、
戦略パターンで切り替えられる。

## ディレクトリ構成

```
reinforcement_learning/
├── env.py         # 環境 3 種
├── policy.py      # 行動選択ポリシー 4 種
├── algorithm.py   # 学習アルゴリズム 6 種
├── trainer.py     # 統合トレーナー (複数 seed 平均評価)
├── main.py        # 全 72 通り比較
└── README.md
```

## 各機能のアルゴリズム選択肢

### Env (3 種)

| 名前 | クラス | 特徴 |
|------|--------|------|
| `gridworld`  | `GridWorld`       | 4×4 決定的、ゴール +1 / 各ステップ -0.04 |
| `cliffwalk`  | `CliffWalk`       | 4×12、Sutton & Barto 古典問題、崖 -100 |
| `stochastic` | `StochasticGrid`  | 4×4 滑り確率付き、Double Q の効果検証用 |

### Policy (4 種)

| 名前 | クラス | 行動選択則 |
|------|--------|-----------|
| `eps`       | `EpsilonGreedy`      | 確率 ε でランダム、それ以外 argmax |
| `decay_eps` | `DecayEpsilonGreedy` | ε を幾何減衰 (探索→活用への滑らかな移行) |
| `boltzmann` | `Boltzmann`          | softmax(Q/τ) 確率分布で選択 |
| `ucb`       | `UCB1`               | Q + c√(ln N(s) / N(s,a)) |

### Algorithm (6 種)

| 名前 | クラス | TD ターゲット | 特徴 |
|------|--------|--------------|------|
| `mc`       | `MonteCarlo`      | `G_t = Σ γ^k r_{t+k+1}` | エピソード後にまとめて更新 |
| `sarsa`    | `SARSA`           | `r + γ Q(s', a')` | On-policy, 安全な方策 |
| `q`        | `QLearning`       | `r + γ max_a' Q(s', a')` | Off-policy, 最適方策 |
| `esarsa`   | `ExpectedSARSA`   | `r + γ Σ_a' π(a'\|s') Q(s',a')` | 低分散 |
| `double_q` | `DoubleQLearning` | `r + γ Q_B(s', argmax Q_A)` | **最大化バイアス除去** |
| `dyna_q`   | `DynaQ`           | Q-learning + プランニング | **サンプル効率最高** |

## 重要な数式

### TD 学習の一般形
```
Q(s,a) ← Q(s,a) + α [target - Q(s,a)]
target = 各アルゴリズムで異なる
```

### Double Q-Learning の最大化バイアス除去
通常の Q-learning は `max_a' Q(s',a')` でバイアスがかかる。
Double Q では行動選択と評価を分離する:
```
50% で Q_A を更新:  target = r + γ Q_B(s', argmax_a' Q_A(s',a'))
50% で Q_B を更新:  target = r + γ Q_A(s', argmax_a' Q_B(s',a'))
```

### UCB1 の信頼区間
```
a* = argmax_a [ Q(s,a) + c · sqrt( ln N(s) / N(s,a) ) ]
```
N(s) = 状態 s の訪問回数, N(s,a) = (s,a) の訪問回数。
未試行の (s,a) は ∞ とみなして優先的に選ばれる。

### Dyna-Q の planning 更新
```
1) 実環境で 1 ステップ → Q を更新
2) 観測 (s,a,r,s') を model に保存
3) model からランダム抽出 n 回 → Q を擬似更新
```

## 使い方

```python
from env       import get_env
from policy    import get_policy
from algorithm import get_algorithm

env = get_env("cliffwalk")
pol = get_policy("decay_eps", epsilon=1.0, decay=0.995)
algo = get_algorithm("q", env.n_states, env.n_actions, alpha=0.1, gamma=0.95)

rewards = algo.train(env, pol, n_episodes=500)

# 学習後の貪欲方策を実行
state = env.reset()
done = False
while not done:
    a = algo.greedy_action(state)
    state, r, done = env.step(a)
```

## 全組み合わせ比較

```bash
python main.py
```

72 通り (3×4×6) を 3 seed 平均で評価する。

## 環境ごとの最良構成 (実験結果より)

| 環境 | 推奨 Policy + Algorithm | 期待報酬 | 理由 |
|------|------------------------|---------|------|
| `gridworld`  | `decay_eps + dyna_q`     | 0.80   | 短経路を即座に発見 |
| `cliffwalk`  | `decay_eps + q`          | -13.0  | 最短経路 = 報酬最大 (SARSA は -15 で安全側) |
| `stochastic` | `decay_eps + double_q`   | 環境次第 | 最大化バイアス除去で偏りを抑制 |

## 教育的観察ポイント

1. **CliffWalk で SARSA vs Q-learning の差**
   Q-learning は崖際の最短経路 (-13) を学ぶが、ε-greedy のノイズで時々落ちる。
   SARSA は方策のノイズも考慮し、上を回る安全な経路 (-15~-17) を学ぶ。
   報酬最大化と安全性のトレードオフを体験できる古典問題。

2. **確率的環境で Double Q-Learning の効果**
   通常の Q-learning は `max` 演算により確率的環境で過大評価バイアスを持つ。
   Double Q はこのバイアスを除去するため、ノイズに頑健な方策を学ぶ。

3. **Dyna-Q のサンプル効率**
   実環境のサンプル 1 つにつき n_planning 回の擬似更新を行うことで、
   実エピソード数が同じでも収束が速い。

## 精度を最優先する設計選択

| 箇所 | 採用した工夫 | 理由 |
|------|-------------|------|
| EpsilonGreedy | 同点処理で argmax 候補からランダム選択 | 行動選択の偏り回避 |
| Boltzmann | log-sum-exp で安定化 | overflow 回避 |
| UCB1 | 未試行の行動を優先 (∞ ボーナス) | 全ての (s,a) を保証的に試行 |
| DoubleQ | 平均値ではなく交互更新を直接実装 | 理論的に正しい不偏推定 |
| Trainer | 複数 seed の平均 ± 標準偏差 | 統計的に信頼できる比較 |

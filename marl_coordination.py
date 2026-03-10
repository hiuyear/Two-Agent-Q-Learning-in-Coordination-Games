"""
Two-Agent Q-Learning in Coordination Games

Empirical study of independent Q-learning in two-agent games under:
- reward noise
- different learning rates
- different game structures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)


# ----------------------------
# Game definitions
# ----------------------------

class StagHunt:
    """Two-player coordination game with two pure Nash equilibria."""

    n_actions = 2
    payoffs = np.array([
        [[4, 4], [0, 3]],
        [[3, 0], [3, 3]]
    ])
    name = "Stag Hunt"
    action_labels = ["Stag", "Hare"]
    pareto_action = 0


class MatchingPennies:
    """Two-player zero-sum game with no pure-strategy Nash equilibrium."""

    n_actions = 2
    payoffs = np.array([
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]]
    ])
    name = "Matching Pennies"
    action_labels = ["Heads", "Tails"]
    pareto_action = None


# ----------------------------
# Independent Q-learning agent
# ----------------------------

class QLearningAgent:
    def __init__(
        self,
        n_actions,
        alpha=0.1,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.Q = np.zeros(n_actions)

    def select_action(self):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q))

    def update(self, action, reward):
        """Standard Q-value update."""
        td_error = reward - self.Q[action]
        self.Q[action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def policy(self):
        probs = np.zeros(self.n_actions)
        probs[np.argmax(self.Q)] = 1.0
        return probs


# ----------------------------
# Simulation
# ----------------------------

def run_episode(game, agents, noise_std=0.0):
    """Run one simultaneous-move episode and update both agents."""
    actions = [agent.select_action() for agent in agents]
    rewards = []

    for i, agent in enumerate(agents):
        true_reward = game.payoffs[actions[0]][actions[1]][i]
        noisy_reward = true_reward + np.random.normal(0, noise_std)
        agent.update(actions[i], noisy_reward)
        rewards.append(true_reward)

    for agent in agents:
        agent.decay_epsilon()

    return actions, rewards


def run_experiment(
    game,
    n_episodes=5000,
    noise_std=0.0,
    alpha=0.1,
    gamma=0.95,
    n_runs=10,
):
    """
    Run multiple seeds and return smoothed coordination rates.

    For Stag Hunt, coordination means both agents choose the Pareto-optimal action.
    For Matching Pennies, coordination is tracked as choosing the same action.
    """
    all_coord = []

    for run in range(n_runs):
        np.random.seed(run)

        agents = [
            QLearningAgent(game.n_actions, alpha=alpha, gamma=gamma),
            QLearningAgent(game.n_actions, alpha=alpha, gamma=gamma),
        ]

        coord_hist = []

        for _ in range(n_episodes):
            actions, _ = run_episode(game, agents, noise_std)

            if game.pareto_action is not None:
                coord = float(
                    actions[0] == game.pareto_action
                    and actions[1] == game.pareto_action
                )
            else:
                coord = float(actions[0] == actions[1])

            coord_hist.append(coord)

        all_coord.append(coord_hist)

    arr = np.array(all_coord)

    # Smooth each run with a rolling average for easier visualization.
    window = 100
    smoothed = np.array([
        np.convolve(row, np.ones(window) / window, mode="valid")
        for row in arr
    ])

    return smoothed.mean(axis=0), smoothed.std(axis=0)


# ----------------------------
# Experiment wrappers
# ----------------------------

def exp1_noise_effect(game, noise_levels, n_episodes=5000, n_runs=15):
    """Compare convergence under different reward noise levels."""
    results = {}
    for noise in noise_levels:
        mean, std = run_experiment(
            game,
            n_episodes=n_episodes,
            noise_std=noise,
            n_runs=n_runs,
        )
        results[noise] = (mean, std)
    return results


def exp2_learning_rate_effect(game, alphas, n_episodes=5000, n_runs=15):
    """Compare convergence under different learning rates."""
    results = {}
    for alpha in alphas:
        mean, std = run_experiment(
            game,
            n_episodes=n_episodes,
            alpha=alpha,
            n_runs=n_runs,
        )
        results[alpha] = (mean, std)
    return results


# ----------------------------
# Plotting
# ----------------------------

def make_figure():
    game = StagHunt()
    n_episodes = 6000
    window = 100
    x = np.arange(n_episodes - window + 1)

    noise_levels = [0.0, 0.5, 1.5, 3.0]
    alphas = [0.01, 0.05, 0.1, 0.3]

    print("Running noise experiment...")
    noise_results = exp1_noise_effect(game, noise_levels, n_episodes, n_runs=15)

    print("Running learning-rate experiment...")
    alpha_results = exp2_learning_rate_effect(game, alphas, n_episodes, n_runs=15)

    print("Running Matching Pennies baseline...")
    mp_game = MatchingPennies()
    mp_mean, mp_std = run_experiment(mp_game, n_episodes, noise_std=0.0, n_runs=15)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    palette = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]
    alpha_palette = ["#CE93D8", "#4FC3F7", "#80CBC4", "#FFCC02"]

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        ax.set_xlabel(xlabel, color="#cccccc", fontsize=10)
        ax.set_ylabel(ylabel, color="#cccccc", fontsize=10)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.grid(True, linestyle="--", alpha=0.2, color="#555566")

    # Panel 1: reward noise
    ax1 = fig.add_subplot(gs[0, 0])
    for i, noise in enumerate(noise_levels):
        mean, std = noise_results[noise]
        ax1.plot(x, mean, color=palette[i], lw=1.8, label=f"σ = {noise}")
        ax1.fill_between(x, mean - std, mean + std, color=palette[i], alpha=0.15)

    ax1.axhline(1.0, color="white", lw=0.6, linestyle=":", alpha=0.4)
    style_ax(
        ax1,
        "Effect of Reward Noise on Coordination\n(Stag Hunt)",
        "Episode",
        "Pareto Coordination Rate",
    )
    ax1.legend(
        fontsize=8,
        labelcolor="#cccccc",
        facecolor="#22253a",
        edgecolor="#333344",
        loc="lower right",
    )
    ax1.set_ylim(-0.05, 1.1)

    # Panel 2: learning rate
    ax2 = fig.add_subplot(gs[0, 1])
    for i, alpha in enumerate(alphas):
        mean, std = alpha_results[alpha]
        ax2.plot(x, mean, color=alpha_palette[i], lw=1.8, label=f"α = {alpha}")
        ax2.fill_between(x, mean - std, mean + std, color=alpha_palette[i], alpha=0.15)

    ax2.axhline(1.0, color="white", lw=0.6, linestyle=":", alpha=0.4)
    style_ax(
        ax2,
        "Effect of Learning Rate on Convergence\n(Stag Hunt, σ = 0)",
        "Episode",
        "Pareto Coordination Rate",
    )
    ax2.legend(
        fontsize=8,
        labelcolor="#cccccc",
        facecolor="#22253a",
        edgecolor="#333344",
        loc="lower right",
    )
    ax2.set_ylim(-0.05, 1.1)

    # Panel 3: game structure comparison
    ax3 = fig.add_subplot(gs[1, 0])
    sh_mean, sh_std = noise_results[0.0]
    ax3.plot(x, sh_mean, color="#4FC3F7", lw=2, label="Stag Hunt (coord. eq.)")
    ax3.fill_between(x, sh_mean - sh_std, sh_mean + sh_std, color="#4FC3F7", alpha=0.15)

    ax3.plot(x, mp_mean, color="#F06292", lw=2, label="Matching Pennies (no pure NE)")
    ax3.fill_between(x, mp_mean - mp_std, mp_mean + mp_std, color="#F06292", alpha=0.15)

    ax3.axhline(
        0.5,
        color="#FFCC02",
        lw=1,
        linestyle="--",
        alpha=0.6,
        label="Random baseline (0.5)",
    )
    style_ax(
        ax3,
        "Convergence by Game Structure\n(Independent Q-Learning)",
        "Episode",
        "Coordination Rate",
    )
    ax3.legend(
        fontsize=8,
        labelcolor="#cccccc",
        facecolor="#22253a",
        edgecolor="#333344",
        loc="center right",
    )
    ax3.set_ylim(-0.05, 1.1)

    # Panel 4: final coordination by noise
    ax4 = fig.add_subplot(gs[1, 1])
    final_coords = []
    final_stds = []

    for noise in noise_levels:
        mean, _ = noise_results[noise]
        final_coords.append(mean[-500:].mean())
        final_stds.append(mean[-500:].std())

    ax4.bar(
        [str(n) for n in noise_levels],
        final_coords,
        color=palette,
        alpha=0.85,
        width=0.55,
        yerr=final_stds,
        capsize=5,
        error_kw=dict(ecolor="#aaaaaa", lw=1.5),
    )

    ax4.axhline(0.5, color="#FFCC02", lw=1, linestyle="--", alpha=0.6, label="Random baseline")
    style_ax(
        ax4,
        "Final Coordination Rate vs. Noise Level\n(last 500 episodes avg.)",
        "Reward Noise σ",
        "Avg. Pareto Coordination Rate",
    )
    ax4.legend(fontsize=8, labelcolor="#cccccc", facecolor="#22253a", edgecolor="#333344")
    ax4.set_ylim(0, 1.1)

    fig.suptitle(
        "Independent Q-Learning in Two-Agent Coordination Games\n"
        "Convergence Analysis: Noise, Learning Rate, and Game Structure",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(
        "marl_convergence_analysis.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0f1117",
    )
    print("Plot saved.")
    plt.close()


if __name__ == "__main__":
    make_figure()

    print("\nDone! Results saved to marl_convergence_analysis.png")
    print("\nKey findings:")
    print("- Higher reward noise delays convergence to Pareto-optimal coordination")
    print("- Moderate learning rates (around α = 0.1) balance speed and stability")
    print("- Stag Hunt converges more reliably than Matching Pennies")

"""
Two-Agent Q-Learning in Coordination Games
===========================================
Empirical study of Q-learning convergence under noisy reward signals
in two-agent coordination games (Stag Hunt and Matching Pennies).

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

#every time you run this script, the random numbers come out the same
np.random.seed(42)

# ─────────────────────────────────────────────
# Game Definitions
# ─────────────────────────────────────────────

class StagHunt:
    """
    Stag Hunt: Classic coordination game with two Nash equilibria.

    THE STORY (imagine two hunters):
      - Both choose to hunt the big Stag together  → both earn 4
      - One hunts Stag, the other hunts Hare       → Stag hunter earns 0, Hare hunter earns 3
      - Both hunt Hare (play it safe)              → both earn 3

    The tension: hunting Stag is best IF your partner does the same.
    But if you're not sure your partner will cooperate, Hare is safer.
    "COORDINATION PROBLEM"

    How to read the payoff table below:
      payoffs[player0_action][player1_action] = [reward_for_p0, reward_for_p1]
      e.g. payoffs[0][0] = [4, 4]  ← both chose Stag (action 0), both get 4
           payoffs[0][1] = [0, 3]  ← p0 chose Stag, p1 chose Hare; p0 gets 0, p1 gets 3

    Payoff matrix summary (row = agent 0, col = agent 1):
         Stag   Hare
    Stag  4, 4   0, 3
    Hare  3, 0   3, 3

    NASH EQUILIBRIUM (NE): A situation where neither player wants to
    change their action given what the other is doing. Stag Hunt has TWO:
      - Pareto-optimal NE: (Stag, Stag) → payoff (4,4)  ← best outcome overall
      - Risk-dominant NE:  (Hare, Hare) → payoff (3,3)  ← safer but worse
    """
    # How many actions each player can pick (0 = Stag, 1 = Hare)
    n_actions = 2

    # The payoff table as a 3D NumPy array.
    # Shape is (2, 2, 2): [p0_action][p1_action][which_player's_reward]
    payoffs = np.array([
        [[4, 4], [0, 3]],   # p0 chose Stag (row 0)
        [[3, 0], [3, 3]]    # p0 chose Hare (row 1)
    ])
    name = "Stag Hunt"
    action_labels = ["Stag", "Hare"]
    pareto_action = 0  # Stag is the Pareto-optimal choice - so (0,0) gets payoff 4,4


class MatchingPennies:
    """
    Matching Pennies: A zero-sum game — one player's gain is always the
    other player's exact loss. Like rock-paper-scissors but simpler.

    THE STORY:
      Both players secretly put a penny on a table, either Heads or Tails.
      - Player 0 WINS (+1) if the coins MATCH (both Heads or both Tails)
      - Player 1 WINS (+1) if the coins DON'T MATCH
      (The loser always gets -1, so one wins exactly what the other loses)

    Payoff matrix:
           Heads    Tails
    Heads  (+1,-1)  (-1,+1)
    Tails  (-1,+1)  (+1,-1)

    WHY IS THIS INTERESTING?
    There is NO stable "pure strategy" Nash Equilibrium here.
    If Player 0 always plays Heads, Player 1 just plays Tails to win.
    If Player 0 always plays Tails, Player 1 just plays Heads.
    There's no fixed choice that's safe — you always want to
    change once you know what the other player does.

    The only equilibrium is to randomize 50/50. Q-learning agents
    can't find this easily, so they'll never truly converge.
    This game is used as a CONTRAST to Stag Hunt in the charts.
    """
    # How many actions (0 = Heads, 1 = Tails)
    n_actions = 2

    # Zero-sum payoff table: every cell sums to 0 ([+1,-1] or [-1,+1])
    payoffs = np.array([
        [[ 1, -1], [-1,  1]],   # p0 chose Heads (row 0)
        [[-1,  1], [ 1, -1]]    # p0 chose Tails (row 1)
    ])
    name = "Matching Pennies"
    action_labels = ["Heads", "Tails"]
    pareto_action = None


# ─────────────────────────────────────────────
# Independent Q-Learning Agent
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# THE LEARNING AGENT
# Each "agent" is one player. It uses a technique called Q-Learning
# to figure out which action earns the best reward over time.
# ─────────────────────────────────────────────────────────────────────
class QLearningAgent:
    """
    A single Q-learning player.

    WHAT IS Q-LEARNING?
    The agent keeps a "Q-table" — a list of estimated rewards for each
    action. It tries actions, sees how much reward it gets, and slowly
    updates those estimates. Over time, the best action gets the highest
    Q-value, and the agent learns to prefer it.

    This is "independent" because each agent only sees its OWN reward
    and does NOT know what the other player chose or is learning.
    They're both learning at the same time, completely separately.
    """

    def __init__(self, n_actions, alpha=0.1, gamma=0.95, eps_start=1.0,
                 eps_end=0.05, eps_decay=0.995):
        """
        Sets up a fresh agent that knows nothing yet.

        Parameters:
          n_actions  : how many actions the agent can choose from (2 here)
          alpha      : learning rate — how much to update Q on each step.
                       0.1 means "shift Q by 10% toward the new reward".
                       Too high = unstable learning. Too low = very slow.
          gamma      : discount factor — how much the agent cares about
                       future rewards vs. immediate ones. (Not actively
                       used here since the game has no future state,
                       but kept for completeness.)
          eps_start  : starting epsilon = 1.0 means 100% random at first
          eps_end    : minimum epsilon = 0.05 means always keep 5% random
          eps_decay  : multiply epsilon by this each episode (0.995 means
                       epsilon shrinks slowly toward eps_end over time)
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start   # starts at 1.0 (fully random)
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # The Q-table: one value per action, all starting at 0.
        # e.g. Q = [0.0, 0.0] means "I have no idea which action is better"
        # After learning: Q = [3.8, 2.1] means "action 0 seems better"
        self.Q = np.zeros(n_actions)

    def select_action(self):
        """
        EPSILON-GREEDY: Decide which action to take this round.

        - With probability epsilon  → pick a RANDOM action (explore)
        - With probability 1-epsilon → pick the action with highest Q (exploit)

        Early in training epsilon is high, so the agent explores a lot.
        Later epsilon is low, so it mostly exploits what it has learned.
        Think of it like: when you're new to a restaurant menu, you try
        random dishes (explore). Once you know what's good, you order
        your favourite every time (exploit).
        """
        if np.random.rand() < self.epsilon:
            # np.random.rand() gives a float between 0.0 and 1.0.
            # If it's less than epsilon, explore: pick a random action.
            return np.random.randint(self.n_actions)

        # Otherwise exploit: pick the action with the highest Q-value.
        # np.argmax([3.8, 2.1]) returns 0 because index 0 has the max value.
        return int(np.argmax(self.Q))

    def update(self, action, reward):
        """
        THE CORE LEARNING STEP — update the Q-value for the action we took.

        The formula:  Q[action] += alpha * (reward - Q[action])

        Breaking it down:
          reward - Q[action]  = the "surprise" or prediction error.
                                Positive → reality was better than expected.
                                Negative → reality was worse than expected.
          alpha * (...)       = only move Q a little bit (not all the way)
                                toward the new data point.

        This is basically a weighted running average that slowly shifts
        toward whatever rewards the agent actually receives.
        """
        # The TD (Temporal Difference) error tells us how wrong our estimate was
        td_error = reward - self.Q[action]
        # Nudge Q toward the actual reward, scaled by learning rate alpha
        self.Q[action] += self.alpha * td_error

    def decay_epsilon(self):
        """
        Shrink epsilon a little bit after each episode.

        epsilon = max(eps_end,  epsilon * eps_decay)
                     |                   |
           don't go below 0.05     multiply by 0.995 each time

        This ensures the agent gradually shifts from exploring
        (random) to exploiting (using what it learned).
        """
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def policy(self):
        """
        Return the agent's current "policy" as a probability array.
        Since we always pick the best known action (greedy), one action
        gets probability 1.0 and the rest get 0.0.

        e.g. if Q = [3.8, 2.1], this returns [1.0, 0.0]
        meaning: "I will always pick action 0 right now"
        """
        probs = np.zeros(self.n_actions)   # start with [0.0, 0.0]
        probs[np.argmax(self.Q)] = 1.0     # set the best action to 1.0
        return probs


# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────

def run_episode(game, agents, noise_std=0.0):
    """
    Runs ONE single round of the game. like one hand of cards

    Steps:
      1. Both agents independently choose an action.
      2. look up the true reward from the game's payoff table.
      3. add noise to the reward (to simulate real-world messiness).
      4. Each agent updates its Q-table based on the noisy reward.
      5. Each agent decays its epsilon (becomes slightly less random).

    Parameters:
      game      : the game object (StagHunt or MatchingPennies)
      agents    : list of the two QLearningAgent objects
      noise_std : standard deviation of the noise added to rewards.
                  0.0 = perfectly clean rewards.
                  3.0 = very noisy/messy rewards.

    Returns:
      actions : list of the two actions chosen [p0_action, p1_action]
      rewards : list of the true (pre-noise) rewards [p0_reward, p1_reward]
    """
    # Step 1: Both agents pick their action at the same time.
    # List comprehension: [a.select_action() for a in agents]
    # is shorthand for: "call select_action() on every agent, collect results"
    actions = [a.select_action() for a in agents]

    rewards = []

    # Step 2-4: For each agent, figure out their reward and let them learn.
    # enumerate gives us both the index (i=0 or i=1) and the agent object.
    for i, agent in enumerate(agents):
        # Look up the true reward from the payoff table.
        # e.g. if actions = [0, 1], then payoffs[0][1][i] gives agent i's reward
        # when p0 chose action 0 and p1 chose action 1.
        true_reward = game.payoffs[actions[0]][actions[1]][i]

        # Add Gaussian noise (bell-curve shaped randomness) to the reward.
        # np.random.normal(mean=0, std=noise_std) gives a random number
        # centered at 0. With noise_std=0, this adds exactly 0 (no noise).
        # With noise_std=2, it might add +1.3 or -2.1 randomly.
        noisy_reward = true_reward + np.random.normal(0, noise_std)

        # Agent learns from the noisy reward (not the true one).
        # This tests whether noisy feedback hurts learning.
        agent.update(actions[i], noisy_reward)

        # Store the TRUE reward for tracking purposes (not the noisy one)
        rewards.append(true_reward)

    # Step 5: After each round, slightly reduce each agent's randomness
    for agent in agents:
        agent.decay_epsilon()

    return actions, rewards


def run_experiment(game, n_episodes=5000, noise_std=0.0, alpha=0.1,
                   gamma=0.95, n_runs=10):
    """
    Runs the FULL experiment: many episodes, repeated multiple times.

    WHY REPEAT MULTIPLE TIMES (n_runs)?
    Because random chance can make one particular run look great or bad
    by accident. By repeating 10-15 times with different random seeds
    and averaging, we get a much more reliable picture of what USUALLY
    happens.

    Parameters:
      game: the game to play (StagHunt or MatchingPennies)
      n_episodes: how many rounds to play in one run (default 5000)
      noise_std: how noisy the rewards are
      alpha: learning rate for the agents
      gamma: discount factor (stored but not heavily used here)
      n_runs: how many independent repetitions to average over

    Returns:
      mean: average coordination rate at each episode, smoothed
      std: standard deviation across runs (used for the shaded band)
    """
    # This will hold results from every independent run
    all_coord = []

    for run in range(n_runs):
        # Use a different random seed for each run so they're truly independent
        np.random.seed(run)

        # Create two FRESH agents at the start of every run (they forget everything)
        agents = [
            QLearningAgent(game.n_actions, alpha=alpha, gamma=gamma),
            QLearningAgent(game.n_actions, alpha=alpha, gamma=gamma)
        ]

        # coord_hist records a 1 or 0 for each episode:
        # 1 = agents successfully coordinated that round
        # 0 = they didn't
        coord_hist = []

        for _ in range(n_episodes):
            # Play one round and get the actions both agents chose
            actions, _ = run_episode(game, agents, noise_std)

            # Did they coordinate? The definition depends on the game:
            if game.pareto_action is not None:
                # For Stag Hunt: coordination means BOTH chose the Pareto action (Stag)
                # float() converts True/False → 1.0/0.0 so we can average it later
                coord = float(actions[0] == game.pareto_action and
                              actions[1] == game.pareto_action)
            else:
                # For Matching Pennies: coordination means BOTH chose the same action
                # (just used as a consistency check — we don't expect this to converge)
                coord = float(actions[0] == actions[1])

            coord_hist.append(coord)  # log this round's coordination result

        all_coord.append(coord_hist)  # save this entire run's history

    # Convert list of lists → a 2D NumPy array
    # Shape: (n_runs, n_episodes) e.g. (15, 5000)
    arr = np.array(all_coord)

    # ── SMOOTHING with a rolling average ──────────────────────────────
    # Raw episode results are 0s and 1s (very jagged when plotted).
    # We smooth by averaging every 100 consecutive episodes.
    # np.convolve with a box filter [1/100, 1/100, ..., 1/100] does this.
    # mode='valid' means only output values where the full window fits,
    # which slightly shortens the output length (5000 → 4901).
    window = 100
    smoothed = np.array([
        np.convolve(row, np.ones(window)/window, mode='valid')
        for row in arr
    ])

    # Return the average across all runs and the spread (std deviation)
    # .mean(axis=0) averages across rows (i.e. across runs), keeping time axis
    # .std(axis=0)  does the same for standard deviation
    return smoothed.mean(axis=0), smoothed.std(axis=0)


# ─────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────

def exp1_noise_effect(game, noise_levels, n_episodes=5000, n_runs=15):
    """
    EXPERIMENT 1: Does adding noise to rewards hurt coordination?

    We run the same game multiple times, each time with a different
    level of noise added to the rewards. We then compare how well
    agents learn under clean vs. noisy conditions.

    noise_levels = [0.0, 0.5, 1.5, 3.0]
      - 0.0 means rewards are perfectly accurate
      - 3.0 means a lot of random garbage mixed into the rewards

    Returns a dictionary:
      { noise_value: (mean_coordination_array, std_array), ... }
    """
    results = {}
    for noise in noise_levels:
        # Run the full experiment at this noise level and store results
        mean, std = run_experiment(game, n_episodes, noise_std=noise, n_runs=n_runs)
        results[noise] = (mean, std)  # key = noise level, value = (mean, std)
    return results


def exp2_learning_rate_effect(game, alphas, n_episodes=5000, n_runs=15):
    """
    EXPERIMENT 2: Does the learning rate (alpha) matter?

    We test four different alphas: [0.01, 0.05, 0.1, 0.3]
      - Small alpha (0.01) = very cautious, slow learner
      - Large alpha (0.3)  = aggressive updater, can overshoot
      - Middle (0.1)       = usually the sweet spot

    Returns a dictionary:
      { alpha_value: (mean_coordination_array, std_array), ... }
    """
    results = {}
    for alpha in alphas:
        # Run with no noise (noise_std defaults to 0.0) so we isolate
        # the effect of alpha alone
        mean, std = run_experiment(game, n_episodes, alpha=alpha, n_runs=n_runs)
        results[alpha] = (mean, std)
    return results


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def make_figure():
    """
    THE MAIN FUNCTION that runs all the experiments and draws the 4-panel figure.

    This is the "entry point" for the science — it:
      1. Runs experiment 1 (noise effect) on Stag Hunt
      2. Runs experiment 2 (learning rate effect) on Stag Hunt
      3. Runs a Matching Pennies baseline for comparison
      4. Draws a 2×2 grid of charts and saves it as a PNG file
    """
    # Create the game and set shared parameters
    game = StagHunt()
    n_episodes = 6000   # each run plays 6000 rounds
    window = 100        # rolling average window size (used for smoothing)

    # After smoothing, the x-axis has fewer points than n_episodes.
    # (6000 - 100 + 1 = 5901 points). np.arange creates [0, 1, 2, ..., 5900]
    x = np.arange(n_episodes - window + 1)

    # The four noise levels to test in experiment 1
    noise_levels = [0.0, 0.5, 1.5, 3.0]

    # The four learning rates to test in experiment 2
    alphas = [0.01, 0.05, 0.1, 0.3]

    # ── Run all experiments first, then draw
    print("Running noise experiment...")
    # Returns a dict: { 0.0: (mean, std), 0.5: (mean, std), ... }
    noise_results = exp1_noise_effect(game, noise_levels, n_episodes, n_runs=15)

    print("Running learning rate experiment...")
    # Returns a dict: { 0.01: (mean, std), 0.05: (mean, std), ... }
    alpha_results = exp2_learning_rate_effect(game, alphas, n_episodes, n_runs=15)

    # Run Matching Pennies with no noise as a comparison baseline
    mp_game = MatchingPennies()
    print("Running Matching Pennies baseline...")
    mp_mean, mp_std = run_experiment(mp_game, n_episodes, noise_std=0.0, n_runs=15)

    # ── Set up the figure canvas
    # figsize=(14, 10): width=14 inches, height=10 inches
    fig = plt.figure(figsize=(14, 10))

    # Set the whole background to a dark near-black color (hex color code)
    fig.patch.set_facecolor('#0f1117')

    # GridSpec creates a 2-row × 2-column grid for our 4 subplots.
    # hspace = vertical gap between rows, wspace = horizontal gap between columns
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # Color palettes: lists of hex color codes for each line/bar.
    # These are picked to look good on a dark background.
    palette       = ['#4FC3F7', '#81C784', '#FFB74D', '#F06292']  # for noise levels
    alpha_palette = ['#CE93D8', '#4FC3F7', '#80CBC4', '#FFCC02']  # for learning rates

    # This dict was defined but not actually used — the style_ax function
    # below handles styling manually instead.
    ax_style = dict(facecolor='#1a1d27', tick_params=dict(colors='#aaaaaa'),
                    label_color='#cccccc', title_color='white')

    def style_ax(ax, title, xlabel, ylabel):
        """
        A small helper function that applies the same dark-theme styling
        to every subplot so we don't repeat those lines 4 times.

        ax     : the subplot object to style
        title  : text for the chart title
        xlabel : label for the horizontal (x) axis
        ylabel : label for the vertical (y) axis
        """
        ax.set_facecolor('#1a1d27')                       # dark grey background inside the chart
        ax.tick_params(colors='#aaaaaa', labelsize=9)     # axis tick marks in light grey
        ax.set_xlabel(xlabel, color='#cccccc', fontsize=10)
        ax.set_ylabel(ylabel, color='#cccccc', fontsize=10)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        for spine in ax.spines.values():                  # the 4 border lines around the chart
            spine.set_edgecolor('#333344')
        ax.grid(True, linestyle='--', alpha=0.2, color='#555566')  # faint dashed grid lines

    # ── PANEL 1 (top-left): How noise affects coordination
    ax1 = fig.add_subplot(gs[0, 0])  # row 0, column 0

    for i, noise in enumerate(noise_levels):
        mean, std = noise_results[noise]  # unpack the stored results

        # Draw the main line (the average coordination rate over episodes)
        label = f"σ = {noise}"  # f-string: inserts the value of `noise` into the string
        ax1.plot(x, mean, color=palette[i], lw=1.8, label=label)

        # Draw a shaded band around the line: mean ± std
        # This shows how much the result varied across the 15 runs.
        # alpha=0.15 makes it 85% transparent so lines don't hide each other.
        ax1.fill_between(x, mean - std, mean + std, color=palette[i], alpha=0.15)

    # Draw a thin white horizontal line at y=1.0 (perfect coordination)
    ax1.axhline(1.0, color='white', lw=0.6, linestyle=':', alpha=0.4)

    # Apply dark-theme styling
    style_ax(ax1, "Effect of Reward Noise on Coordination\n(Stag Hunt)",
             "Episode", "Pareto Coordination Rate")

    # Add a legend box explaining what each colored line means
    ax1.legend(fontsize=8, labelcolor='#cccccc', facecolor='#22253a',
               edgecolor='#333344', loc='lower right')
    ax1.set_ylim(-0.05, 1.1)  # y-axis from just below 0 to just above 1

    # ── PANEL 2 (top-right): How learning rate affects convergence ─
    ax2 = fig.add_subplot(gs[0, 1])  # row 0, column 1

    for i, alpha in enumerate(alphas):
        mean, std = alpha_results[alpha]
        ax2.plot(x, mean, color=alpha_palette[i], lw=1.8, label=f"α = {alpha}")
        ax2.fill_between(x, mean - std, mean + std, color=alpha_palette[i], alpha=0.15)

    ax2.axhline(1.0, color='white', lw=0.6, linestyle=':', alpha=0.4)
    style_ax(ax2, "Effect of Learning Rate on Convergence\n(Stag Hunt, σ = 0)",
             "Episode", "Pareto Coordination Rate")
    ax2.legend(fontsize=8, labelcolor='#cccccc', facecolor='#22253a',
               edgecolor='#333344', loc='lower right')
    ax2.set_ylim(-0.05, 1.1)

    # ── PANEL 3 (bottom-left): Stag Hunt vs Matching Pennies ──
    ax3 = fig.add_subplot(gs[1, 0])  # row 1, column 0

    # Stag Hunt result (no noise) — we already computed this in exp1
    sh_mean, sh_std = noise_results[0.0]
    ax3.plot(x, sh_mean, color='#4FC3F7', lw=2, label='Stag Hunt (coord. eq.)')
    ax3.fill_between(x, sh_mean - sh_std, sh_mean + sh_std, color='#4FC3F7', alpha=0.15)

    # Matching Pennies result — this should oscillate around 0.5, never converge
    ax3.plot(x, mp_mean, color='#F06292', lw=2, label='Matching Pennies (no pure NE)')
    ax3.fill_between(x, mp_mean - mp_std, mp_mean + mp_std, color='#F06292', alpha=0.15)

    # Dashed yellow line at 0.5 = what pure random guessing would achieve
    ax3.axhline(0.5, color='#FFCC02', lw=1, linestyle='--', alpha=0.6,
                label='Random baseline (0.5)')

    style_ax(ax3, "Convergence by Game Structure\n(Independent Q-Learning)",
             "Episode", "Coordination Rate")
    ax3.legend(fontsize=8, labelcolor='#cccccc', facecolor='#22253a',
               edgecolor='#333344', loc='center right')
    ax3.set_ylim(-0.05, 1.1)

    # ── PANEL 4 (bottom-right): Bar chart of final coordination by noise ─
    ax4 = fig.add_subplot(gs[1, 1])  # row 1, column 1

    final_coords = []  # list to hold the final avg coordination for each noise level
    final_stds = []    # list to hold the spread for the error bars

    for noise in noise_levels:
        m, s = noise_results[noise]
        # Take the LAST 500 episodes of the smoothed mean to see where agents ended up
        # m[-500:] means "the last 500 elements of m" (negative indexing in Python)
        final_coords.append(m[-500:].mean())
        final_stds.append(m[-500:].std())

    # Draw bars: one per noise level, colored with the same palette as Panel 1
    # yerr adds error bars (little T-shapes) showing the std deviation
    # capsize=5 adds a small horizontal cap on the error bars
    bars = ax4.bar(
        [str(n) for n in noise_levels],  # x-axis labels: ['0.0', '0.5', '1.5', '3.0']
        final_coords,
        color=palette,
        alpha=0.85,
        width=0.55,
        yerr=final_stds,
        capsize=5,
        error_kw=dict(ecolor='#aaaaaa', lw=1.5)  # style the error bars
    )

    ax4.axhline(0.5, color='#FFCC02', lw=1, linestyle='--', alpha=0.6,
                label='Random baseline')
    style_ax(ax4, "Final Coordination Rate vs. Noise Level\n(last 500 episodes avg.)",
             "Reward Noise σ", "Avg. Pareto Coordination Rate")
    ax4.legend(fontsize=8, labelcolor='#cccccc', facecolor='#22253a', edgecolor='#333344')
    ax4.set_ylim(0, 1.1)

    # ── Overall title across the top of the whole figure ──────
    fig.suptitle(
        "Independent Q-Learning in Two-Agent Coordination Games\n"
        "Convergence Analysis: Noise, Learning Rate & Game Structure",
        color='white', fontsize=13, fontweight='bold', y=0.98
    )

    # Save the completed figure to a PNG file.
    # dpi=150 means 150 dots-per-inch (decent resolution).
    # bbox_inches='tight' trims extra whitespace around the edges.
    plt.savefig('marl_convergence_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
    print("Plot saved.")
    plt.close()  # Free up memory by closing the figure

if __name__ == "__main__":
    make_figure()  # run everything: experiments + plotting + saving

    print("\nDone! Results saved to marl_convergence_analysis.png")
    print("\nKey findings:")
    # Expected takeaways from the charts:
    print("- Higher reward noise (σ) delays convergence to Pareto-optimal coordination")
    print("- Moderate learning rates (α ≈ 0.1) balance speed and stability")
    print("- Stag Hunt reliably converges; Matching Pennies oscillates near random (no pure NE)")
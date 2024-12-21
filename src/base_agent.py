from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import max_dict


class BaseAgent(ABC):
    """
    Constructor for the base class, initializes key parameters for the RL agent.

      :param env: The environment in which the agent will interact (e.g., OpenAI Gym environment).
      :param alpha: Learning rate for updating Q-values or policy.
      :param gamma: Discount factor used in reinforcement learning, which determines the importance
                    of future rewards.
      :param epsilon: The initial exploration-exploitation trade-off ratio.
                      High epsilon means more exploration.
      :param epsilon_decay: Decay rate for epsilon during training, controls how quickly exploration
                             decreases over time.
      :param epsilon_min: Minimum epsilon value to avoid zero exploration during training.
      :param policy_strategy: The policy used for action selection, default is "epsilon_greedy".
                              "epsilon_soft" is also available.
    """

    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.5,
        epsilon=0.9,
        epsilon_decay=0.9,
        epsilon_min=0.05,
        policy_strategy="epsilon_greedy",
        plot=False
    ) -> None:
        """
        Constructor for the base class
        """
        self.env = env
        self.alpha = alpha  # Learning Rate
        self.gamma = gamma  # Discount Factor
        self.epsilon = epsilon  # Exploration Ratio
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.policy_strategy = policy_strategy  # ["epsilon_greedy", ["epsilon_soft"]

        self.Q_values = {}
        self.policy = self.make_policy(
            self.Q_values,
            self.epsilon,
            self.env.action_space.n,
            method=self.policy_strategy,
        )

        self.episode_df = []
        columns = ["episode", "step", "state", "action", "reward", "next_state", "next_action", "alpha", "epsilon", "gamma"]
        self.history = pd.DataFrame(columns=columns)
        self.plot = plot



    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def plot_training(self):
        pass

    def initialize_plot(self):
        """
        Initializes the plot and sets up interactive mode to allow non-blocking updates during training.
        """
        # plt.ion()  # Enable interactive mode
        subplots = 3
        self.fig, self.ax = plt.subplots(subplots, figsize=(8,4*subplots))  # Create a new figure and axis
        self.ax[0].set_title("Total Rewards per Episode")
        self.ax[0].set_xlabel("Episode")
        self.ax[0].set_ylabel("Total Reward")
        self.ax[1].set_title("Timesteps per Episode")
        self.ax[1].set_xlabel("Episode")
        self.ax[1].set_ylabel("Timesteps")

        plt.tight_layout()

        return self.fig, self.ax

    def update_plot(self):
        """
        Updates the plot with the new total_rewards data.

        Args:
            ax: The axis object where the plot is drawn.
            total_rewards: List of total rewards for each episode.
        """
        rewards_by_episode = self.history.groupby("episode")["reward"].sum()
        timesteps_by_episode = self.history.groupby("episode")["step"].sum()

        alpha_by_episode = self.history.groupby("episode")["alpha"].mean()  # Average gamma per episode
        gamma_by_episode = self.history.groupby("episode")["gamma"].mean()  # Average gamma per episode
        epsilon_by_episode = self.history.groupby("episode")["epsilon"].mean()  # Average epsilon per episode

        for ax in self.ax:
            ax.clear()

        self.ax[0].set_title("Total Rewards per Episode")
        self.ax[0].set_xlabel("Episode")
        self.ax[0].set_ylabel("Total Reward")
        
        self.ax[1].set_title("Timesteps per Episode")
        self.ax[1].set_xlabel("Episode")
        self.ax[1].set_ylabel("Timesteps")

        self.ax[2].set_title("Gamma and Epsilon per Episode")
        self.ax[2].set_xlabel("Episode")
        self.ax[2].set_ylabel("Value")

        self.ax[0].plot(rewards_by_episode)  # Plot the new data
        self.ax[1].plot(timesteps_by_episode)  # Plot the new data
        
        self.ax[2].plot(alpha_by_episode, label="Alpha", color="green", linestyle="-.")
        self.ax[2].plot(gamma_by_episode, label="Gamma", color="orange", linestyle="--")
        self.ax[2].plot(epsilon_by_episode, label="Epsilon", color="red", linestyle="-")
        self.ax[2].legend(loc="upper right")  # Add a legend to differentiate lines

        plt.draw()  # Redraw the plot
        plt.pause(0.1)  # Pause briefly to allow updates

    def make_policy(self, Q, epsilon, num_actions, method="epsilon_greedy"):
        """
        Generates a policy based on Q using epsilon-greedy or epsilon-soft.

        Args:
            Q: Dictionary where each state maps to Q-values Q(s, a).
            epsilon: Exploration level (between 0 and 1).
            num_actions: Number of possible actions.
            method: Method to use ('epsilon_greedy' or 'epsilon_soft').

        Returns:
            policy_fn: Function that, given a state, returns the action or probabilities.
        """
        if method not in ["epsilon_greedy", "epsilon_soft"]:
            raise ValueError(
                "Invalid method. Choose 'epsilon_greedy' or 'epsilon_soft'."
            )

        def policy_fn(state):
            q_values = Q[state]

            if method == "epsilon_greedy":
                if np.random.rand() < epsilon:
                    # Exploration: Select a random action
                    return np.random.choice(num_actions)
                else:
                    # Exploitation: Select the action with the highest Q-value
                    return max_dict(q_values)[0]

            elif method == "epsilon_soft":
                # Initial probabilities: All actions have probability epsilon / num_actions
                probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
                # Increase the probability of the best action
                best_action = max_dict(q_values)[0]
                probs[best_action] += 1.0 - epsilon
                # Ensure that the probabilities sum to 1
                probs = probs / np.sum(probs)
                # Select a random action with these probabilities
                action = np.random.choice(len(probs), p=probs)  # Probs do not sum 1
                return action

        return policy_fn

    def print_policy(self, policy, height, width):
        switch_action = {
            0: "Left",
            1: "Down",
            2: "Right",
            3: "Up",
            4: "Stay",
        }

        for row in range(height):
            print(
                "------------------------------------------------------------------------------------------"
            )
            for col in range(width):
                state = row * width + col  # Mapeo de (row, col) a estado lineal
                act = policy(state)
                action_str = switch_action[act]
                print("  %s  |" % action_str, end="")
                # print("  %s  |" % act, end="")
            print("")

        print(
            "------------------------------------------------------------------------------------------"
        )

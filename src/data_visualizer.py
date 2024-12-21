###
### Include functions to plot and display information via graphs or terminal prints.
###
###
import matplotlib.pyplot as plt
import numpy as np


def print_policy(policy, height, width):
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
            arr = np.array(
                policy(state)
            )  # Obtenemos las probabilidades para el estado actual
            act = int(np.argmax(arr))  # Elegimos la acción con la máxima probabilidad
            action_str = switch_action[act]
            print("  %s  |" % action_str, end="")
        print("")

    print(
        "------------------------------------------------------------------------------------------"
    )


def initialize_plot():
    """
    Initializes the plot and sets up interactive mode to allow non-blocking updates during training.
    """
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()  # Create a new figure and axis
    ax.set_title("Total Rewards per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    return fig, ax


def update_plot(ax, total_rewards):
    """
    Updates the plot with the new total_rewards data.

    Args:
        ax: The axis object where the plot is drawn.
        total_rewards: List of total rewards for each episode.
    """
    ax.clear()  # Clear the previous plot
    ax.plot(total_rewards)  # Plot the new data
    ax.set_title("Total Rewards per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    plt.draw()  # Redraw the plot
    plt.pause(0.1)  # Pause briefly to allow updates

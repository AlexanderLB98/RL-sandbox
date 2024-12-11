# RL-sandbox
Exploring and implementing diverse reinforcement learning algorithms, experimenting with Gymnasium environments, and creating custom scenarios to push the boundaries of agent learning.


## Algorithm: Monte-Carlo on policy first visit algorithm
We implemented the Monte Carlo On-Policy First Visit algorithm in . The algorithm is designed to estimate the value function based on first visits to state-action pairs during episodes. The key steps involved include:

- Generating episodes using the current policy.
- Updating the value function for each state-action pair based on the rewards received.
- Improving the policy over time by following the action that maximizes the expected return from each state.

The algorithm was tested in the Gym Gridworld environment, which consists of a grid where the agent must navigate from a start state to a goal state. The agent receives rewards for reaching the goal, and penalties for other actions.

## Experiment: Testing on Gym Gridworlds

The Monte-Carlo On-Policy First Visit algorithm was trained and tested in a standard [Gridworld](https://github.com/podondra/gym-gridworlds/tree/master?tab=readme-ov-file) environment. The agent's objective is to reach the goal state, and the rewards are structured accordingly. The environment is configured to provide a negative reward for each step, incentivizing the agent to reach the goal as efficiently as possible.
Observations and Results

After training the agent for 100 episodes, the algorithm seemed to get "stuck" early in the grid, consistently selecting suboptimal actions that resulted in a less negative reward. This behavior suggests that the agent finds it more beneficial to stay near the beginning of the grid, rather than moving towards the end where the goal is located. This can be attributed to the structure of the reward function, which might be providing a less negative reward for staying near the start compared to the reward for moving towards the goal.

- Issue observed: The agent appears to settle into a local optimum, where it prefers staying close to the starting point rather than progressing toward the goal.
- Hypothesis: The algorithm may be underexploring the environment, leading it to prefer suboptimal actions early on due to a lack of exploration of the state space. This could be mitigated by increasing the number of training episodes or tweaking the reward structure to better reflect the desirability of reaching the goal.
from src.sarsa import sarsa


import gym_gridworlds
import gymnasium as gym



if __name__ == "__main__":
    
    
    num_episodes = 2000
    epsilon_0 = 0.5  # Epsilon_0
    epsilon_decay = 0.9
    epsilon_min = 0.05
    # Update epsilon for next episode
    epsilon = max(epsilon_0 * epsilon_decay, 0.01)
    gamma = 1

    env = gym.make("Gym-Gridworlds/Ex2-4x4-v0", render_mode="human")

    env.reset()
    env.render()
    print("Action space is {} ".format(env.action_space))
    print("Observation space is {} ".format(env.observation_space))

    ######################## SOLUCIÓN ###########################
    
    

    
    model = sarsa(env)
    model.train()

    env.close()
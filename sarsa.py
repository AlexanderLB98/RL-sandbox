import numpy as np
import pandas as pd

import gym_gridworlds
import gymnasium as gym
from collections import defaultdict

from src.policies import make_policy
from src.exploration_strategy import epsilon_greedy
from src.utils import print_policy, max_dict



def sarsa(env, n_episodes = 10, alpha = 0.5, gamma = 0.5, epsilon = 0.9, epsilon_decay = 0.999, epsilon_min = 0.05, plot=False, policy_method = "epsilon_greedy"):
    """
    Implementa el algoritmo SARSA, dentro de on policy Time Difference.

    Args:
        env: El entorno de OpenAI Gym.
        num_episodes: Número de episodios para entrenar el agente.
        alpha/learning_rate: Tasa de aprendizaje.
        gamma: Factor de descuento.
        epsilon: Parámetro de exploración inicial para la política epsilon-soft.
        epsilon_decay: Tasa de decaimiento de epsilon por episodio.
        epsilon_min: Valor mínimo de epsilon.
        plot[Bool]: mostrar figuras
    Returns:
        Q: Diccionario que mapea estado -> valores de acción.
        policy: Política final derivada de Q.
    """

    # 1. Inicializar arbitráreamente Q(s, a), excepto Q(terminal,) = 0
    # Inicializo una función Q que inicializa a cero cualquier input 
    # si no existe.
    # Q = defaultdict(
    #     lambda: np.zeros(env.action_space.n)
    # )  
    policy_method = "epsilon_soft"

    Q = {}
    for s in range(env.observation_space.n):
        Q[s] = {}
        for a in range(env.action_space.n):
            Q[s][a] = 0
    print(Q)

    rewards = {}
    
    columns = ["episode", "step", "state", "action", "reward", "next_state", "next_action"]
    history = pd.DataFrame(columns=columns)

    # Crear política epsilon-greedy
    policy = make_policy(Q, epsilon, env.action_space.n, method=policy_method)
        
    # 2. Loop por episodio
    for episode in range(n_episodes):
        i = 0 # counter
        obs = env.reset()[0]
        done = False
        rewards[episode] = 0
        episode_df = []

        # 3. Loop por step
        while not done:
            # action = epsilon_greedy(q_values=Q, epsilon=epsilon, n_actions=env.action_space.n)
            action1 = policy(state=obs)
            new_obs, reward, terminated, truncated, info = env.step(action1)
            done = terminated or truncated
            if not done:
                action2 = policy(state=new_obs)
                q_next = Q[new_obs][action2]
            else:
                q_next = 0

            
            rewards[episode] += reward
            i += 1
            
            episode_df.append([episode, i, obs, action1, reward, new_obs, action2])
            # history.append([obs, action1, reward, new_obs, action2]) # SARSA
            Q[obs][action1] = Q[obs][action1] + alpha * (reward + gamma * q_next - Q[obs][action1])
            
            # Update policy every timestep (on policy)
            policy = make_policy(Q, epsilon, env.action_space.n, method=policy_method)
            
            obs = new_obs
            action1 = action2

            

        
        episode_df = pd.DataFrame(episode_df, columns=columns)
        history = pd.concat([history, episode_df], ignore_index=True)
        print(Q)
        print(f"Episodio {episode} completado en {i} pasos con recompensa {rewards[episode]}.")
        print(f"Last 5 actions:  { list(history['action'].iloc[-5:])}")
        print(f"Q[0]: {Q[0]}")
        print_policy(policy, 4,4)

        # Update epsilon for next episode
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    
    return Q, policy, history

if __name__ == "__main__":
    
    num_episodes = 10000
    learning_rate = 0.2
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
    Q = sarsa(env, n_episodes=num_episodes)

    env.close()


    print(Q)
    action = epsilon_greedy(q_values=Q, epsilon=0.9, n_actions=env.action_space.n)
    print(action)
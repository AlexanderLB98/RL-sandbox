from collections import defaultdict

import gym_gridworlds
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from src.data_visualizer import initialize_plot, print_policy, update_plot
from src.policies import make_policy


def mc_on_policy_first_visit_epsilon_soft(
    env, num_episodes, epsilon, epsilon_min, epsilon_decay, gamma, plot=False
):
    # Extra Source: https://github.com/RubenPerezUoc/Aprendizaje-por-refuerzo-Universitat-Oberta-Catalunya/blob/main/M05/M05-E1%20-%20MonteCarlo%20en%20WindyGridWorld%20%5Bes%5D.ipynb

    # Initialize Returns
    # Almacenamos la suma y el número de retornos de cada estado para calcular
    # el promedio. Podríamos usar un array para guardar todos los retornos
    # (como en el libro) pero es ineficiente en términos de memoria.
    policy_method = "epsilon_soft"
    policy_method = "epsilon_greedy"
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # vars for plotting
    reward_ep = []

    if plot:
        fig, ax = initialize_plot()

    # Initialize Q[state][action] function (Maps states and actions)
    Q = defaultdict(
        lambda: np.zeros(env.action_space.n)
    )  # Inicializo política aleatoria (map state with action)
    # Initialize epsilon-soft arbitrary policy
    policy = make_policy(Q, epsilon, env.action_space.n, method=policy_method)

    # Loop over all episones
    for n in range(num_episodes):
        # Inicializamos el entorno
        G = 0  #

        # Generamos un episodio y lo almacenamos
        # Un episodio es un array de las tuplas (state, action, reward)
        episode = []
        obs, info = env.reset()
        t, total_reward, terminated, truncated = 0, 0, False, False
        done = terminated or truncated
        while not done:
            policy = make_policy(Q, epsilon, env.action_space.n, method=policy_method)
            #probs = policy(obs)
            # action = np.random.choice(len(probs), p=probs)  # Probs do not sum 1
            action = policy(state=obs)
            # Ejecutar la acción y esperar la respuesta del entorno
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # if n % 1000 == 0:
            # Imprimir time-step
            # print("Episode {}. Step {}. Action: {} -> Obs: {} and reward: {}. Terminated {}. Truncated {}.".format(n,t+1, switch_action[action], new_obs, reward, terminated, truncated))

            # Encontramos todos los pares (estado, acción) que hemos visitado en este episodio
            # Para ello recorremos la lista episode, cogiendo lo que nos interesa
            # Convertimos cada estado en una tupla para poder usarlo como clave del diccionario
            # sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
            sa_in_episode = [(x[0], x[1]) for x in episode]
            for state, action in sa_in_episode:
                sa_pair = (state, action)
                # Encontramos la primera aparición del par (estado, acción) en el episodio
                first_occurence_idx = next(
                    i for i, x in enumerate(episode) if x[0] == state and x[1] == action
                )

                # Sumamos todas las recompensas desde la primera aparición
                G = sum(
                    [
                        x[2] * (gamma**i)
                        for i, x in enumerate(episode[first_occurence_idx:])
                    ]
                )
                # Calculamos el retorno promedio para este estado en todos los episodios muestreados
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
                # La política se mejora implícitamente al ir cambiando los valores de Q

            # Update epsilon for next episode
            epsilon = max(epsilon * epsilon_decay, 0.01)
            # Update policy for next episode
            policy = make_policy(Q, epsilon, env.action_space.n, method=policy_method)

            # Actualizar variables
            episode.append((obs, action, reward, terminated, truncated, info))
            obs = new_obs
            total_reward += reward
            t += 1
            # time.sleep(0.5) #Se añade para ralentizar el renderizado y poder apreciar los movimientos del agente
        print("Episodio {} completado. Retorno = {}".format(n, G))
        reward_ep.append(total_reward)
        # plot_rewards(reward_ep)
        if plot:
            update_plot(ax, reward_ep)
        if n % 10 == 0:
            print_policy(policy, 4, 4)

    print_policy(policy, 4, 4)

    if plot:
        # Turn off interactive mode after plotting and before blocking the script with plt.show()
        plt.ioff()  # Disable interactive mode
        plt.show()  # Block execution until the plot window is closed

    return Q, policy


if __name__ == "__main__":

    num_episodes = 200
    epsilon_0 = 0.5  # Epsilon_0
    epsilon_decay = 0.999
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
    Q_mc, policy = mc_on_policy_first_visit_epsilon_soft(
        env, num_episodes, epsilon_0, epsilon_min, epsilon_decay, gamma, plot=True
    )

    env.close()

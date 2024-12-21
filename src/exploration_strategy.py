import gymnasium as gym

import numpy as np
from collections import defaultdict

def epsilon_greedy(q_values, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions) # Exploracion
    return np.argmax(q_values)    # Explotacion


def make_epsilon_soft_policy(Q, epsilon, num_Actions=5):
    """
    Crea una política epsilon-soft basada en una función de valor de acción Q y epsilon

    Args:
        Q: Un diccionario cuya correspondencia es state -> action-values.
           Cada valor es un array de numpy de longitud num_Actions (see below)
        epsilon: La probabilidad de seleccionar una acción aleatoria (float entre 0 and 1).
        num_Actions: Número de acciones del entorno. En este caso será 5.

    Returns:
        Una función que tome como argumento la observación y devuelva como resultado
        las probabilidades de cada acción como un array de numpy de longitud num_Actions.
    """

    def policy_fn(observation) -> np.ndarray:

        A = np.ones(num_Actions, dtype=float) * epsilon / num_Actions
        best_action = np.argmax(Q[observation])
        A[best_action] += (
            1.0 - epsilon
        ) / num_Actions  # Aumenta la probabilidad de la mejor accion
        # En teoría para que sea epsilon soft debería dividirse la linea anterior entre num_actions, pero entonces la
        # prob no me da 1 y habŕía que normalizar.

        # Normalizar las probabilidades para que sumen 1
        sum_probs = np.sum(A)
        normalized_probs = A / sum_probs

        return normalized_probs  # Action probability
 
    return policy_fn
    
if __name__ == "__main__":
    
    env = gym.make("Gym-Gridworlds/Ex2-4x4-v0", render_mode="human")

    Q = defaultdict(
        lambda: np.zeros(env.action_space.n)
    )  

    
    action = epsilon_greedy(q_values=Q, epsilon=0.9, n_actions=env.action_space.n)
"""
Métodos de selección de acción según diferentes políticas.
"""

import numpy as np

from src.utils import max_dict

def make_policy(Q, epsilon, num_actions, method="epsilon_greedy"):
    """
    Genera una política basada en Q usando epsilon-greedy o epsilon-soft.

    Args:
        Q: Diccionario donde cada estado mapea a los valores Q(s, a).
        epsilon: Nivel de exploración (0 a 1).
        num_actions: Número de acciones posibles.
        method: Método a usar ('epsilon_greedy' o 'epsilon_soft').

    Returns:
        policy_fn: Función que, dado un estado, devuelve la acción o las probabilidades.
    """
    if method not in ["epsilon_greedy", "epsilon_soft"]:
        raise ValueError("Invalid method. Choose 'epsilon_greedy' or 'epsilon_soft'.")

    def policy_fn(state):
        q_values = Q[state]

        if method == "epsilon_greedy":
            if np.random.rand() < epsilon:
                # Exploración: Selecciona una acción aleatoria
                return np.random.choice(num_actions)
            else:
                # Explotación: Selecciona la acción con mayor valor Q
                # return np.argmax(q_values)
                return max_dict(q_values)[0]

        elif method == "epsilon_soft":
            # Probabilidades iniciales: Todas las acciones tienen probabilidad epsilon / num_actions
            probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
            # Incrementa la probabilidad de la mejor acción
            # best_action = np.argmax(q_values) Estoy usando np con un dict
            # best_action = max(q_values, key=q_values.get)
            best_action = max_dict(q_values)[0]
            # print(f'Best action: {best_action}')
            probs[best_action] += (1.0 - epsilon)
            # Asegura que las probabilidades sumen 1
            probs = probs / np.sum(probs)
            # Selecciona una acción aleatoria con estas probs
            action = np.random.choice(len(probs), p=probs)  # Probs do not sum 1
            return action

    return policy_fn


"""Ejemplo de uso"""
if __name__ == "__main__":
    Q = {"state_1": [2.0, 1.5, 3.2, 0.5, 1.0]}  # Diccionario de Q
    epsilon = 0.3
    num_actions = 5

    # Crear política epsilon-greedy
    epsilon_greedy_policy = make_policy(Q, epsilon, num_actions, method="epsilon_greedy")

    # Seleccionar una acción para el estado 'state_1'
    action = epsilon_greedy_policy("state_1")
    print(f"Acción seleccionada (Epsilon-greedy): {action}")

    # Crear política epsilon-soft
    epsilon_soft_policy = make_policy(Q, epsilon, num_actions, method="epsilon_soft")

    # Obtener las probabilidades para el estado 'state_1'
    action_probs = epsilon_soft_policy("state_1")
    print(f"Acción seleccionada (Epsilon-soft) {action_probs}")

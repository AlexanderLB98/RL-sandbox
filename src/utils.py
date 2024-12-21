import numpy as np

class DotDict(dict):
    """dot.notation access to dictionary attributes.
    Ref: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val 
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_policy(policy, height, width):
    switch_action = {
        0: "Left",
        1: "Down",
        2: "Right",
        3: "Up",
        4: "Stay",
    }
    
    for row in range(height):
        print("------------------------------------------------------------------------------------------")
        for col in range(width):
            state = row * width + col      # Mapeo de (row, col) a estado lineal
            act = policy(state)
            #arr = np.array(policy(state))  # Obtenemos las probabilidades para el estado actual
            #act = int(np.argmax(arr))      # Elegimos la acción con la máxima probabilidad
            action_str = switch_action[act]
            print("  %s  |" % action_str, end="")
            # print("  %s  |" % act, end="")
        print("")

    print("------------------------------------------------------------------------------------------")


def max_dict(d):
    """
    retorna el argmax (key) y max (value) 
    de un diccionario, o de un ndarray
    """
    if isinstance(d, dict):
        max_key = None
        max_val = float('-inf')
        for k, v in d.items():
            if v > max_val:
                max_val = v
                max_key = k
        return max_key, max_val
    elif isinstance(d, np.ndarray):
        max_val = np.max(d)
        max_key = np.argmax(d)
        return max_key, max_val
    else:
        raise TypeError("La entrada debe ser un diccionario o un numpy array.")

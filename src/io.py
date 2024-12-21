import pickle

import pandas as pd
import yaml

from src.utils import DotDict


def load_config(config_path):
    with open(config_path, 'r') as file:
        return DotDict(yaml.safe_load(file))

def save_figure(fig, filename, dpi=300, file_format="png", tight_layout=True, transparent=False):
    """
    Saves a Matplotlib figure to a file with the specified parameters.

    Args:
        fig (matplotlib.figure.Figure): The Matplotlib figure object to save.
        filename (str): The path or filename (with or without extension) to save the figure.
        dpi (int): The resolution of the saved figure in dots per inch (default: 300).
        file_format (str): The format to save the figure (e.g., 'png', 'pdf', 'svg', 'jpg').
        tight_layout (bool): Whether to apply tight layout to the figure before saving (default: True).
        transparent (bool): Whether to save the figure with a transparent background (default: False).

    Returns:
        None
    """
    # Ensure the filename has the correct extension
    if not filename.endswith(f".{file_format}"):
        filename += f".{file_format}"

    # Apply tight layout if required
    if tight_layout:
        fig.tight_layout()

    # Save the figure with the specified options
    fig.savefig(
        filename, 
        dpi=dpi, 
        format=file_format, 
        bbox_inches="tight" if tight_layout else None, 
        transparent=transparent
    )
    print(f"Figure saved as {filename}")

def save_object(object, filename):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    
    # Save the model to the specified file
    with open(filename, "wb") as file:
        pickle.dump(object, file)
    
    print(f"Object saved as {filename}")

def load_object(filename):
    try:
        with open(filename, "rb") as file:
            object = pickle.load(file)
        print(f"Object loaded from {filename}")
        return object
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        raise
    except pickle.UnpicklingError:
        print(f"Error: The file '{filename}' could not be unpickled. Ensure it is a valid pickle file.")
        raise

# def load_config(filename):
#     try:
#         with open(filename, "r") as file:
#             config = yaml.safe_load(file)
#         print(f"Configuration loaded from {filename}")
#         return config
#     except FileNotFoundError:
#         print(f"Error: The file '{filename}' does not exist.")
#         raise
#     except yaml.YAMLError as e:
#         print(f"Error: The file '{filename}' contains invalid YAML syntax. Details: {e}")
#         raise

def load_dataframe(filename: str, delimiter: str = ';') -> pd.DataFrame:
    try:
        df = pd.read_csv(filename, sep=delimiter)
        print(f"Data loaded from {filename}")
        return df
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        raise

def save_dataframe(df, filename, delimiter=";"):
    try:
        df.to_csv(filename, index=False, sep=";")
        print(f"DataFrame saved to {filename}.")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")
        raise
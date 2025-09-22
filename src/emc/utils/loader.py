import json
import os
import pickle

import numpy as np
import scipy.io

from emc.utils.paths import get_paths


def load_pickle_file(file_path):
    """
    Loads data from a pickle file.
    Args:
        file_path (str): The path to the pickle file.
    Returns:
        data: The data loaded from the pickle file.
    Raises:
        FileNotFoundError: If the file is not found.
        pickle.UnpicklingError: If the file cannot be unpickled.
        Exception: For any other unexpected errors.
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except pickle.UnpicklingError:
        print(f"Error: The file {file_path} could not be unpickled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_json_file(file_path):
    """
    Loads data from a JSON file.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        data: The data loaded from the JSON file.
    Raises:
        FileNotFoundError: If the file is not found.
        json.JSONDecodeError: If the file contains invalid JSON.
        Exception: For any other unexpected errors.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_cwru_sample_key(sample_id):
    """
    Maps a sample ID to a specific key used in the CWRU dataset.
    Args:
        sample_id (int): The sample ID.
    Returns:
        str: The corresponding key for the sample ID.
    """
    sample_id = 56 if sample_id == 3001 else sample_id
    sample_id = 57 if sample_id == 3002 else sample_id
    sample_id = 58 if sample_id == 3003 else sample_id
    sample_id = 59 if sample_id == 3004 else sample_id
    sample_id = 48 if sample_id == 3005 else sample_id
    sample_id = 49 if sample_id == 3006 else sample_id
    sample_id = 50 if sample_id == 3007 else sample_id
    return f"X{str(sample_id).zfill(3)}_DE_time"
    
def load_cwru_data(condition_sequence, points_per_condition):
    """
    Loads data from the CWRU dataset based on a sequence of conditions.
    Args:
        condition_sequence (list of str): A list of condition identifiers.
        points_per_condition (int): The number of data points to load per condition.
    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The loaded data.
            - change_points (list of int): Indices where conditions change in the data.
            - y_true (list of int): True labels for the data points.
    """
    cnd_id_map = {
        "L0-OK": 97,
        "L0-IR:07": 105,
        "L0-IR:14": 169,
        "L0-IR:21": 209,
        "L0-IR:28": 3001,
        "L1-OK": 98,
        "L1-IR:07": 106,
        "L1-IR:14": 170,
        "L1-IR:21": 210,
        "L1-IR:28": 3002,
        "L2-OK": 99,
        "L2-IR:07": 107,
        "L2-IR:14": 171,
        "L2-IR:21": 211,
        "L2-IR:28": 3003,
        "L3-OK": 100,
        "L3-IR:07": 108,
        "L3-IR:14": 172,
        "L3-IR:21": 212,
        "L3-IR:28": 3004
    }
    paths = get_paths()

    # sample_id_seq = [cnd_id_map[i] for i in condition_sequence]
    X = []
    change_points = [0]
    y_true = []
    for i, sample_cnd in enumerate(condition_sequence):
        sample_id = cnd_id_map[sample_cnd]
        data_path = os.path.join(paths["data"]["cwru"]["samples"], f"{sample_id}.mat")
        sample_data_raw = scipy.io.loadmat(data_path)
        if isinstance(points_per_condition, list):
            duration = points_per_condition[i]
        elif isinstance(points_per_condition, int):
            duration = points_per_condition
        sample_data = sample_data_raw[get_cwru_sample_key(sample_id)].flatten()[:duration]
        X.extend(sample_data.tolist())
        change_points.append(len(X))
        y_true.extend([i]*len(sample_data))
    X = np.array(X).reshape(-1, 1)
    return X, change_points, y_true

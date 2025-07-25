# Matéo begins
# This file contains functions to load data from pkl files or CSV files and convert them to a dictionary format, and plot comparisons of different runs. ERM data can also be loaded from pkl files.
# Beware of the variable names in the dictionnaries, as they are not standardized.
import os
import pickle
import csv
import numpy as np

import matplotlib.pyplot as plt


def load_or_create_pickle(data_path: str) -> dict:
    """
    Load data from pickle if it exists. Otherwise, read CSV, convert to a dict of lists,
    save as pickle, and return the dict.

    CSV is expected to have a header row. Each subsequent row is parsed according to the header.

    Returns:
        data_dict: dict where keys are column names and values are lists of column values (floats).
    """

    csv_path = data_path + ".csv"
    pickle_path = data_path + ".pkl"

    if os.path.exists(pickle_path):
        # Load existing pickle
        with open(pickle_path, "rb") as f:
            data_dict = pickle.load(f)
        return data_dict

    # Otherwise, read the CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Neither pickle nor CSV found for base path:\n  CSV: {csv_path}\n  PKL: {pickle_path}")

    data_dict = {}
    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file {csv_path} is empty.")

        # Initialize lists for each column
        for col in header:
            data_dict[col] = []

        # Parse each row
        for row_idx, row in enumerate(reader, start=2):
            if len(row) != len(header):
                # Skip malformed rows
                print(f"Warning: Skipping malformed CSV row {row_idx} ({len(row)} columns, expected {len(header)}).")
                continue
            for col, val in zip(header, row):
                try:
                    data_dict[col].append(float(val))
                except ValueError:
                    # If conversion fails (e.g., empty string), append NaN
                    data_dict[col].append(float("nan"))

    # Save to pickle for next time, adding that is is CSV-based
    data_dict["source"] = "csv"
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(data_dict, f)
    except IOError as e:
        print(f"Warning: Could not write pickle file {pickle_path}: {e}")

    if "alpha" in data_dict:
        # Rename 'alpha' to 'alphas' for consistency
        data_dict["alphas"] = data_dict.pop("alpha")
    return data_dict

def load_pickle_ERM(pickle_path, ERM_name: str = "ERM") -> dict:
    """
    Load a pickle file containing ERM data.

    Args:
        pickle_path (str or str list) : Path to the pickle file.

    Returns:
        data_dict: dict where keys are column names and values are lists of column values (floats).
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary in the pickle file, got {type(data)} instead.")
    
    data["source"] = "ERM"
    data["description"] = ERM_name
    if "alpha" in data:
        data["alphas"] = data.pop("alpha")
    required = {"alphas", "m_mean", "m_std", "q_mean", "q_std", "estim_err_mean", "estim_err_std", "gen_err_mean", "gen_err_std"} # here standard names are used.
    if not required.issubset(data):
        missing = required - set(data.keys())
        raise ValueError(f"'{pickle_path}' is missing required keys: {missing}. Ensure it contains the expected data structure.")

    #rename keys to match expected format m_mean -> m
    data["m"] = data.pop("m_mean") # This will allow for a search "m_std" in the pkl file from ERMs.
    data["q"] = data.pop("q_mean")
    data["estim_err"] = data.pop("estim_err_mean")
    data["gen_err"] = data.pop("gen_err_mean")

    return data


def load_multiple_runs(data_paths: list, run_names: list = None, print_keys = False, ERM_data_pickle_paths: list = None, ERM_names:list = None) -> dict:
    """
    Given a list of directories or file bases, each containing a CSV and/or PKL file,
    load all data into a dictionary of run_name -> data_dict.
    
    Expects CSV named "<base_name>.csv" and pickle "<base_name>.pkl" inside each directory.
    If a directory name is given, it will infer base_name from the single CSV or PKL inside;
    otherwise, if a file path is given, it uses that directly as base.

    Returns:
        runs_data: dict where each key is the run identifier (directory name or file basename)
                   and each value is the data_dict loaded via load_or_create_pickle.
    """
    if not run_names is None and len(run_names) != len(data_paths):
        raise ValueError("run_names must be provided if and only if it matches the length of data_paths.")
    
    runs_data = {}

    for idx_data, data_path in enumerate(data_paths):
        
        if os.path.exists(data_path +".csv") or os.path.exists(data_path +".pkl"):
            base = data_path
        else :
            base = os.path.splitext(data_path)[0]
            if not os.path.exists(base+".csv") and not os.path.exists(base+".pkl"):
                print(f"Warning: Neither CSV nor PKL found for base path '{base}'. Skipping.")
                continue
        
        dict_run_data = load_or_create_pickle(base)

        if not dict_run_data:
            print(f"Warning: No data loaded for base path '{base}'. Skipping.")
            continue

        if print_keys:
            print(f"Keys in run data for base '{base}':")
            print(dict_run_data.keys())

        if run_names is None:
            if not "description" in dict_run_data:
                runs_data[f"run_{idx_data}"] = dict_run_data
            else:
                if dict_run_data["description"] in runs_data:
                    print(f"Warning: Duplicate run description '{dict_run_data['description']}' found. Skipping.")
                    continue
                else:   
                    runs_data[dict_run_data["description"]] = dict_run_data
        else:
            if run_names[idx_data] in runs_data:
                print(f"Warning: Duplicate run name '{run_names[idx_data]}' found. Skipping.")
                continue
            runs_data[run_names[idx_data]] = dict_run_data

    if ERM_data_pickle_paths is not None:
        if ERM_names is None:
            ERM_names = [f"ERM_{i}" for i in range(len(ERM_data_pickle_paths))]
        elif len(ERM_names) != len(ERM_data_pickle_paths):
            raise ValueError("If ERM_names is provided, it must match the length of ERM_data_pickle_paths.")

        for ERM_path, ERM_name in zip(ERM_data_pickle_paths, ERM_names):
            try:
                ERM_data = load_pickle_ERM(ERM_path, ERM_name)
                if print_keys:
                    print(f"Keys in run data for path '{ERM_path}':")
                    print(ERM_data.keys())
                if ERM_name in runs_data:
                    print(f"Warning: Duplicate ERM name '{ERM_name}' found. Skipping.")
                    continue
                runs_data[ERM_name] = ERM_data
            except Exception as e:
                print(f"Error loading ERM data from {ERM_path}: {e}")
    return runs_data


def plot_comparison(runs_data: dict, x_key: str, y_key, logx: bool = True, logy: bool = False, title: str = None, save_plot: bool = False, save_dir: str = "./imgs/comparison_plots/", file_name: str = None):
    """
    Plot y_key vs x_key for multiple runs on the same figure.
    If y_key is a list, it will plot the values of the element of the corresponding run's data_dict against x_key. This allows for different naming conventions for y_keys. This is not the case for x_key, which must be the same for all runs.

    runs_data: dict of run_name -> data_dict
    x_key, y_key: column names to plot (must exist in all runs' data_dict)
    logx, logy: whether to use log scale on x or y axis
    title: optional title for the plot
    save_plot: if True, saves the plot to a file
    save_dir: directory to save the plot if save_plot is True
    save_name: name of the file to save the plot, if None it will use a default format based on x_key and y_key
    """
    plt.figure(figsize=(8, 6))
    for idx, (run_name, data) in enumerate(runs_data.items()):
        # check if y_key is a list of strings or a single string
        if isinstance(y_key, list):
            y_key_local = y_key[idx % len(y_key)]
        else:
            y_key_local = y_key
        if x_key not in data or y_key_local not in data:
            print(f"Warning: '{x_key}' or '{y_key_local}' not found in run '{run_name}'. Skipping.")
            continue
        x_vals = data[x_key]
        y_vals = data[y_key_local]
        if y_key_local + "_std" in data:
            y_std = data[y_key_local + "_std"]
            # Filter out NaNs
            filtered = [(x, y, s) for x, y, s in zip(x_vals, y_vals, y_std) if not (np.isnan(x) or np.isnan(y) or np.isnan(s))]
        else:
            filtered = [(x, y) for x, y in zip(x_vals, y_vals) if not (np.isnan(x) or np.isnan(y))]
        if not filtered:
            print(f"Warning: No valid data for '{run_name}' on keys {x_key}, {y_key_local}.")
            continue

        if y_key_local + "_std" in data:
            xs, ys, y_stds = zip(*filtered)
            plt.errorbar(xs, ys, yerr=y_stds, fmt='-', label=run_name)
        else:
            xs, ys = zip(*filtered)
            plt.plot(xs, ys, linestyle='-', label=run_name)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    y_key_final = y_key if isinstance(y_key, str) else y_key[0] # This is for the label of the y-axis.

    plt.xlabel(x_key)
    plt.ylabel(y_key_final)
    plt.title(title or f"{y_key_final} vs {x_key}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        if file_name is None:
            file_name = f"{y_key_final}_vs_{x_key}.png"
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# ------------ User inputs

base_paths = [ # Copy and paste the entire path WITHOUT CSV or PKL at the end. Both formats are supported, and a PKL file will be created if it does not exist.
    "./data/alpha_sweeps_Tukey_p_L2_decorrelated_noise/loss_param_1.0_reg_param_2.0_noise_0.10_1.00_0.10_0.00/alsw_alpha_min_10.0_max_300.0_n_pts_200", 
]
run_names = ["Tukey SE"] # Name as many runs as there are base_paths. If None, the run names will be inferred from the data_dicts.

ERM_data_pickle_paths = [ # Copy and paste the entire path to the ERM data pickle file WITH the extension. It must be a valid pickle file containing the expected data structure.
    "./data/alpha_sweeps_Tukey_p_L2_decorrelated_noise/loss_param_1.0_reg_param_2.0_noise_0.10_1.00_0.10_0.00/ERM_alpha_min_10.00_max_300.000_n_pts_25_reps_30_d_500.pkl",
]
ERM_names = ["ERM_Tukey"] # Name as many ERM data pickle files as there are ERM_data_pickle_paths. If None, the names will be inferred from the file names.

runs = load_multiple_runs(base_paths, 
                          run_names=run_names,
                          print_keys=True,
                          ERM_data_pickle_paths=ERM_data_pickle_paths,
                           ERM_names=ERM_names
                        )
print(f"Loaded runs: {list(runs.keys())}")

plot_comparison(runs, x_key="alphas", y_key=["ms","m"], # 
                logx=True, logy=True, save_plot=True, 
                save_dir="./imgs/comparison_plots/alpha_sweeps", 
                file_name="Tukey_m_vs_alphas_SE_ERM.png")

#Matéo ends

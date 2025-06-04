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

    return data_dict


def load_multiple_runs(data_paths: list, run_names: list = None, print_keys = False) -> dict:
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
    return runs_data


def plot_comparison(runs_data: dict, x_key: str, y_key, logx: bool = True, logy: bool = False, title: str = None, save_plot: bool = False, save_dir: str = "./imgs/comparison_plots/", file_name: str = None):
    """
    Plot y_key vs x_key for multiple runs on the same figure. If y_key is a list, it will plot each element of the corresponding run's data_dict against x_key.

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
        # Filter out NaNs
        filtered = [(x, y) for x, y in zip(x_vals, y_vals) if not (np.isnan(x) or np.isnan(y))]
        if not filtered:
            print(f"Warning: No valid data for '{run_name}' on keys {x_key}, {y_key_local}.")
            continue
        xs, ys = zip(*filtered)
        plt.plot(xs, ys, linestyle='-', label=run_name)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    y_key_final = y_key if isinstance(y_key, str) else y_key[0]

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



base_paths = [
    "./data/Tukey_evolved_lambda_opt_barrier/optimal_lambda_se_tukey_evolved_alpha_min_50_max_10000_n_alpha_pts_200_delta_in_0.1_delta_out_1.0_percentage_0.1_beta_0.0_tau_1.0_c_0.0", 
    "./data/alpha_sweeps_Tukey_L2_decorrelated_noise_opt_reg_param_for_excess_gen_error/loss_param_1.0_noise_0.10_1.00_0.10_0.00/alsw_alpha_min_50.0_max_10000.0_n_pts_200_min_reg_param_0.0",
]

runs = load_multiple_runs(base_paths, 
                          run_names=["Tukey", "Tukey_new"],
                          print_keys=True)
print(f"Loaded runs: {list(runs.keys())}")

plot_comparison(runs, x_key="alphas", y_key=["gen_error","excess_gen_error"], logx=True, logy=True, save_plot=True, save_dir="./imgs/comparison_plots/alpha_sweeps", file_name="Tukey_excess_gen_error_vs_alphas.png")

plot_comparison(runs, x_key="alphas", y_key=["reg_params_opt","opt_reg_params"], logx=True, logy=True, save_plot=True, save_dir="./imgs/comparison_plots/alpha_sweeps", file_name="Tukey_opt_reg_params_vs_alphas.png")

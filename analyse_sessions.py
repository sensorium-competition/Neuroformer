import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(
    "/mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/Neuroformer"
)
sys.path.append(
    "/mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/Neuroformer/neuroformer"
)

from neuroformer.analysis import calc_corr_psth, get_rates
from neuroformer.utils import process_predictions


def analyze_sessions(base_path):
    """
    Analyzes all session files in a given directory, calculates the Pearson
    correlation between true and predicted firing rates, and saves the
    results to CSV files.

    Args:
        base_path (str): The path to the directory containing the session .pkl files.
    """
    # Get a list of all session files (assuming they are .pkl files)
    root_path = base_path
    base_path = base_path + "/inference"
    try:
        session_files = [f for f in os.listdir(base_path) if f.endswith(".pkl")]
        if not session_files:
            print(f"No .pkl files found in '{base_path}'. Please check the directory.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{base_path}' was not found.")
        return

    tokenizer_path = root_path + "/tokenizer.pkl"
    with open(tokenizer_path, "rb") as f:
        tokenizers = pickle.load(f)

    # Loop over all session files
    for session_file in session_files:
        session_name = os.path.splitext(session_file)[0]
        file_path = os.path.join(base_path, session_file)
        tokenizer = tokenizers[session_name[14:]]

        print(f"Processing session: {session_name}")

        # try:
        # Load the inference results
        with open(file_path, "rb") as f:
            result = pickle.load(f)

        # Process predictions to get spike trains
        # Assuming 'pred', 'true', and 'dt' are keys in the loaded dictionary
        df_pred, df_true = process_predictions(
            result, tokenizer.stoi, tokenizer.itos, 0
        )

        # Calculate firing rates
        rates_true = get_rates(
            df_true,
            list(set(result["true"]).union(set(result["true"]))),
            result["Interval"],
        )
        rates_pred = get_rates(
            df_pred,
            list(set(result["true"]).union(set(result["true"]))),
            result["Interval"],
        )

        # Calculate Pearson correlation
        correlation_df = calc_corr_psth(
            rates_true,
            rates_pred,
            list(set(result["true"]).union(set(result["true"]))),
        )

        # Save the DataFrame to a CSV file
        output_filename = f"{session_name}_correlation.csv"
        output_path = os.path.join(base_path, output_filename)
        correlation_df.to_csv(output_path, index=False)

        print(f"  - Saved correlation data to: {output_filename}")
        print(f"  - Correlation Stats:")
        print(f"    - Min:", np.nanmin(correlation_df["pearson_r"].to_numpy()))
        print(f"    - Max:", np.nanmax(correlation_df["pearson_r"].to_numpy()))
        print(f"    - Mean:", np.nanmean(correlation_df["pearson_r"].to_numpy()))
        print("-" * 30)

        # except KeyError as e:
        #     print(f"  - Could not process {session_name}. Missing key in data: {e}")
        #     print("-" * 30)
        # except Exception as e:
        #     print(f"  - Could not process {session_name}. An error occurred: {e}")
        #     print("-" * 30)


if __name__ == "__main__":
    # ==============================================================================
    # IMPORTANT: Please change this path to the location of your inference results.
    # ==============================================================================
    path_to_sessions = "/mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/Neuroformer/models/train_8_1_sec"
    # ==============================================================================

    analyze_sessions(path_to_sessions)

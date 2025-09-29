import pickle

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def binned_spike_count(spike_times, bin_size, t_max):
    """
    Bins spike times into fixed-size bins and returns spike counts.

    Args:
        spike_times (np.ndarray): An array of absolute spike times.
        bin_size (float): The size of each time bin in seconds.
        t_max (float): The maximum time to consider for binning.

    Returns:
        tuple: A tuple containing:
            - binned_counts (np.ndarray): The spike counts for each bin.
            - bin_edges (np.ndarray): The edges of the time bins.
    """
    bins = np.arange(0, t_max + bin_size, bin_size)
    binned_counts, bin_edges = np.histogram(spike_times, bins=bins)
    return binned_counts, bin_edges


def align_and_analyze(result_file_path, bin_size=0.05):
    """
    Loads a result file, aligns ground truth and predicted spikes
    by time-binning, and performs correlation analysis.

    Args:
        result_file_path (str): The file path to the pickle file containing
                                the simulation results.
        bin_size (float): The size of the time bins for alignment.
    """
    # 1. Load the result dictionary from the pkl file
    with open(result_file_path, "rb") as f:
        result = pickle.load(f)

    # 2. Convert time differences to absolute times
    # The error indicates the data is in string format, so we convert to float first.
    try:
        true_times = np.cumsum(np.array(result["time"]).astype(float))
        predicted_times = np.cumsum(np.array(result["dt"]).astype(float))
    except ValueError as e:
        print(f"Error converting data to float: {e}")
        print("This often means there are non-numeric strings in the time data.")
        return

    true_ids = result["true"]
    predicted_ids = result["ID"]

    # Determine the maximum time from the ground truth data to ensure alignment
    t_max = np.max(true_times)

    # 3. Bin the ground truth data
    unique_true_ids = np.unique(true_ids)
    binned_true_counts = []
    for neuron_id in unique_true_ids:
        neuron_times = true_times[true_ids == neuron_id]
        counts, _ = binned_spike_count(neuron_times, bin_size, t_max)
        binned_true_counts.append(counts)

    binned_true_counts = np.array(binned_true_counts)

    # 4. Bin the predicted data
    unique_pred_ids = np.unique(predicted_ids)
    binned_pred_counts = []
    for neuron_id in unique_pred_ids:
        neuron_times = predicted_times[predicted_ids == neuron_id]
        counts, _ = binned_spike_count(neuron_times, bin_size, t_max)
        binned_pred_counts.append(counts)

    binned_pred_counts = np.array(binned_pred_counts)

    # 5. Align the two binned arrays
    aligned_pred_counts = np.zeros(binned_true_counts.shape)
    true_id_map = {id: i for i, id in enumerate(unique_true_ids)}
    for i, pred_id in enumerate(unique_pred_ids):
        if pred_id in true_id_map:
            aligned_pred_counts[true_id_map[pred_id]] = binned_pred_counts[i]

    # 6. Calculate Pearson Correlation
    true_flat = binned_true_counts.flatten()
    pred_flat = aligned_pred_counts.flatten()

    non_zero_indices = np.where((true_flat != 0) | (pred_flat != 0))
    true_aligned = true_flat[non_zero_indices]
    pred_aligned = pred_flat[non_zero_indices]

    if len(true_aligned) > 1:
        corr, p_value = pearsonr(true_aligned, pred_aligned)
        print(f"Pearson Correlation Coefficient (r): {corr}")
        print(f"P-value: {p_value}")
    else:
        print("Not enough data points to compute a meaningful correlation.")

    # 7. Create a DataFrame from the binned, aligned data
    df_aligned = pd.DataFrame(
        {"binned_true_counts": true_aligned, "binned_predicted_counts": pred_aligned}
    )
    print("\nAligned DataFrame:")
    print(df_aligned.head())
    print("...")
    print(df_aligned.tail())


# Example usage (assuming 'results_trial_sample-True...pkl' exists)
# Replace 'path/to/your/results_trial.pkl' with the actual file path
align_and_analyze(
    "models/test_7/inference/results_trial_dynamic26872-17-20-Video-021a75e56847d574b9acbcc06c675055_30hz.pkl"
)

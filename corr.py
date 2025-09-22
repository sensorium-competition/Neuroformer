# load pickle file
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def binned_spike_count(spike_times, bin_size):
    max_time = spike_times.max()
    # rolling window binning
    bins = np.arange(0, max_time + bin_size, bin_size)
    print(len(bins))
    counts, _ = np.histogram(spike_times, bins=bins)
    return counts, bins


pickle_file = "./models/test_7/inference/results_trial_dynamic26872-17-20-Video-021a75e56847d574b9acbcc06c675055_30hz.pkl"
result = pickle.load(open(pickle_file, "rb"))

gt_counts = {}
pred_counts = {}

df = pd.DataFrame(result)

# Convert spike times to seconds from microseconds if necessary
df["dt"] = pd.to_numeric(df["dt"])
df["dt"] = df["dt"] / 1e6

# Calculate cumulative time to create a continuous timeline
df["Time"] = df.groupby("true")["time"].cumsum()
df["Time_pred"] = df.groupby("DT")["dt"].cumsum()

# Loop through each neuron and get the spike counts
for neuron_id in set(df["true"]):
    neuron_activity = df[df["true"] == neuron_id]
    counts, bins = binned_spike_count(neuron_activity["Time"].values, bin_size=(1 / 30))
    gt_counts[neuron_id] = counts

for neuron_id in set(df["ID"]):
    neuron_activity = df[df["ID"] == neuron_id]
    counts, bins = binned_spike_count(
        neuron_activity["Time_pred"].values, bin_size=(1 / 30)
    )
    pred_counts[neuron_id] = counts

print(len(set(result["true"])), len(set(result["ID"])))

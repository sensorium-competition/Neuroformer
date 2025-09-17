# %%
import glob
import os
import pickle
import sys
from pathlib import Path, PurePath

path = Path.cwd()
parent_path = path.parents[1]
sys.path.append(str(PurePath(parent_path, "neuroformer")))
sys.path.append("neuroformer")

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuroformer.data_utils import NFDataloader, Tokenizer, round_n
from neuroformer.datasets import load_experanto
from neuroformer.model_neuroformer import load_model_and_tokenizer
from neuroformer.simulation import decode_modality, generate_spikes
from neuroformer.utils import (
    all_device,
    create_modalities_dict,
    get_attr,
    load_config,
    recursive_print,
    running_jupyter,
    set_seed,
)
from scipy.stats import pearsonr
from torch.utils.data.dataloader import DataLoader

parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

# set up logging
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from neuroformer.default_args import DefaultArgs, parse_args

if running_jupyter():  # or __name__ == "__main__":
    print("Running in Jupyter")
    args = DefaultArgs()
    # Using a config file that specifies multiple data paths
    args.config = "./configs/Experanto/mconf_all_30Hz_two.yaml"
    args.dataset = "experanto"
    args.ckpt_path = "./models/test_7"
    args.predict_modes = ["treadmill"]
else:
    print("Running in terminal")
    args = parse_args()

# SET SEED - VERY IMPORTANT
set_seed(args.seed)

print(f"CONTRASTIVE {args.contrastive}")
print(f"VISUAL: {args.visual}")
print(f"PAST_STATE: {args.past_state}")


config = load_config(args.config)
config, tokenizers, model = load_model_and_tokenizer(args.ckpt_path)


# %%
"""

-- DATA --
Loading multi-session experanto data.

"""

test_datasets = {}
for i, data_path in enumerate(config.data.paths):
    data, intervals, train_intervals, test_intervals, finetune_intervals, callback = (
        load_experanto(config, data_path)
    )
    # This part is added to handle the structure of the experanto data
    data["spikes"] = torch.round(data["spikes"]).type(torch.int8).numpy()
    data["stimulus"] = (
        data["stimulus"]
        .type(torch.float32)
        .unsqueeze(0)
        .squeeze()
        .numpy()
        .reshape(-1, 36, 64)
    )
    data["dilation"] = data["dilation"].type(torch.float32).numpy()
    data["d_dilation"] = data["d_dilation"].type(torch.float32).numpy()
    data["pupil_x"] = data["pupil_x"].type(torch.float32).numpy()
    data["pupil_y"] = data["pupil_y"].type(torch.float32).numpy()
    data["treadmill"] = data["treadmill"].type(torch.float32).numpy()
    data["session"] = data["session"]

    spikes_dict = {
        "ID": data["spikes"],
        "Frames": data["stimulus"],
        "Interval": intervals,
        "dt": config.resolution.dt,
        "id_block_size": config.block_size.id,
        "prev_id_block_size": config.block_size.prev_id,
        "frame_block_size": config.block_size.frame,
        "window": config.window.curr,
        "window_prev": config.window.prev,
        "frame_window": config.window.frame,
        "session": data["session"],
    }

    frames = {
        "feats": data["stimulus"],
        "callback": callback,
        "window": config.window.frame,
        "dt": config.resolution.dt,
    }
    modalities = (
        create_modalities_dict(data, config.modalities)
        if get_attr(config, "modalities", None)
        else None
    )

    # Create a separate NFDataloader for each session's test data
    test_datasets[data["session"]] = NFDataloader(
        spikes_dict,
        tokenizers[data["session"]],
        config,
        dataset=args.dataset,
        frames=frames,
        intervals=test_intervals,
        modalities=modalities,
    )


if config.gru_only:
    model_name = "GRU"
elif config.mlp_only:
    model_name = "MLP"
elif config.gru2_only:
    model_name = "GRU_2.0"
else:
    model_name = "Neuroformer"

CKPT_PATH = args.ckpt_path

# Define the parameters for generate_spikes
sample = True
top_p = 0.95
top_p_t = 0.95
temp = 1.0
temp_t = 1.0
frame_end = 0
true_past = args.true_past
get_dt = True
gpu = True
pred_dt = True


# --- INFERENCE LOOP PER SESSION ---
# Iterate over each session's NFDataloader and run inference
for session_name, test_dataset in test_datasets.items():
    print(f"--- Running inference for session: {session_name} ---")

    # The `test_dataset` is now a NFDataloader, not a NFCombinedDataset
    results_trial = generate_spikes(
        model,
        test_dataset,
        config.window.curr,
        config.window.prev,
        tokenizers[session_name],  # Pass the specific tokenizer for this session
        sample=sample,
        top_p=top_p,
        top_p_t=top_p_t,
        temp=temp,
        temp_t=temp_t,
        frame_end=frame_end,
        true_past=true_past,
        get_dt=get_dt,
        gpu=gpu,
        pred_dt=pred_dt,
        plot_probs=False,
    )

    # Create a unique filename for each session's results
    filename = f"results_trial_{session_name}.pkl"
    save_inference_path = os.path.join(CKPT_PATH, "inference")
    if not os.path.exists(save_inference_path):
        os.makedirs(save_inference_path)

    print(f"Saving inference results in {os.path.join(save_inference_path, filename)}")
    with open(os.path.join(save_inference_path, filename), "wb") as f:
        pickle.dump(results_trial, f)

# %%
model.load_state_dict(
    torch.load(os.path.join(CKPT_PATH, f"model.pt"), map_location=torch.device("cpu"))
)

if args.predict_modes is not None:
    # --- DECODING LOOP PER SESSION ---
    for session_name, test_dataset in test_datasets.items():
        print(f"--- Decoding modalities for session: {session_name} ---")
        behavior_preds = {}
        block_type = "behavior"
        block_config = get_attr(config.modalities, block_type).variables
        for mode in args.predict_modes:
            mode_config = get_attr(block_config, mode)
            behavior_preds[mode] = decode_modality(
                model,
                test_dataset,
                modality=mode,
                block_type=block_type,
                objective=get_attr(mode_config, "objective"),
            )
            filename = f"behavior_preds_{mode}_{session_name}.csv"
            save_inference_path = os.path.join(CKPT_PATH, "inference")
            if not os.path.exists(save_inference_path):
                os.makedirs(save_inference_path)
            print(
                f"Saving inference results in {os.path.join(save_inference_path, filename)}"
            )
            behavior_preds[mode].to_csv(os.path.join(save_inference_path, filename))

# The plotting part can be adjusted to load and visualize results from all sessions if needed.
# For now, it will plot the results of the last session processed.

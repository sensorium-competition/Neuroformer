import sys

sys.path.append("./neuroformer")

import itertools
import torch
import numpy as np
import pandas as pd
import pickle
import os
import json
from neuroformer.dataset import build_dataloader
from experanto.dataloaders import ChunkDataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# with open('/mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/Neuroformer/configs/config.json', 'r') as file:
#     loader_config = json.load(file)

# print('Create Dataloader.')
# train_dataloader, val_dataloader = build_dataloader(loader_config)
# print('Dataloader Created')

base_path = "/mnt/vast-react/projects/neural_foundation_model/"
data = [
    "upsampling_without_hamming_30.0Hz/dynamic21067-10-18-Video-512cdd91f251cf74fff81375926f4002_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic22846-10-16-Video-512cdd91f251cf74fff81375926f4002_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic23343-5-17-Video-512cdd91f251cf74fff81375926f4002_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic23656-14-22-Video-512cdd91f251cf74fff81375926f4002_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic23964-4-22-Video-512cdd91f251cf74fff81375926f4002_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic29156-11-10-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic29228-2-10-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic29234-6-9-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic29513-3-5-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "upsampling_without_hamming_30.0Hz/dynamic29514-2-9-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "test_upsampling_without_hamming_30.0Hz/dynamic26872-17-20-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "test_upsampling_without_hamming_30.0Hz/dynamic29623-4-9-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "test_upsampling_without_hamming_30.0Hz/dynamic29755-2-8-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "test_upsampling_without_hamming_30.0Hz/dynamic27204-5-13-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "test_upsampling_without_hamming_30.0Hz/dynamic29647-19-8-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "test_upsampling_without_hamming_30.0Hz/dynamic29515-10-12-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "test_upsampling_without_hamming_30.0Hz/dynamic29712-5-9-Video-021a75e56847d574b9acbcc06c675055_30hz"
]

paths = [base_path + path for path in data]
# paths = ['/mnt/vast-react/projects/neural_foundation_model/dynamic29513-3-5-Video-full', '/mnt/vast-react/projects/neural_foundation_model/dynamic29514-2-9-Video-full']
config_path = "experanto_chunck_dataset.yaml"

# Train and test yaml diff
# <         tier: oracle
# ---
# >         tier: train
# 12c12 # ignoring this diff for now
# <       sample_stride: 60
# ---
# >       sample_stride: 1
# 55c55
# <   shuffle: false
# ---
# >   shuffle: true

cfg = OmegaConf.load(config_path)
# num_samples = 100

for path in paths:
    data = {
        "spikes": [],
        "stimulus": [],
        "dilation": [],
        "d_dilation": [],
        "pupil_x": [],
        "pupil_y": [],
        "treadmill": [],
        "session": [],
    }
    print(f"Loading {path}")
    print("Create Dataset")
    dataset = ChunkDataset(path, **cfg.dataset)
    print("Dataset Created")

    print(len(dataset))
    print(dataset.__getitem__(0).keys())

    session = path.split("/")[-1]
    data["session"] = session

    num_samples = len(dataset)
    for i in tqdm(range(num_samples)):
        x = dataset.__getitem__(i)
        data["spikes"].append(x["responses"])
        data["stimulus"].append(x["screen"])
        data["dilation"].append(x["eye_tracker"][:, 0])
        data["d_dilation"].append(x["eye_tracker"][:, 1])
        data["pupil_x"].append(x["eye_tracker"][:, 2])
        data["pupil_y"].append(x["eye_tracker"][:, 3])
        data["treadmill"].append(x["treadmill"])

    data["spikes"] = torch.concat(data["spikes"], dim=0).T
    data["stimulus"] = torch.concat(data["stimulus"], dim=0).squeeze()
    data["dilation"] = torch.concat(data["dilation"], dim=0).squeeze()
    data["d_dilation"] = torch.concat(data["d_dilation"], dim=0).squeeze()
    data["pupil_x"] = torch.concat(data["pupil_x"], dim=0).squeeze()
    data["pupil_y"] = torch.concat(data["pupil_y"], dim=0).squeeze()
    data["treadmill"] = torch.concat(data["treadmill"], dim=0).squeeze()

    print("Data shapes:")
    print(data.keys())
    print(data["spikes"].shape)
    print(data["stimulus"].shape)
    print(data["dilation"].shape)
    print(data["d_dilation"].shape)
    print(data["pupil_x"].shape)
    print(data["pupil_y"].shape)
    print(data["treadmill"].shape)
    print(data["session"])

    # save the data dictionary as a pickle file
    # /mnt/vast-react/projects/neural_foundation_model/Neuroformer_data_format/30Hz/
    with open(f"data/Experanto/30Hz/{session}-{num_samples}.pkl", "wb") as f:
        pickle.dump(data, f)

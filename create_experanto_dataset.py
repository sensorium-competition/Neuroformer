import sys
sys.path.append('./neuroformer')

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

path = '/mnt/vast-react/projects/neural_foundation_model/dynamic29513-3-5-Video-full'
config_path = 'experanto_chunck_dataset.yaml'

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

print('Create Dataset')
dataset = ChunkDataset(path, **cfg.dataset)
print('Dataset Created')

print(len(dataset))
print(dataset.__getitem__(0).keys())

adsasd

data = {"spikes":[], "stimulus":[]}
num_samples = 2000
for i in tqdm(range(num_samples)):
    x = dataset.__getitem__(i)
    data["spikes"].append(x["responses"])
    data["stimulus"].append(x["screen"])

data["spikes"] = torch.concat(data["spikes"], dim=0).T
data["stimulus"] = torch.concat(data["stimulus"], dim=0).squeeze()

print(data.keys())
print(data["spikes"].shape)
print(data["stimulus"].shape)

# save the data dictionary as a pickle file
with open(f'data/Experanto/val_data-{num_samples}.pkl', 'wb') as f:
    pickle.dump(data, f)
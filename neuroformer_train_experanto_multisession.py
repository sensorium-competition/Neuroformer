# %%

import glob
import os
import sys
from pathlib import Path, PurePath

path = Path.cwd()
parent_path = path.parents[1]
sys.path.append(str(PurePath(parent_path, "neuroformer")))
sys.path.append("neuroformer")
sys.path.append(".")
sys.path.append("../")

import json
import math

import numpy as np
import pandas as pd
import torch
from neuroformer.data_utils import NFCombinedDataset, NFDataloader, Tokenizer, round_n
from neuroformer.dataset import build_dataloader
from neuroformer.datasets import load_experanto, load_V1AL, load_visnav
from neuroformer.model_neuroformer import Neuroformer, NeuroformerConfig
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils import (
    all_device,
    create_modalities_dict,
    dict_to_device,
    dict_to_object,
    get_attr,
    load_config,
    object_to_dict,
    recursive_print,
    running_jupyter,
    set_seed,
    update_object,
)
from neuroformer.visualize import set_plot_params
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from experanto.utils import LongCycler

parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"
# set up logging
import logging

import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from neuroformer.default_args import DefaultArgs, parse_args

if running_jupyter():  # or __name__ == "__main__":
    print("Running in Jupyter")
    args = DefaultArgs()
else:
    print("Running in terminal")
    args = parse_args()

# SET SEED - VERY IMPORTANT
set_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"CONTRASTIVE {args.contrastive}")
print(f"VISUAL: {args.visual}")
print(f"PAST_STATE: {args.past_state}")

# Use the function
if args.config is None:
    config_path = "./configs/NF_1.5/VisNav_VR_Expt/gru2_only_cls/mconf.yaml"
else:
    config_path = args.config
config = load_config(config_path)  # replace 'config.yaml' with your file path


# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""
train_datasets = {}
test_datasets = {}
finetune_datasets = {}
tokenizers = {}
for i, data_path in enumerate(config.data.paths):
    if args.dataset in ["lateral", "medial"]:
        device = torch.device("cpu")
        (
            data,
            intervals,
            train_intervals,
            test_intervals,
            finetune_intervals,
            callback,
        ) = load_visnav(
            args.dataset,
            config,
            selection=config.selection if hasattr(config, "selection") else None,
        )
    elif args.dataset == "V1AL":
        device = torch.device("cpu")
        (
            data,
            intervals,
            train_intervals,
            test_intervals,
            finetune_intervals,
            callback,
        ) = load_V1AL(config)
    elif args.dataset == "experanto":
        device = torch.device("cpu")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        (
            data,
            intervals,
            train_intervals,
            test_intervals,
            finetune_intervals,
            callback,
        ) = load_experanto(config, data_path)

        # if i == 1:
        #     data["spikes"] = torch.round(data["spikes"][:, :data['spikes'].shape[1]//4]).type(torch.int8).to(device).numpy()
        #     data["stimulus"] = data["stimulus"][:data['dilation'].shape[0]//4, :, :, :].type(torch.float32).unsqueeze(0).to(device).squeeze().numpy().reshape(-1, 36, 64)
        #     data["dilation"] = data["dilation"][:data['dilation'].shape[0]//4].type(torch.float32).to(device).numpy()
        #     data["d_dilation"] = data["d_dilation"][:data['d_dilation'].shape[0]//4].type(torch.float32).to(device).numpy()
        #     data["pupil_x"] = data["pupil_x"][:data['pupil_x'].shape[0]//4].type(torch.float32).to(device).numpy()
        #     data["pupil_y"] = data["pupil_y"][:data['pupil_y'].shape[0]//4].type(torch.float32).to(device).numpy()
        #     data["treadmill"] = data["treadmill"][:data['treadmill'].shape[0]//4].type(torch.float32).to(device).numpy()
        #     data["session"] = data["session"]
        # else:
        data["spikes"] = torch.round(data["spikes"]).type(torch.int8).to(device).numpy()
        data["stimulus"] = (
            data["stimulus"]
            .type(torch.float32)
            .unsqueeze(0)
            .to(device)
            .squeeze()
            .numpy()
            .reshape(-1, 36, 64)
        )
        data["dilation"] = data["dilation"].type(torch.float32).to(device).numpy()
        data["d_dilation"] = data["d_dilation"].type(torch.float32).to(device).numpy()
        data["pupil_x"] = data["pupil_x"].type(torch.float32).to(device).numpy()
        data["pupil_y"] = data["pupil_y"].type(torch.float32).to(device).numpy()
        data["treadmill"] = data["treadmill"].type(torch.float32).to(device).numpy()
        data["session"] = data["session"]

    # Change the data to experanto data
    # spikes = data['spikes'] # data['spikes'].shape (2023 (neuron), 150578 (activation))
    # stimulus = data['stimulus'] # data['stimulus'].shape (30117 (frame), 30 (H), 100 (W))
    # %%
    window = config.window.curr
    window_prev = config.window.prev
    dt = config.resolution.dt

    # -------- #

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

    """ 
    - see mconf.yaml "modalities" structure:

    modalities:
    behavior:
        n_layers: 4
        window: 0.05
        variables:
        speed:
            data: speed
            dt: 0.05
            predict: true
            objective: regression
        phi:
            data: phi
            dt: 0.05
            predict: true
            objective: regression
        th:
            data: th
            dt: 0.05
            predict: true
            objective: regression


    Modalities: any additional modalities other than spikes and frames
        Behavior: the name of the <modality type>
            Variables: the name of the <modality>
                Data: the data of the <modality> in shape (n_samples, n_features)
                dt: the time resolution of the <modality>, used to index n_samples
                Predict: whether to predict this modality or not.
                        If you set predict to false, then it will 
                        not be used as an input in the model,
                        but rather to be predicted as an output. 
                Objective: regression or classification

    """

    frames = {
        "feats": data["stimulus"],
        "callback": callback,
        "window": config.window.frame,
        "dt": config.resolution.dt,
    }

    def configure_token_types(config, modalities):
        max_window = max(config.window.curr, config.window.prev)
        dt_range = math.ceil(max_window / dt) + 1
        n_dt = [round_n(x, dt) for x in np.arange(0, max_window + dt, dt)]

        token_types = {
            "ID": {
                "tokens": list(
                    np.arange(
                        0,
                        (
                            data["spikes"].shape[0]
                            if (
                                isinstance(data["spikes"], np.ndarray)
                                or isinstance(data["spikes"], torch.Tensor)
                            )
                            else data["spikes"][1].shape[0]
                        ),
                    )
                )
            },
            "dt": {"tokens": n_dt, "resolution": dt},
            **(
                {
                    modality: {
                        "tokens": sorted(list(set(eval(modality)))),
                        "resolution": details.get("resolution"),
                    }
                    # if we have to classify the modality,
                    # then we need to tokenize it
                    for modality, details in modalities.items()
                    if config.modalities is not None
                    if details.get("predict", False)
                    and details.get("objective", "") == "classification"
                }
                if modalities is not None
                else {}
            ),
        }
        return token_types

    modalities = (
        create_modalities_dict(data, config.modalities)
        if get_attr(config, "modalities", None)
        else None
    )
    token_types = configure_token_types(config, modalities)
    tokenizer = Tokenizer(token_types)

    # %%
    if modalities is not None:
        for modality_type, modality in modalities.items():
            for variable_type, variable in modality.items():
                print(variable_type, variable)

    # %%
    print("Train Dataset>>>>>>>>>>>>>>>>>>")
    train_dataset = NFDataloader(
        spikes_dict,
        tokenizer,
        config,
        dataset=args.dataset,
        frames=frames,
        intervals=train_intervals,
        modalities=modalities,
        device=device,
    )
    print("Test Dataset>>>>>>>>>>>>>>>>>>")
    test_dataset = NFDataloader(
        spikes_dict,
        tokenizer,
        config,
        dataset=args.dataset,
        frames=frames,
        intervals=test_intervals,
        modalities=modalities,
        device=device,
    )
    print("Finetune Dataset>>>>>>>>>>>>>>>>>>")
    finetune_dataset = NFDataloader(
        spikes_dict,
        tokenizer,
        config,
        dataset=args.dataset,
        frames=frames,
        intervals=finetune_intervals,
        modalities=modalities,
        device=device,
    )

    print(f"train: {len(train_dataset)}, test: {len(test_dataset)}")
    iterable = iter(train_dataset)
    x, y = next(iterable)
    print(x["id"])
    print(x["dt"])
    # dict_to_device(x, device=device)
    recursive_print(x)

    train_datasets[data["session"]] = train_dataset
    test_datasets[data["session"]] = test_dataset
    finetune_datasets[data["session"]] = finetune_dataset
    tokenizers[data["session"]] = tokenizer

# train_dataloaders = LongCycler({session : DataLoader(train_datasets[session], batch_size=2, shuffle=True, num_workers=0) for session in train_datasets.keys()})
# test_dataloaders = LongCycler({session : DataLoader(test_datasets[session], batch_size=2, shuffle=True, num_workers=0) for session in test_datasets.keys()})
# finetune_dataloaders = LongCycler({session : DataLoader(finetune_datasets[session], batch_size=2, shuffle=True, num_workers=0) for session in finetune_datasets.keys()})

# Create model
model = Neuroformer(config, tokenizers).to(device)

# Create a DataLoader
loader = DataLoader(
    NFCombinedDataset(
        [test_datasets[session] for session in test_datasets.keys()],
        block_size=2,
        randomized=True,
    ),
    batch_size=2,
    shuffle=False,
    num_workers=0,
)
iterable = iter(loader)
x, y = next(iterable)
# dict_to_device(y, device=device)
recursive_print(y)
preds, features, loss = model(x, y)

train_dataset = NFCombinedDataset(
    [train_datasets[session] for session in train_datasets.keys()],
    block_size=config.training.batch_size,
    randomized=True,
    repeat_short=True,
)
test_dataset = NFCombinedDataset(
    [test_datasets[session] for session in test_datasets.keys()],
    block_size=config.training.batch_size,
    randomized=False,
)
finetune_dataset = NFCombinedDataset(
    [finetune_datasets[session] for session in finetune_datasets.keys()],
    block_size=config.training.batch_size,
    randomized=False,
)

# print("Loader>>>>>>>>>>>>>>>>>>")
# from tqdm import tqdm
# from torch.utils.data.distributed import DistributedSampler
# from utils import object_to_dict, save_yaml, all_device

# data = train_dataset
# loader = DataLoader(data, pin_memory=False,
#                                 batch_size=config.training.batch_size,
#                                 num_workers=16, sampler=None)

# pbar = tqdm(enumerate(loader), total=len(loader), disable=False)

# for it, (x, y) in pbar:

#     # place data on the correct device
#     x = all_device(x, torch.device('cuda', torch.cuda.current_device()))
#     y = all_device(y, torch.device('cuda', torch.cuda.current_device()))

# asdfasfga

if config.gru_only:
    model_name = "GRU"
elif config.mlp_only:
    model_name = "MLP"
elif config.gru2_only:
    model_name = "GRU_2.0"
else:
    model_name = "Neuroformer"

CKPT_PATH = f"./models/NF.15/Visnav_VR_Expt/{args.dataset}/{model_name}/{args.title}/sec_1_beh/{str(config.layers)}/{args.seed}"
CKPT_PATH = CKPT_PATH.replace("namespace", "").replace(" ", "_")

if os.path.exists(CKPT_PATH):
    counter = 1
    print(f"CKPT_PATH {CKPT_PATH} exists!")
    while os.path.exists(CKPT_PATH + f"_{counter}"):
        counter += 1

if args.resume is not None:
    model.load_state_dict(torch.load(args.resume), strict=False)

if args.sweep_id is not None:
    # this is for hyperparameter sweeps
    from neuroformer.hparam_sweep import train_sweep

    print(f"-- SWEEP_ID -- {args.sweep_id}")
    wandb.agent(args.sweep_id, function=train_sweep)
else:
    # Create a TrainerConfig and Trainer
    print(
        f"Final_epoch: {len(train_dataset)} * {(config.block_size.id)} * {(config.training.epochs)} = {len(train_dataset) * (config.block_size.id) * (config.training.epochs)}"
    )
    total_tokens = (
        len(train_dataset) * (config.block_size.id) * (config.training.epochs)
    )
    tconf = TrainerConfig(
        max_epochs=config.training.epochs,
        batch_size=config.training.batch_size,  # This should be equal to the block size
        learning_rate=1e-4,
        num_workers=4,
        lr_decay=True,
        patience=3,
        warmup_tokens=int(0.1 * total_tokens),
        decay_weights=True,
        weight_decay=1.0,
        shuffle=False,  # this shuffle should stay false, because shuffling is done in the dataset and the dataloader should not do shuffling
        final_tokens=total_tokens,
        clip_norm=1.0,
        grad_norm_clip=1.0,
        show_grads=False,
        ckpt_path=CKPT_PATH,
        no_pbar=False,
        dist=args.dist,
        save_every=100,
        eval_every=1,
        min_eval_epoch=2,
        use_wandb=True,
        wandb_project="neuroformer-experanto",
        wandb_group=f"1.5.1_visnav_{args.dataset}",
        wandb_name=args.title,
        wandb_entity="ecker-lab",
    )

    trainer = Trainer(model, train_dataset, test_dataset, tconf, config)
    trainer.train()

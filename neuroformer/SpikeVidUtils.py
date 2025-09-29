import collections
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import Q
from torch.utils import data

pd.options.mode.chained_assignment = None  # default='warn'

import copy
import math

from PIL import Image
from scipy import io as scipyio
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


def trial_df(data):
    """
    # data: Number of neurons (N,)
    # returns: Time x ID x Trial dataframe
    """
    # gather data from all the different neurons/trials
    df_dict = collections.defaultdict(dict)
    for ID in range(len(data)):
        df = data[ID]
        trial_dict = collections.defaultdict(list)
        for trial in range(df.shape[-1]):
            for row in range(df.shape[0]):
                if df[row, trial] <= 0 or df[row, trial] is None:
                    continue
                trial_dict[trial].append(df[row, trial])
        if len(trial_dict) == 0:
            continue
        df_dict[ID] = trial_dict

    # organize in one single dataframe
    df_list = []
    for ID in df_dict.keys():
        df_list.append(
            pd.concat(
                pd.DataFrame(
                    {
                        "Time": v,
                        "ID": ID,
                        "Trial": k,
                    }
                )
                for k, v in df_dict[ID].items()
            )
        )

    df = pd.concat(df_list).sort_values(by=["Trial", "Time"])
    return df


def trial_df_real(data):
    """
    # data: Number of neurons (N,)
    # returns: Time x ID x Trial dataframe
    """
    # gather data from all the different neurons/trials
    df_dict = collections.defaultdict(dict)
    for ID in range(len(data)):
        df = data[ID]
        if type(df) == np.void:
            df = df[0][:, 0]
        trial_dict = collections.defaultdict(list)
        for trial in range(df.shape[-1]):
            for row in range(df.shape[0]):
                if df[row, trial] is None:
                    continue
                trial_dict[trial] = df[row, trial].flatten().tolist()
        if len(trial_dict) == 0:
            continue
        df_dict[ID] = trial_dict

    # organize in one single dataframe
    df_list = []
    for ID in df_dict.keys():
        df_list.append(
            pd.concat(
                pd.DataFrame(
                    {
                        "Time": v,
                        "ID": ID,
                        "Trial": k,
                    }
                )
                for k, v in df_dict[ID].items()
            )
        )

    df = pd.concat(df_list).sort_values(by=["Trial", "Time"])
    return df


def trial_df_combo3(data, n_stim=0):
    """
    # data: Number of neurons (N,)
    # returns: Time x ID x Trial dataframe
    """
    # gather data from all the different neurons/trials
    df_dict = collections.defaultdict(dict)
    for ID in range(len(data)):
        df = data[ID]
        if type(df) == np.void:
            df = df[0][:, n_stim]
        trial_dict = collections.defaultdict(list)
        for trial_no in range(df.shape[-1]):
            trial = df[trial_no].flatten()
            if len(trial) == 0:
                continue
            trial_dict[trial_no + (20) * n_stim] = trial.tolist()
        if len(trial_dict) == 0:
            continue
        df_dict[ID] = trial_dict

    # organize in one single dataframe
    df_list = []
    for ID in df_dict.keys():
        df_list.append(
            pd.concat(
                pd.DataFrame(
                    {
                        "Time": v,
                        "ID": ID,
                        "Trial": k,
                    }
                )
                for k, v in df_dict[ID].items()
            )
        )

    df = pd.concat(df_list).sort_values(by=["Trial", "Time"])
    return df


def trial_combo_rampV2(spike_data, video_stack):
    df_dict = collections.defaultdict(dict)
    dt = spike_data["ComboRamp_22"][0][2].item()[5]
    stim_time = spike_data["ComboRamp_22"][0][2].item()[1]
    for ID in range(len(spike_data["ComboRamp_22"][0][0])):
        df = spike_data["ComboRamp_22"][0][0][ID][0]
        # if type(df) == np.coid:
        trial_dict = collections.defaultdict(dict)
        for trial_no in range(df.shape[0]):
            trial = df[trial_no].flatten()
            # if len(trial) == 0:
            #     continue
            # trial_dict[trial_no] = [x for x in trial if isinstance(x, float)]
            for n_stim, st in enumerate(trial):
                st = (
                    list(st)
                    if isinstance(st, np.ndarray) or isinstance(st, list)
                    else [st]
                )
                trial_dict[trial_no][n_stim] = st
                # print(trial_dict[trial_no])
            # trial_dict[trial_no] =
        # if len(trial_dict) == 0:
        #     continue
        df_dict[ID] = trial_dict

    df_list = []
    for ID in df_dict.keys():
        for trial in df_dict[ID].keys():
            df_list.append(
                pd.concat(
                    pd.DataFrame({"Time": v, "ID": ID, "Trial": trial, "Stimulus": k})
                    for k, v in df_dict[ID][trial].items()
                )
            )

    df = pd.concat(df_list)
    df["Time"] *= dt
    df = df.sort_values(
        by=[
            "Trial",
            "Stimulus",
            "Time",
        ]
    )


def set_trials(df, trials):
    """
    Add a trial column to df according
    to trial intervals specified in trials.
    """
    df["Trial"] = 1
    # for i, trial in enumerate(trials):
    #     if i < len(trials) - 1:
    #         df.loc[(df['Time'] >= trial[0]) & (df['Time'] < trial[1]), 'Trial'] = i
    #     else:
    #         df.loc[(df['Time'] >= trial[0]), 'Trial'] = i
    # df['Trial'][df['Trial'] < 2] = 1

    for i in range(len(trials)):
        trial = trials[i]
        interval_1 = trial[1]
        n_trial = trial[0]

        if i < len(trials) - 1:
            trial_next = trials[i + 1] if i < len(trials) - 1 else df["Time"].max()
            interval_2 = trial_next[1]
        else:
            interval_2 = df["Time"].max() + 1
        df["Trial"][(df["Time"] >= interval_1) & (df["Time"] < interval_2)] = i
        df["Time"][(df["Time"] >= interval_1) & (df["Time"] < interval_2)] -= interval_1
    assert np.all(df["Trial"] != -1), "Some trials were not assigned"
    return df


def get_df_visnav(data, trials=None, dt_vars=None):
    """
    # data: Number of neurons (N,)
    # returns: Time x ID x Trial dataframe
    """
    # gather data from all the different neurons/trials
    df_dict = collections.defaultdict(dict)
    for ID in range(len(data)):
        df = data[ID]
        trial_dict = collections.defaultdict(list)
        for i, trial in enumerate(df):
            if len(trial.shape) == 0:
                continue
            for row in trial:
                trial_dict[i].append(row)
        if len(trial_dict) == 0:
            continue
        df_dict[ID] = trial_dict

    # organize in one single dataframe
    df_list = []
    for ID in df_dict.keys():
        df_list.append(
            pd.concat(
                pd.DataFrame(
                    {
                        "Time": v,
                        "ID": ID,
                        "Trial": k,
                    }
                )
                for k, v in df_dict[ID].items()
            )
        )

    df = pd.concat(df_list).sort_values(by=["Trial", "Time"])
    df = df.dropna().reset_index(drop=True)
    df["raw_time"] = df["Time"]
    if dt_vars is not None:
        df["raw_interval"] = make_intervals(df, dt_vars, "raw_time")

    if trials is not None:

        for i in range(len(trials)):
            trial = trials[i]
            interval_1 = trial[1]
            n_trial = trial[0]

            if i < len(trials) - 1:
                trial_next = trials[i + 1] if i < len(trials) - 1 else df["Time"].max()
                interval_2 = trial_next[1]
            else:
                interval_2 = df["Time"].max() + 1
            df["Trial"][(df["Time"] >= interval_1) & (df["Time"] < interval_2)] = i
            df["Time"][
                (df["Time"] >= interval_1) & (df["Time"] < interval_2)
            ] -= interval_1

    df["Trial"][df["Trial"] == 0] = 1

    return df

    def trim_trial(df, video_stack):
        fps = 20
        for stimulus in video_stack.keys():
            max_time = video_stack[stimulus].shape[1] / fps
            df.drop(
                df[(df["Stimulus"] == stimulus) & (df["Time"] > max_time)].index,
                inplace=True,
            )
            df.drop(
                df[(df["Stimulus"] == stimulus) & (df["Time"] < 0)].index, inplace=True
            )
        return df

    df = df.reset_index(drop=True)
    df = trim_trial(df, video_stack).reset_index(drop=True)
    return df


def split_intervals(df, interval=1):
    new_df = df.sort_values(["Trial", "Time"]).reset_index().T.iloc[1:]
    prev_t = 0
    n = 0
    for column in range(new_df.shape[-1]):
        t = new_df.iloc[0, column]
        if t < prev_t:
            prev_t = 0
        dt = t - prev_t
        if dt >= interval:
            idx = column
            new_column = [new_df.iloc[0, idx - 1], ".", new_df.iloc[2, idx]]
            col_name = "n_ %i" % (idx)
            new_df.insert(idx, idx, new_column, allow_duplicates=True)
            n += 1
            prev_t = t

    return new_df.T.reset_index().iloc[:, 1:]


def set_intervals(
    df, window, window_prev, pred_window=None, max_window=None, min_window=None
):
    """
    Set intervals for predictions
    """
    min_interval = window + window_prev if min_window is None else min_window
    pred_interval = window if pred_window is None else pred_window
    df["Interval"] = make_intervals(df, pred_interval)
    if max_window is not None:
        df = df[df["Interval"] < max_window]
    if min_window is not None:
        df = df[df["Interval"] > min_window]
    df = df.sort_values(by=["Interval", "Trial"])
    return df.reset_index(drop=True)


# def make_intervals(data, window):
#     intervals = []
#     for trial in sorted(data['Trial'].unique()):
#         df = data[data['Trial'] == trial]
#         rn = 0
#         while True:

#             rn += window
#             interval = df[(df['Time'] < rn) & (df['Time'] >= rn - window)]
#             intervals += [rn] * len(interval)
#             if rn > max(df['Time']):
#                 break
#     intervals = np.array(intervals).round(2)
#     return intervals


def make_intervals(data, window, col="Time"):
    def round_up_to_nearest_half_int(num, window):
        return math.ceil(num * (1 / window)) / (1 / window)

    # print(f"3: {data['Interval'].max()}")
    intervals = data[col].apply(lambda x: round_up_to_nearest_half_int(x, window))
    # print(f"4: {data['Interval'].max()}")

    return intervals


def create_full_trial(df, trials=None):
    if trials is not None:
        df_full = df[df["Trial"].isin(trials)].reset_index(drop=True)
    else:
        df_full = df.copy()
    df_full.loc[df_full["Trial"] > 20, "Interval"] += 32
    df_full.loc[df_full["Trial"] > 20, "Time"] += 32
    df_full.loc[df_full["Trial"] > 40, "Interval"] += 32
    df_full.loc[df_full["Trial"] > 40, "Time"] += 32
    df_full["Trial"] = 0
    return df_full


def make_intervals_v2(data, window):
    intervals = []
    groups = data.groupby(["Stimulus", "Trial"]).size().index.values
    for trial_stim in groups:
        stim = trial_stim[0]
        trial = trial_stim[1]
        df = data[(data["Trial"] == trial) & (data["Stimulus"] == stim)]
        rn = 0
        while True:
            rn += window
            interval = df[(df["Time"] < rn) & (df["Time"] >= rn - window)]
            intervals += [rn] * len(interval)
            if rn > max(df["Time"]):
                break
    intervals = np.array(intervals).round(2)
    return intervals


def group_intervals(df, dt):
    """group data into intervals"""
    bins = int(max(df["Time"]) / dt)
    intervals = pd.cut(df["Time"], bins=int(max(df["Time"]) / dt))
    labels = [dt + dt * n for n in range(0, int(max(df["Time"]) / dt))]
    df["intervals"] = pd.cut(
        df["Time"], bins=int(max(df["Time"]) / dt), labels=labels
    ).astype("float")
    df["intervals"].round(decimals=1)
    return df


def split_idx(df, block_size):
    """assign indexer to intervals for DataLoader class"""
    new_row = []
    current_i = -1
    seq_l = 0
    prev_dt = 0
    for row in df.iterrows():
        dt = row[-1]["intervals"]
        if dt == prev_dt:
            if seq_l > block_size // 2:
                current_i = current_i + 1
                seq_l = 0
        else:
            current_i = current_i + 1
            seq_l = 0
        new_row.append(current_i)
        prev_dt = dt
    df["idx"] = new_row
    return df


# Tf combo = 1/60
# Tf = 1/20   # preriod
def get_frame_idx(t, Tf):
    """
    Get the raw index of frame at certain neuron firing time

    # Tf:
    video 1/f (s)
    """
    idx = math.ceil(t / Tf)
    return idx if idx > 0 else 0


def get_interval_idx(idx, Tf):
    interval = idx * Tf
    return interval


# Tf combo = 1/60
def dt_frames_idx(t, Tf, dt_frames=0.25):
    """
    Get the sequence of frames index at certain time

    # dt_frames:
    # dt / Tf
    """
    return int(t // Tf) // dt_frames


def image_dataset(frame_stack, size=(64, 112)):
    """Convert frames into images tensor compatible with Resnet"""
    H, W = size[0], size[1]
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((H, W)),
            transforms.CenterCrop((H, W)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # frame_stack = io.imread(path)
    im_list = []
    for idx in range(len(frame_stack)):
        # image = Image.fromarray(frame_stack[idx])   # .convert('RGB')
        image = frame_stack[idx]
        image = preprocess(image).unsqueeze(0)
        image = (image / image.max()) - 0.5
        im_list.append(image)

    im_stack = torch.cat(im_list)
    print("im_sack size: {}".format(im_stack.size()))
    return im_stack


def r3d_18_dataset(frame_stack):
    """Convert frames into images tensor compatible with Resnet"""
    preprocess = transforms.Compose(
        [
            transforms.Resize((128, 171), interpolation=Image.BILINEAR),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
            ),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # frame_stack = io.imread(path)
    im_list = []
    for idx in range(len(frame_stack)):
        image = frame_stack[idx]
        image = Image.fromarray(image).convert("RGB")
        image = preprocess(image).unsqueeze(0)
        image /= 255
        im_list.append(image)

    im_stack = torch.cat(im_list)
    print("im_sack size: {}".format(im_stack.size()))

    im_stack = im_stack.transpose(0, 1)
    return im_stack


def video_dataset(frame_stack):
    """
    Scale, Normalize, and Convert to format (C, T, H, W) for ResNet 3D
    """
    frame_stack = frame_stack / frame_stack.max()
    mean = frame_stack.mean()
    std = frame_stack.std()
    frame_stack = (frame_stack - mean) / std
    frame_stack = frame_stack / frame_stack.max()
    video = np.repeat(frame_stack[None, ...], 3, axis=0)
    return video


def neuron_dict(df):
    """
    Convert pandas df[[ID, Time, Trial, Interval]]
    into nested dict
    """
    d = {
        k: f.groupby("Interval")
        .apply(lambda x: {"Time": np.array(x["Time"]), "ID": np.array(x["ID"])})
        .to_dict()
        for k, f in df.groupby("Trial")
    }

    return d


def round_n(x, base):
    return round(base * (round(float(x) / base)), 2)
    # return round(base * float(x)/base)


def get_interval(
    data,
    stoi,
    stoi_dt,
    dt,
    interval,
    trial,
    block_size,
    data_dict=None,
    n_stim=None,
    pad=True,
):
    """
    Returns interval[0] >= data < interval[1]
    chunk = ID
    dt_chunk = dt
    pad_n
    """
    window = max(list(stoi_dt.keys()))
    if data_dict is None:
        data = data[data["Trial"] == trial]
        data = data[
            (data["Interval"] > interval[0]) & (data["Interval"] <= interval[1])
        ][-(block_size - 2) :]
        if n_stim is not None:
            data = data[data["Stimulus"] == n_stim]
    else:
        data = data_dict[trial]
        if interval[1] in data:
            data = data[interval[1]]
        else:
            data = {"Time": np.array([]), "ID": np.array([])}

    chunk = data["ID"][-(block_size - 2) :]
    dix = [stoi[s] for s in chunk]
    dix = ([stoi["SOS"]] + dix + [stoi["EOS"]])[-block_size:]
    pad_n = (
        block_size - (len(dix) + 1 - 2) if pad else 0
    )  # len chunk is 1 unit bigger than x, y
    dix = dix + [stoi["PAD"]] * pad_n

    dt_chunk = data["Time"] - (interval[0])
    dt_chunk = [dt_ if dt_ <= window else window for dt_ in dt_chunk]
    dt_chunk = [stoi_dt[round_n(dt_, dt)] for dt_ in dt_chunk]

    if "EOS" in stoi_dt.keys():
        dt_chunk = (dt_chunk + stoi_dt["EOS"])[-block_size:]
        dt_chunk = (
            [0] + dt_chunk + stoi_dt["EOS"] + [stoi_dt["PAD"]] * (pad_n)
        )  # 0 = SOS, max = EOS
    else:
        if len(dt_chunk) > 0:
            dt_max = max(dt_chunk)
        else:
            dt_max = 0
        dt_chunk = ([0] + dt_chunk + [dt_max] * (pad_n + 1))[
            -block_size:
        ]  # 0 = SOS, max = EOS

    return dix, dt_chunk, pad_n


def get_interval_trials(df, window, window_prev, frame_window, dt):
    # start_interval = max(window, window_prev)
    start_interval = max(window + window_prev, frame_window)
    curr_intervals = np.arange(
        start_interval + dt, max(df["Interval"]) + window, window, dtype=np.float32
    )
    real_intervals = np.arange(
        start_interval + dt, max(df["real_interval"]) + dt, dt, dtype=np.float32
    )
    trials = sorted(df["Trial"].unique())
    intervals = np.array(np.meshgrid(curr_intervals, real_intervals, trials)).T.reshape(
        -1, 3
    )
    return intervals


def pad_x(x, length, pad_token, device=None):
    """
    pad x with pad_token to length
    """
    if torch.is_tensor(x):
        x = x.tolist()

    pad_n = int(length - len(x))
    if pad_n < 0:
        x = x[-(length + 1) :]
    if pad_n > 0:
        x = x + [pad_token] * pad_n
    x = torch.tensor(x, dtype=torch.long)
    return x.to(device) if device is not None else x


def pad_x_nd(x, length, pad_token, axis=0, device=None):
    x = F.pad(x, (0) * (x.dim() * 2), "constant", pad_token)


def pad_tensor(x, length, pad_value=0):
    """
    pad tensor along last dim
    """
    n_pad = length - x.shape[-1]
    if n_pad < 0:
        return x[..., -length:]
    else:
        pad = list(x.shape)
        pad[-1] = n_pad
        pad_tensor = torch.zeros(pad, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad_tensor], dim=-1)


def get_var(data, interval, variable=None, trial=None):
    """
    Returns interval[0] >= data < interval[1]
    """
    if trial is not None:
        data = data[data["Trial"] == trial]
    if variable is not None:
        data = data[variable]
    data = data[(data["Interval"] > interval[0]) & (data["Interval"] <= interval[1])]
    return data


def round_n(x, base):
    return round(base * (round(float(x) / base)), 6)


def resample_groups(dataframe, extra_samples_factor):
    # Calculate group sizes
    group_size = (
        dataframe.groupby(["Interval", "Trial"]).size().reset_index(name="count")
    )
    # Calculate inverse weights (so that smaller groups have higher chances)
    weights = 1 / group_size["count"]
    # Normalizing the weights so they sum to 1
    weights = weights / weights.sum()
    # Define how many extra samples you want
    num_extra_samples = (len(group_size) * extra_samples_factor) - len(group_size)
    # Perform stratified resampling
    sampled_groups = pd.concat(
        [
            group_size,
            group_size.sample(n=num_extra_samples, replace=True, weights=weights),
        ],
        ignore_index=True,
    )
    # Expand the count column into rows
    expanded_sampled_groups = sampled_groups.loc[
        sampled_groups.index.repeat(sampled_groups["count"])
    ].reset_index(drop=True)
    # Drop the 'count' column as it is no longer needed
    expanded_sampled_groups = expanded_sampled_groups.drop(columns=["count"])
    # Return the expanded_sampled_groups DataFrame
    return expanded_sampled_groups


def tokenizer(neurons, max_window, dt, no_eos_dt=False):
    feat_encodings = neurons + ["SOS"] + ["EOS"] + ["PAD"]
    stoi = {ch: i for i, ch in enumerate(feat_encodings)}
    itos = {i: ch for i, ch in enumerate(feat_encodings)}
    dt_range = (
        math.ceil(max_window / dt) + 1
    )  # add first / last interval for SOS / EOS'
    n_dt = [round(dt * n, 2) for n in range(dt_range)]
    """
    add exclusive EOS and PAD dts,
    if not we use max_dt for EOS and PAD IDs
    inside dataloder
    """
    if no_eos_dt is False:
        n_dt += ["EOS"] + ["PAD"]
    else:
        n_dt += ["PAD"]
    stoi_dt = {ch: i for i, ch in enumerate(n_dt)}
    itos_dt = {i: ch for i, ch in enumerate(n_dt)}
    return stoi, itos, stoi_dt, itos_dt


# dataloader class
class SpikeTimeVidData2(Dataset):
    """
    # data:
    0 col: Time
    1 col: Neuron ID

    # block_size:
    Transformer Window

    # dt
    Time intervals from data col 0

    # stoi, itos: dictionaries mapping neuron ID to transformer vocab
    and vice-versa.
    """

    def __init__(
        self,
        data,
        frames,
        block_size,
        id_block_size,
        frame_block_size,
        id_prev_block_size,
        window,
        dt,
        frame_memory,
        stoi,
        itos,
        neurons,
        stoi_dt=None,
        itos_dt=None,
        frame_feats=None,
        pred=False,
        data_dict=None,
        window_prev=None,
        dataset=None,
        intervals=None,
        behavior=None,
        predict_behavior=False,
        **kwargs,
    ):

        self.stoi = stoi
        self.itos = itos
        self.stoi_dt = stoi_dt
        self.itos_dt = itos_dt

        self.dt = dt
        self.dt_max = max(list(stoi_dt.values()))

        self.frame_block_size = frame_block_size
        self.block_size = block_size
        self.id_prev_block_size = id_prev_block_size
        self.id_block_size = id_block_size

        assert self.id_block_size > 0
        # assert self.frame_block_size + self.id_block_size + self.prev_id_block_size == self.block_size

        self.frame_memory = frame_memory
        self.data = data.reset_index(drop=True)
        self.data_dict = None
        self.intervals = intervals
        self.frame_feats = frame_feats
        self.behavior_feats = behavior

        self.population_size = len([*stoi.keys()])
        self.id_population_size = len([*stoi.keys()])
        self.dt_population_size = len([*stoi_dt.keys()])
        # print the pop, id and dt sizes
        print("Population Size: ", self.population_size)
        print("ID Population Size: ", self.id_population_size)
        print("DT Population Size: ", self.dt_population_size)
        self.dataset = dataset

        # keep track of which interval we are and how many neurons remain
        # (i.e did not fit in block)
        # self.t = self.data['Interval'].unique()
        self.window = window  # interval window (prediction)
        self.window_prev = window if window_prev is None else window_prev
        # assert self.window_prev % self.window == 0, "window_prev must be a multiple of window"
        self.frame_window = 1.0
        self.min_interval = window + window_prev
        print(f"Min Interval: {self.min_interval}")
        self.min_trial = data["Trial"].min()
        self.resample_data = False

        # calculate intervals on init
        self.set_intervals(data)

        self.idx = 0
        self.window = window  # interval window (prediction)
        self.window_prev = window if window_prev is None else window_prev
        self.interval = 0  # our current interval
        self.pred = pred

        self.dt_frames = None
        self.behavior_vars = None
        self.predict_behavior = predict_behavior

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.t)

    def set_intervals(self, data):
        self.data = data.reset_index(drop=True)
        self.min_trial = data["Trial"].min()

        if self.intervals is not None:
            print(f"Using smooth intervals")
            self.t = self.intervals
        elif self.resample_data:
            self.t = resample_groups(self.data, 10)
        else:
            if self.dataset != "combo_v2":
                self.t = self.data.drop_duplicates(
                    subset=["Interval", "Trial"]
                )  # .sample(frac=1).reset_index(drop=True) # interval-trial unique pairs
            elif self.dataset == "combo_v2":
                self.t = self.data.drop_duplicates(
                    subset=["Interval", "Trial", "Stimulus"]
                )

            if self.dataset == "LRN":
                # self.t = self.t[self.t['Interval'] >= self.min_interval].reset_index(drop=True)
                self.t = self.t[
                    ~(
                        (self.t["Trial"] == self.min_trial)
                        & (self.t["Interval"] <= self.min_interval)
                    )
                ].reset_index(drop=True)
            # elif self.dataset == 'LRN':
            # for LRN dataset the current trial continuous from the previous one
            # self.t[self.t['Trial'] == self.min_trial] = self.t[self.t['Trial'] == self.min_trial][self.t[self.t['Trial'] == self.min_trial]['Interval'] >= self.min_interval]
            # self.t = self.t.dropna().reset_index(drop=True)

    def copy(self, data, t=None, **kwargs):
        """return new class with everything the same except data,
        and the recalculation of self.t and self.size"""
        new = copy.deepcopy(self)
        new.data = data
        for k, v in kwargs.items():
            setattr(new, k, v)

        if t is not None:
            new.intervals = t
        new.set_intervals(data)
        return new

    # return round(base * float(x)/base)

    def calc_intervals(self, interval):
        prev_int = round_n(interval - self.window, self.dt)
        prev_id_interval = round_n(prev_int - self.window_prev, self.dt), prev_int
        current_int = prev_int
        current_id_interval = prev_int, round_n(current_int + self.window, self.dt)
        assert prev_id_interval[1] == current_id_interval[0]
        return prev_id_interval, current_id_interval

    def get_data_prev_trial(self, trial, prev_id_interval):
        """
        the start of a trial cotninues from the end of the previous trial
        """
        data = self.data
        max_time_prev = data[data["Trial"] == trial - 1]["Time"].max()

        prev_trial = trial - 1
        prev_trial_interval = (max_time_prev + prev_id_interval[0], max_time_prev)
        prev_trial_data = data[
            (data["Trial"] == prev_trial) & (data["Time"] > prev_trial_interval[0])
        ]
        # prev_trial_data['Time'] = prev_trial_data['Time'] - prev_trial_interval[0]

        current_trial_data = data[
            (data["Trial"] == trial)
            & (data["Time"] > prev_id_interval[0])
            & (data["Time"] <= prev_id_interval[1])
        ]
        t_diff = prev_trial_interval[1] - prev_id_interval[0]

        try:
            prev_trial_data["Time"] = (
                prev_trial_data["Time"] - prev_trial_interval[0].min()
            )
        except:
            print(
                f"trial: {trial}, inteval: {prev_id_interval}, prev_trial_interval: {prev_trial_interval}"
            )
        current_trial_data["Time"] = current_trial_data["Time"] - prev_id_interval[0]

        # connect the two trials
        prev_id_data = pd.concat([prev_trial_data, current_trial_data], axis=0)
        prev_id_data = prev_id_data.sort_values(by=["Time"])

        # prev_id_interval = (data[data['Trial'] == trial - 1]['Time'].max(), prev_id_interval[1])
        return prev_id_data, prev_id_interval

    def get_behavior(self, data, interval, variables=None, trial=None):
        """
        Returns interval[0] >= data < interval[1]
        """
        data = get_var(data, interval, trial=trial)
        variables = data.columns if variables is None else variables
        behavior = torch.tensor(
            np.array(data[variables]), dtype=torch.float32
        ).transpose(0, 1)
        if len(data) > 0:
            behavior_dt = torch.tensor(
                np.array(data["Time"]) - np.array(data["Interval"]), dtype=torch.float32
            )
        else:
            behavior_dt = torch.tensor([0], dtype=torch.float32)
        # pad
        behavior = pad_tensor(behavior, self.samples_per_behavior, self.stoi["PAD"])
        behavior_dt = pad_tensor(
            behavior_dt, self.samples_per_behavior, self.stoi_dt["PAD"]
        )
        return behavior.unsqueeze(-1), behavior_dt

    def get_interval(
        self,
        interval,
        trial,
        block_size,
        data=None,
        data_dict=None,
        n_stim=None,
        pad=True,
    ):
        """
        Returns interval[0] >= data < interval[1]
        chunk = ID
        dt_chunk = dt
        pad_n
        """
        data = self.data if data is None else data
        if self.data_dict is None:
            if interval[0] < 0 and self.dataset in ["LRN", "vinav"]:
                data, id_interval = self.get_data_prev_trial(trial, interval)
            else:
                data = data[data["Trial"] == trial]
                data = data[
                    (data["Time"] >= interval[0]) & (data["Time"] < interval[1])
                ][-(block_size - 2) :]
                if n_stim is not None:
                    data = data[data["Stimulus"] == n_stim]
        # else:
        #     data = self.data_dict[trial]
        #     if interval[1] in data:
        #         data = data[interval[1]]
        #     else:
        #         data = {'Time': np.array([]), 'ID': np.array([])}

        chunk = data["ID"][-(block_size - 2) :]
        dix = [self.stoi[s] for s in chunk]
        # trial_token = self.stoi['Trial ' + str(int(trial))]
        dix = ([self.stoi["SOS"]] + dix + [self.stoi["EOS"]])[-block_size:]
        # dix = ([trial_token] + dix + [self.stoi['EOS']])[-block_size:]
        pad_n = (
            block_size - (len(dix) + 1 - 2) if pad else 0
        )  # len chunk is 1 unit bigger than x, y
        dix = dix + [self.stoi["PAD"]] * pad_n

        # print(data['Time'], "int", interval[0])
        dt_chunk = (data["Time"] - (interval[0])) if interval[0] > 0 else data["Time"]
        dt_chunk = [self.stoi_dt[round_n(dt, self.dt)] for dt in dt_chunk]

        if "EOS" in self.stoi_dt.keys():
            eos_token = self.stoi_dt["EOS"]
        else:
            eos_token = max(self.stoi_dt.values())

        dt_chunk = ([0] + dt_chunk + [eos_token])[-block_size:]
        dt_chunk = dt_chunk + [self.stoi_dt["PAD"]] * pad_n
        # else:
        #     dt_chunk = [0] + max(self.stoi_dt.values()) * pad_n

        return dix, dt_chunk, pad_n

    def __getitem__(self, idx):
        """
        Using an odd Block_Size, in order to be able to
        appropriately mask joint image and id encodings.

        Example for block_size = n:

        x = [frame_token_1... frame_token_n ..., id_1, id_n,]
        y = [frame_token_2... frame_token_n + 1 ..., id_2, id_n + 1,]

        """

        # grab a chunk of (block_size + 1) characters from the data
        if isinstance(self.t, pd.DataFrame):
            t = self.t.iloc[idx]
        else:
            # (curr_interva, real_interval, trial)
            interval_ = self.t[idx]
            t = dict()
            # t['Interval'] = interval_[0].astype(float)
            # if 'real_interval' in self.data.columns:
            #     t['real_interval'] = interval_[0].astype(float)
            t["Interval"] = interval_
            t["Trial"] = interval_[2].astype(int) if self.dataset != "visnav" else 0
            t["Stimulus"] = (
                torch.zeros(1, dtype=torch.long) if self.dataset == "LRN" else None
            )

        x = collections.defaultdict(list)
        y = collections.defaultdict(list)

        n_stim = None if "Stimulus" not in t else t["Stimulus"]

        # get intervals
        prev_id_interval, current_id_interval = self.calc_intervals(t["Interval"])

        ## PREV ##
        id_prev, dt_prev, pad_prev = self.get_interval(
            prev_id_interval, t["Trial"], self.id_prev_block_size, n_stim
        )
        x["id_prev"] = torch.tensor(id_prev[:-1], dtype=torch.long)
        x["dt_prev"] = torch.tensor(dt_prev[:-1], dtype=torch.float)  # + 0.5
        x["pad_prev"] = torch.tensor(pad_prev, dtype=torch.long)

        ## CURRENT ##
        idn, dt, pad = self.get_interval(
            current_id_interval, t["Trial"], self.id_block_size, n_stim
        )
        x["id"] = torch.tensor(idn[:-1], dtype=torch.long)
        x["dt"] = torch.tensor(dt[:-1], dtype=torch.float)  # + 1
        x["pad"] = torch.tensor(pad, dtype=torch.long)  # to attend eos

        y["id"] = torch.tensor(idn[1:], dtype=torch.long)
        y["dt"] = torch.tensor(dt[1:], dtype=torch.long)
        x["interval"] = torch.tensor(t["Interval"], dtype=torch.float32)
        x["trial"] = torch.tensor(t["Trial"], dtype=torch.long)

        ## BEHAVIOR ##
        if self.behavior_feats is not None:
            # check if self.window_behavior exists
            if hasattr(self, "window_behavior") and self.window_behavior is not None:
                behavior_interval = (
                    current_id_interval[1] - self.window_behavior,
                    current_id_interval[1],
                )
            else:
                behavior_interval = (prev_id_interval[0], current_id_interval[1])
            behavior, behavior_dt = self.get_behavior(
                self.behavior_feats,
                behavior_interval,
                variables=self.behavior_vars,
                trial=t["Trial"],
            )
            if self.predict_behavior:
                # pick only current_state behavior
                # TODO: implement for more than just 0.05 curr window
                y["behavior"], y["behavior_dt"] = behavior[:, -1], behavior_dt[-1]
                y["behavior"] = torch.tensor(
                    [
                        self.stoi_speed[round_n(float(i), self.dt_speed)]
                        for i in y["behavior"]
                    ],
                    dtype=torch.long,
                )
                # y['behavior'], y['behavior_dt'] = behavior, behavior_dt
            else:
                x["behavior"], x["behavior_dt"] = behavior, behavior_dt

        # for backbone:
        ## TODO: IMPLEMENT THIS DATASET BY DATASET
        # if self.frame_feats is not None:
        #     if isinstance(self.frame_feats, dict):
        #         n_stim = int(t['Stimulus'])
        #     elif len(self.frame_feats) == 8:
        #         if self.t['Trial'].max() <= 8:
        #             n_stim = int(t['Trial'])
        #         else:
        #             n_stim = int(t['Trial'] // 200) - 1
        #     elif self.frame_feats.shape[0] == 1:
        #         n_stim = 0
        if self.dataset == "Combo3_V1AL":
            if t["Trial"] <= 20:
                n_stim = 0
            elif t["Trial"] <= 40:
                n_stim = 1
            elif t["Trial"] <= 60:
                n_stim = 2
        elif self.dataset == "NaturalStim":
            n_stim = t["Trial"]

        #     elif self.dataset == 'combo_v2':
        #         n_stim = n_stim
        if self.frame_feats is not None:
            # t['Interval'] += self.window
            dt_frames = self.dt_frames if self.dt_frames is not None else 1 / 20
            frame_interval = (
                t["Interval"] if "raw_interval" not in t else t["raw_interval"]
            )
            # add 0.5 so we start at end of interval and then go backwards
            frame_interval += self.window
            frame_idx = get_frame_idx(
                frame_interval, dt_frames
            )  # get last 1 second of frames
            frame_window = self.frame_window
            n_frames = math.ceil(int(1 / dt_frames) * frame_window)
            frame_idx = frame_idx if frame_idx >= n_frames else n_frames
            f_b = n_frames
            # f_f = n_frames - f_b
            frame_feats_stim = (
                self.frame_feats[n_stim] if n_stim is not None else self.frame_feats
            )
            frame_idx = min(frame_idx, frame_feats_stim.shape[0])
            # frame_idx = frame_idx if frame_idx < frame_feats_stim.shape[1] else frame_feats_stim.shape[1]
            f_diff = frame_idx - n_frames
            if f_diff < 0:
                f_b = frame_idx
            if self.dataset == "LIF2":
                frame_idx = frame_idx - 1
                x["frames"] = frame_feats_stim[:, frame_idx].type(torch.float32)
                x["frames"] = x["frames"].repeat(1, 1).transpose(0, 1)
            elif self.dataset == "visnav":
                offset = 0
                f_idx_0 = max(0, frame_idx - f_b - offset)
                f_idx_1 = f_idx_0 + f_b
                x["frames"] = (
                    frame_feats_stim[f_idx_0:f_idx_1].type(torch.float32).unsqueeze(0)
                )
            else:
                f_idx_0 = max(0, frame_idx - f_b)
                # f_idx_1 = min(f_idx_0 + f_b, frame_feats_stim.shape[1])
                f_idx_1 = f_idx_0 + f_b
                assert (
                    f_idx_1 - f_idx_0 == n_frames
                ), f"f_idx_1 {f_idx_1} - f_idx_0 {f_idx_0} != n_frames {n_frames}"
                x["frames"] = (
                    frame_feats_stim[f_idx_0:f_idx_1].type(torch.float32).unsqueeze(0)
                )
            # else:
            #     x['frames'] = frame_feats_stim[:, :, frame_idx - f_b:frame_idx].type(torch.float32)

            if n_stim is not None:
                x["stimulus"] = torch.tensor(n_stim, dtype=torch.long)
            x["f_idx"] = torch.tensor([f_idx_0, f_idx_1])

        x["cid"] = torch.tensor(current_id_interval)
        x["pid"] = torch.tensor(prev_id_interval)

        return x, y


# # video encodings
# frame_idx = math.ceil((t['Interval'] / self.window) - 1)    # get last 1 second of frames
# frame_idx = len(self.frame_feats) - 1 if frame_idx >= len(self.frame_feats) else frame_idx
# if self.frames is not None:
#     frames = self.frames[frame_idx]
#     fdix = [self.stoi[s] for s in frames]
#     y_frames = fdix[1:] + [self.stoi['SOS']]
#     y['frames'] = torch.tensor(y_frames, dtype=torch.long)

# if self.frame_feats is not None:
#     x['frames'] = torch.tensor(self.frame_feats[frame_idx], dtype=torch.float32)
# else:
#     x['frames'] = torch.tensor(fdix, dtype=torch.long)

# if self.frames is not None:
#     x['frame_codes'] = torch.tensor(fdix, dtype=torch.long)

# def top_k_logits(logits, k):
#     v, ix = torch.topk(logits, k)
#     out = logits.clone()
#     out[out < v[:, [-1]]] = -float('inf')
#     return out


# @torch.no_grad()
# def sample(model, loader, temperature=1/0, sample=False, top_k=None):
#     block_size = model.get_block_size()
#     model.eval()
#     for x, y in loader:
#         for key, value in x.items():
#             x[key] = x[key].to(self.device)
#         y = y.to(self.device)


"""

loader = DataLoader(train_dataset, batch_size=5, shuffle=False, pin_memory=False)
iterable = iter(train_dataset)
x, y = next(iterable)

T = len(x['id'])
P = x['pad'] - 1
T_prev = len(x['id_prev'])
P_prev = x['pad_prev'] - 4

iv = float(x['interval'])

xid = x['id'][: T - P]
xid = [itos[int(i)] for i in xid]

xid_prev = x['id_prev'][: T_prev - P_prev]
xid_prev = [itos[int(i)] for i in xid_prev]

print(f"iv: {iv}, ix+window: {iv + window} pid: {x['pid']} cid: {x['cid']}")
print(f"x: {xid}")

print(f"xid_prev: {xid_prev}")

tdiff = 0.1
t_var = 'Time' # 'Interval'
int_var = 'cid'
# df[(df[t_var] >= iv - tdiff) & (df[t_var] <= iv + (window + tdiff)) & (df['Trial'] == int(x['trial']))]
# df[(df[t_var] >= float(x[int_var][0]) - tdiff) & (df[t_var] <= float(x[int_var][1] + tdiff)) & (df['Trial'] == int(x['trial']))]
df[(df[t_var] > float(x[int_var][0]) - tdiff) & (df[t_var] <= float(x['cid'][1] + tdiff)) & (df['Trial'] == int(x['trial']))]

t_var = 'Time' # 'Interval'
int_var = 'pid'
df[(df[t_var] > round(float(x[int_var][0]), 2) - tdiff) & (df[t_var] <= round(float(x[int_var][1]), 2)) & (df['Trial'] == int(x['trial']))]
"""
"""

x, y = next(iterable)

model.cpu()
features, logits, loss = model(x, y)

"""

"""

# df.groupby(['Interval', 'Trial']).size().plot.bar()
# df.groupby(['Interval', 'Trial']).agg(['nunique'])model_path
n_unique = len(df.groupby(['Interval', 'Trial']).size())
df.groupby(['Interval', 'Trial']).size().nlargest(int(0.2 * n_unique))
# df.groupby(['Interval_2', 'Trial']).size().mean()

"""

"""
x, y = next(iterable)
model.cuda()
for k, v in x.items():
    x[k] = v.cuda()
for k, v in y.items():
    y[k] = v.cuda()
features, logits, loss = model(x, y)

# add the loss from dict entries
total_loss = 0
for k, v in loss.items():
    if k == 'loss':
        total_loss += v
    else:
        total_loss += v.mean()

# backward pass
total_loss.backward()

"""


"""
==== CREATING DICTS FOR THE DATA ====

# %%
# from utils import df_to_dict

# dict_path = "data/LargeRandLIF2-2/LargeRandNet2_SpikeTime_dict.pkl"

# if not os.path.exists(dict_path):
#     print("Creating dictionary...")
#     df_dict = df_to_dict(df)
#     with open(dict_path, 'wb') as f:
#         pickle.dump(df_dict, f)
# else:
#     print("Loading dictionary...")
#     with open(dict_path, 'rb') as f:
#         df_dict = pickle.load(f)

# int_trials = df.groupby(['Interval', 'Trial']).size()
# print(int_trials.mean())
# # df.groupby(['Interval', 'Trial']).agg(['nunique'])
# var_group = 'Interval'
# n_unique = len(df.groupby([var_group, 'Trial']).size())
# df.groupby([var_group, 'Trial']).size().nlargest(int(0.2 * n_unique))
# # df.groupby(['Interval_2', 'Trial']).size().mean()

# var_group = 'Interval_2'
# n_unique = len(df.groupby([var_group, 'Trial']).size())
# df.groupby([var_group, 'Trial']).size().nlargest(int(0.2 * n_unique))
# # df.groupby([var_group, 'Trial']).size().mean()

# df.groupby([var_group, 'Trial']).size().nlargest(int(0.2 * n_unique))
# df.groupby(['Interval_2', 'Trial']).size().mean()

# n_unique = len(int_trials)
# int_trials.nlargest(int(0.2 * n_unique))



===================================
"""

"""
# get mean and std of stimulus
mean = [stimulus.mean(axis=d) for d in range(len(stimulus.shape))]
std = [stimulus.std(axis=d) for d in range(len(stimulus.shape))]
"""


"""
# randomly sample chunks for train and test sets
# import random
# r_split = 0.8
# n_chunks = len(chunks)
# n_train = int(n_chunks * r_split)
# idx_sample = random.sample(range(n_chunks), n_train)
# idx_rest = [i for i in range(n_chunks) if i not in idx_sample]
# train_chunks = np.concatenate([chunks[i] for i in idx_sample]).flatten()
# test_chunks = np.concatenate([chunks[i] for i in idx_rest]).flatten()
# test_chunks = np.concatenate([c for c in chunks if c.any() not in train_chunks]).flatten()

"""


"""
convolutional research

from einops.layers.torch import Rearrange

kernel_size_ = kernel_size[0], kernel_size[1] * 4, kernel_size[2] * 4
stride_size = 1, 8, 8
padding_size = 0, 0, 0

kernel_size = kernel_size
stride_size = stride_size if hasattr(model_conf, 'stride_size') else kernel_size
padding_size = padding_size if hasattr(model_conf, 'padding_size') else 0

print(f"kernel_size: {kernel_size_}, stride_size: {stride_size}, padding_size: {padding_size}")

conv_block = torch.nn.Sequential(
        nn.Conv3d(1, n_embd, kernel_size=stride_size, stride=kernel_size, padding=padding_size),
        Rearrange('b e t h w -> b t h w e'),
        nn.LayerNorm([n_embd]),
        nn.ReLU(),
)
# out = conv_block(x['frames'])    

conv_3d = nn.Conv3d(1, 256, kernel_size_, stride=stride_size, padding=padding_size, bias=False)
out = conv_3d(x['frames'])


print(f"frame: {x['frames'].shape}, out: {out.shape}")


# make grids

from neuroformer.utils import draw_grid
import cv2

img = video_stack[i, 1, 0].permute(0, 1).numpy()
# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
grid_image = draw_grid(img, out.shape[-2], out.shape[-1])
plt.figure(figsize=(10, 10))
plt.imshow(grid_image)
plt.axis('off')



"""

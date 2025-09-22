import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bkmeans import BKMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from emc.utils.data import grouped_decompose, windowed_decompose
from emc.utils.paths import get_paths

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "har", "output")
Path(output_dir_path).mkdir(parents=True, exist_ok=True)

# load data
act_lbl_idx_map = {
    "climbingdown": 1,
    "climbingup": 2,
    "jumping": 3,
    "lying": 4,
    "running": 5,
    "sitting": 6,
    "standing": 7,
    "walking": 8
}
act_idx_lbl_map = {y: x for x, y in act_lbl_idx_map.items()}
sensor = "chest"
subjects_optim = [1]
activities = [1,2,3,4,5,6,7,8]
data_lim = None
X = pd.DataFrame()
for subject in subjects_optim:
    for act in [act_idx_lbl_map[a] for a in activities]:
        data_path = os.path.join(paths["data"]["realworld_har"]["samples"], f"subject_{subject}", act, f"{sensor}.csv")
        data = pd.read_csv(data_path, sep=",", header=0, usecols=["attr_x","attr_y","attr_z"])[:data_lim]
        X = pd.concat([X, data], ignore_index=True)
        print(f"Subject {subject} - Action {act}: {data.shape}")
print(f"Total: {X.shape}")

# scaling
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# decomposition
prim_len = 5
X_dec = windowed_decompose(X_sc, prim_len)
print(X_dec.shape)

alphabet_dir_path = os.path.join(output_dir_path, "alphabet")
Path(alphabet_dir_path).mkdir(parents=True, exist_ok=True)

# clustering
clue = []
n_clusters_list = np.arange(4,21)
for n_clusters in tqdm(n_clusters_list):
    clusterer = BKMeans(n_clusters=n_clusters, m=20, random_state=42)
    X_cls = clusterer.fit_predict(X_dec)
    db = davies_bouldin_score(X_dec, X_cls)
    clue.append({"n_clusters": n_clusters, "davies_bouldin_score": db, "inertia": clusterer.inertia_})
clue_df = pd.DataFrame.from_records(clue)

# plot db scores
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
    db_plot = sns.lineplot(data=clue_df, x="n_clusters", y="davies_bouldin_score", ax=ax)
    ax.set_xticks(n_clusters_list)
    fig_name = f"s{subjects_optim[0]}_pl:{prim_len}_db.pdf"
    db_plot.get_figure().savefig(os.path.join(alphabet_dir_path, fig_name), bbox_inches="tight")
    plt.close()

# save alphabet
alp_car = 7
clusterer = BKMeans(n_clusters=alp_car, m=50, random_state=42)
clusterer.fit(X_dec)
alp_id = f"s{subjects_optim[0]}_ac:{alp_car}_pl:{prim_len}"
with open(os.path.join(alphabet_dir_path, f"{alp_id}.pkl"), "wb") as file:
    pickle.dump(clusterer, file)
    print(f"Alphabet {alp_id} saved")
with open(os.path.join(alphabet_dir_path, f"{alp_id}_sc.pkl"), "wb") as file:
    pickle.dump(scaler, file)
    print(f"Scaler {alp_id}_sc saved")

# plot primitives
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
    start = 0
    ticks = [0]
    for prm_ind, prm in enumerate(clusterer.cluster_centers_):
        end = start + prim_len*3
        ax.plot(np.arange(start, end), prm)
        start = end
        ticks.append(start)
    ax.set_xticks(ticks)
    fig.savefig(os.path.join(alphabet_dir_path, f"{alp_id}_primitives.pdf"), bbox_inches="tight")
    plt.close()

# --

# prepare data
act_idx_lbl_map = {y: x for x, y in act_lbl_idx_map.items()}
sensor = "chest"
subjects_optim = [1]
activities = [1,2,3,4,5,6,7,8]
number_of_runs = 10
number_of_regimes = 10
master_rng = np.random.default_rng(42)
local_seeds = master_rng.integers(low=0, high=999999, size=number_of_runs, dtype=int)

# optm runs
optm_runs = {}
pbar_conf = {"total": len(subjects_optim)*number_of_runs, "desc": "Generating optm runs"}
with tqdm(**pbar_conf) as pbar:
    for run_index in range(number_of_runs):
        rng = np.random.default_rng(local_seeds[run_index])
        # generate a random sequence of activities
        while True:
            rnd_act_idx_seq = rng.choice(a=activities, size=number_of_regimes)
            if 0 not in np.diff(rnd_act_idx_seq):
                break
        for subject in subjects_optim:
            # combine data of randomly generated activity sequence
            rnd_act_lbl_seq = [act_idx_lbl_map[a] for a in rnd_act_idx_seq]
            X_seq = pd.DataFrame()
            regime_lengths = []
            for act_lbl in rnd_act_lbl_seq:
                data_path = os.path.join(paths["data"]["realworld_har"]["samples"], f"subject_{subject}", act_lbl, f"{sensor}.csv")
                data = pd.read_csv(data_path, sep=",", header=0, usecols=["attr_x","attr_y","attr_z"])
                X_seq = pd.concat([X_seq, data], ignore_index=True)
                regime_lengths.append(len(data))
            X_sc_opt = scaler.transform(X_seq)
            grouped_target_signal = grouped_decompose(X_sc_opt, prim_len)
            prim_id_seq = clusterer.predict(grouped_target_signal)
            optm_runs[f"s{subject}_r{run_index+1}"] = {
                "subprocess_sequence": rnd_act_idx_seq,
                "regime_lengths": regime_lengths,
                "change_points": np.insert(np.cumsum(regime_lengths), 0, 0),
                "discrete_sequence": prim_id_seq,
                "continuous_sequence": X_sc_opt,
            }
            pbar.update()

# test runs
subjects_test = np.setdiff1d(np.arange(1,16), subjects_optim)
test_runs = {}
pbar_conf = {"total": len(subjects_test)*number_of_runs, "desc": "Generating test runs"}
with tqdm(**pbar_conf) as pbar:
    for run_index in range(number_of_runs):
        rng = np.random.default_rng(local_seeds[run_index])
        # generate a random sequence of activities
        while True:
            rnd_act_idx_seq = rng.choice(a=activities, size=number_of_regimes)
            if 0 not in np.diff(rnd_act_idx_seq):
                break
        for subject in subjects_test:
            # combine data of randomly generated activity sequence
            rnd_act_lbl_seq = [act_idx_lbl_map[a] for a in rnd_act_idx_seq]
            X_seq = pd.DataFrame()
            regime_lengths = []
            for act_lbl in rnd_act_lbl_seq:
                data_path = os.path.join(paths["data"]["realworld_har"]["samples"], f"subject_{subject}", act_lbl, f"{sensor}.csv")
                data = pd.read_csv(data_path, sep=",", header=0, usecols=["attr_x","attr_y","attr_z"])
                X_seq = pd.concat([X_seq, data], ignore_index=True)
                regime_lengths.append(len(data))
        
            X_sc_test = scaler.transform(X_seq)
            grouped_target_signal = grouped_decompose(X_sc_test, prim_len)
            prim_id_seq = clusterer.predict(grouped_target_signal)
            change_points = np.insert(np.cumsum(regime_lengths), 0, 0)
            test_runs[f"s{subject}_r{run_index+1}"] = {
                "subprocess_sequence": rnd_act_idx_seq,
                "regime_lengths": regime_lengths,
                "change_points": change_points,
                "change_points_scaled": [int(cp/prim_len) for cp in change_points],
                "discrete_sequence": prim_id_seq,
                "continuous_sequence": X_sc_test,
            }
            pbar.update()

# prepare data
data = {
    "meta": {
        "alp_car": alp_car,
        "prim_len": prim_len
    },
    "optm": optm_runs,
    "test": test_runs
}

# save data
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "har", "output")
Path(output_dir_path).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(output_dir_path, "data.pkl")
with open(data_path, "wb") as file:
    pickle.dump(data, file)
print(f"data saved: {data_path}")

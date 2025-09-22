import os
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from natsort import natsort_keygen
from river.cluster import DBSTREAM, CluStream
from river.stream import iter_array
from sklearn.metrics import cluster

from emc.estimator.EMC import EMCLin as EMC
from emc.estimator.EPSTM import run_epstm
from emc.utils.loader import load_json_file, load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.plot import get_mpl_conf_path, plot_mode_transition, set_size
from emc.utils.sequence import map_symbols

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "eeg", "output")
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")

# load parameters
params_path = os.path.join(output_dir_path, "params.json")
params = load_json_file(params_path)
if params is not None:
    print(f"\n{params_path} loaded.")
    # print(params)

opt_index = 2927
test_ms_seq = data["ms_seq"][opt_index:]
test_labels = data["Y"][opt_index:]
test_X = data["X"][opt_index:, :]
cps_test = np.where(np.diff(test_labels, prepend=np.nan))[0].tolist()
cps_test = cps_test + [len(test_labels)]
# print(f"CPs test: {cps_test}")

methods_to_run = [
    "emc",
    "clustream",
    "dbstream",
    "epstm_tcp",
    "epstm_icp",
    "hmm",
]
results = []

# execute test runs
if "emc" in methods_to_run:
    emc_ins = EMC(
        alpha=data["meta"]["n_clusters"],
        order=params["emc"]["order"],
        lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
        beta=params["emc"]["beta"],
        delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
        eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
        tau=params["emc"]["tau"],
    )
    emc_ins.process_sequence(data["ms_seq"], progress=False)

    # ARI
    labels_true = test_labels
    labels_pred = np.array(emc_ins.pred_mode_hist)[opt_index:]
    emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    stationarity_history = emc_ins.stationarity_history[opt_index:]
    emc_ari_s = cluster.adjusted_rand_score(
        labels_pred=labels_pred[np.argwhere(stationarity_history)].flatten(),
        labels_true=labels_true[np.argwhere(stationarity_history)].flatten()
    )

    print(f"EMC ARI: {emc_ari:.2f} - {emc_ari_s:.2f}")
    results.append({
        "method": "EMC (Ours)",
        "op_type": "online",
        "cps": "not given",
        "ari": emc_ari
    })
    
    results.append({
        "method": r"EMC (Ours) ($\phi=1$)",
        "op_type": "online",
        "cps": "not given",
        "ari": emc_ari_s
    })

if "clustream" in methods_to_run:

    clustream_ins = CluStream(
        n_macro_clusters=params["clustream"]["n_macro_clusters"],
        max_micro_clusters=params["clustream"]["max_micro_clusters"],
        micro_cluster_r_factor=params["clustream"]["micro_cluster_r_factor"],
        time_window=params["clustream"]["time_window"],
        time_gap=params["clustream"]["time_gap"],
        seed=42,
    )
    labels_pred_clustream = []
    for i, (x, _) in enumerate(iter_array(test_X)):
        clustream_ins.learn_one(x)
        labels_pred_clustream.append(clustream_ins.predict_one(x))

    labels_true = test_labels
    labels_pred = labels_pred_clustream
    clustream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    print(f"CluStream ARI: {clustream_ari:.2f}")
    results.append({
        "method": "CluStream",
        "op_type": "online",
        "cps": "not given",
        "ari": clustream_ari
    })

if "dbstream" in methods_to_run:

    dbstream_ins = DBSTREAM(
        clustering_threshold=params["dbstream"]["clustering_threshold"],
        fading_factor=params["dbstream"]["fading_factor"],
        cleanup_interval=params["dbstream"]["cleanup_interval"],
        intersection_factor=params["dbstream"]["intersection_factor"],
        minimum_weight=params["dbstream"]["minimum_weight"]
    )
    labels_pred_dbstream = []
    for i, (x, _) in enumerate(iter_array(test_X)):
        dbstream_ins.learn_one(x)
        labels_pred_dbstream.append(dbstream_ins.predict_one(x))
    labels_true = test_labels
    labels_pred = labels_pred_dbstream
    dbstream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    print(f"DBStream ARI: {dbstream_ari:.2f}")
    results.append({
        "method": "DBStream",
        "op_type": "online",
        "cps": "not given",
        "ari": dbstream_ari
    })

if "epstm_tcp" in methods_to_run:

    epstm_tcp_labels_pred = run_epstm(
        pid_list=set(data["ms_seq"]),
        pid_seq=test_ms_seq,
        change_points=cps_test,
        subsequence_minimum_occurrence=params["epstm"]["subsequence_minimum_occurrence"],
        subsequence_length_limit=params["epstm"]["subsequence_length_limit"],
        I=params["epstm"]["I"],
        clustering_threshold=params["epstm"]["clustering_threshold"]
    )

    # evaluate
    labels_true = test_labels
    length_diff = len(labels_true) - len(epstm_tcp_labels_pred)
    if length_diff > 0:
        print(f"length diff: {length_diff}, labels_true: {len(labels_true)}, epstm_tcp_labels_pred: {len(epstm_tcp_labels_pred)}")
        labels_true = labels_true[:-length_diff]
    epstm_tcp_ari = cluster.adjusted_rand_score(labels_pred=epstm_tcp_labels_pred, labels_true=labels_true)
    print(f"EPSTM-TCP ARI: {epstm_tcp_ari:.2f}")
    results.append({
        "method": "EPSTM",
        "op_type": "offline",
        "cps": "given (true)",
        "ari": epstm_tcp_ari
    })

if "epstm_icp" in methods_to_run:

    delay_low, delay_high = [70, 160]
    fp, fn = [0, 0]
    rng = np.random.default_rng(42)
    imperfect_cps = [0]
    cursor = 0
    for true_cp_idx, true_cp in enumerate(cps_test[1:]):
        # print(true_cp_idx, true_cp, len(cps_test) - 2)
        if true_cp_idx == len(cps_test) - 2:  # last change point
            delay = 0 # no delay in the final change point
        else:
            delay = rng.integers(low=delay_low, high=delay_high, endpoint=True)
        start = cursor
        end = true_cp + delay
        cursor = end
        if end >= len(test_ms_seq):
            end = true_cp
        imperfect_cps.append(end)
    if fp > 0:
        imperfect_cps.extend(rng.integers(len(test_ms_seq), size=1))
        imperfect_cps = np.sort(imperfect_cps)
    # print(f"Imperfect change points: {imperfect_cps}")

    epstm_icp_labels_pred = run_epstm(
        pid_list=set(data["ms_seq"]),
        pid_seq=test_ms_seq,
        change_points=imperfect_cps,
        subsequence_minimum_occurrence=params["epstm"]["subsequence_minimum_occurrence"],
        subsequence_length_limit=params["epstm"]["subsequence_length_limit"],
        I=params["epstm"]["I"],
        clustering_threshold=params["epstm"]["clustering_threshold"]
    )

    # evaluate
    labels_true = test_labels
    length_diff = len(labels_true) - len(epstm_icp_labels_pred)
    if length_diff > 0:
        print(f"length diff: {length_diff}, labels_true: {len(labels_true)}, epstm_icp_labels_pred: {len(epstm_icp_labels_pred)}")
        labels_true = labels_true[:-length_diff]
    epstm_icp_ari = cluster.adjusted_rand_score(labels_pred=epstm_icp_labels_pred, labels_true=labels_true)
    print(f"EPSTM-ICP ARI: {epstm_icp_ari:.2f}")
    results.append({
        "method": "EPSTM",
        "op_type": "offline",
        "cps": "given (imperfect)",
        "ari": epstm_icp_ari
    })

# HMM
if "hmm" in methods_to_run:
    obs_seq, symbol_map = map_symbols(data["ms_seq"])
    obs_seq = obs_seq.reshape(-1, 1)

    # print(f"obs_seq unique: {np.unique(obs_seq)}")
    n_features = len(np.unique(obs_seq))
    n_components = len(set(data["Y"]))
    # print(f"n_components: {n_components}")

    best_hmm_ari = 0
    best_hmm_state_seq = None
    hmm_ari_list = []
    for i in range(100):
        model = hmm.CategoricalHMM(
            n_components=n_components,
            n_features=n_features,
            n_iter=100,
            tol=0.001,
            init_params="ste",
            params="ste",
            random_state=42+i,
            verbose=False
        )
        model.fit(obs_seq)
        logprob, hmm_state_seq = model.decode(obs_seq, algorithm="viterbi")

        hmm_ari = cluster.adjusted_rand_score(data["Y"], hmm_state_seq)
        hmm_ari_list.append(hmm_ari)

        if hmm_ari > best_hmm_ari:
            best_hmm_ari = hmm_ari
            best_hmm_state_seq = hmm_state_seq

    # print(f"HMM ARI: {np.mean(hmm_ari_list):.3f} +/- {np.std(hmm_ari_list):.3f}")
    print(f"HMM ARI (max): {np.max(hmm_ari_list):.3f}")
    results.append({
        "method": "HMM",
        "op_type": "offline",
        "cps": "not given",
        "ari": best_hmm_ari
    })

results.append({
    "method": "BOSD",
    "op_type": "offline",
    "cps": "given",
    "ari": 0.07
})

results_df = pd.DataFrame(results)

# sort
results_df = results_df.sort_values(by=["op_type", "method"], ascending=[False, True], key=natsort_keygen())
methods_to_top = ["EMC (Ours)", r"EMC (Ours) ($\phi=1$)"]
results_df = pd.concat([results_df[results_df["method"].isin(methods_to_top)], results_df[~results_df["method"].isin(methods_to_top)]])

# save results
results_file_name = "eeg_ari.csv"
results_df.to_csv(os.path.join(results_dir_path, results_file_name), index=False, float_format="%.2f")
print(f"Results saved: {os.path.join(results_dir_path, results_file_name)}")

mode_id_to_label_map = {1: "eyes\nopen", 2: "eyes\nclosed"}
change_points = np.where(np.diff(data["Y"], prepend=np.nan))[0].tolist()
mpl_conf_override = {"axes.labelsize":6, "font.size":6, "xtick.labelsize":4, "ytick.labelsize":6, "legend.fontsize":6}
plot_mode_transition(
    subplots=["true_modes","data","discovered_modes"],
    labels_true=data["Y"]+1,
    data=data["ms_seq"],
    drift_scores=emc_ins.drift_scores,
    labels_pred=emc_ins.pred_mode_hist,
    change_points=change_points,
    cp_scale_coeff=emc_ins.tau,
    data_label="Microstate\nSequence",
	data_yticks=[-1,0,1,2,3,4,5,6,7,8],
    data_include_modes=False,
    # ns_regions=ns_regions,
    legend_text=f"ARI: {emc_ari:.2}",
    legend_loc="upper right",
    mode_id_to_label_map=mode_id_to_label_map,
    xtick_rotation=60,
    xtick_ids_to_pad=[0,18],
    xtick_ids_to_skip=[7,16,18,19],
    mpl_conf_path=get_mpl_conf_path("pub"),
    mpl_conf_override=mpl_conf_override,
    fig_size=set_size(**{"width":"tpami_half", "aspect_ratio": 1.25, "fraction":1}),
    fig_path=os.path.join(results_dir_path, "eeg_mt.pdf"),
)
print(f"figure saved: {results_dir_path}/eeg_mt.pdf")

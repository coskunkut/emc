import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from natsort import natsort_keygen
from river.cluster import DBSTREAM, CluStream
from river.stream import iter_array
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import cluster
from tqdm.auto import tqdm

from emc.estimator.EMC import EMCLin as EMC
from emc.estimator.EPSTM import construct_pst, match_pst_pairs
from emc.utils.loader import load_json_file, load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.plot import get_mpl_conf_path, plot_mode_transition, set_size
from emc.utils.sequence import map_symbols

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "har", "output")
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")
alp_car = data["meta"]["alp_car"]
prim_len = data["meta"]["prim_len"]
optm_runs = data["optm"]
test_runs = data["test"]

# load parameters
params_path = os.path.join(output_dir_path, "params.json")
params = load_json_file(params_path)
if params is not None:
    print(f"{params_path} loaded")

methods_to_run = [
    "emc",
    "clustream",
    "dbstream",
    "epstm_icp"
    "epstm_tcp",
    "hmm",
]

def execute_single_run(run_desc):

    run_id, run_params = run_desc
    result_dict = {"run_id": run_id, "subject": run_id.split("_")[0]}

    labels_true = np.repeat(run_params["subprocess_sequence"], run_params["regime_lengths"])

    # EMC
    if "emc" in methods_to_run:
        emc_ins = EMC(
            alpha=alp_car,
            order=params["emc"]["order"],
            lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
            beta=params["emc"]["beta"],
            delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
            eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
            tau=params["emc"]["tau"],
        )
        emc_ins.process_sequence(run_params["discrete_sequence"], progress=False)
        labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
        length_diff = len(labels_true) - len(labels_pred)
        if length_diff > 0:
            labels_true = labels_true[:-length_diff]
        emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
        result_dict["emc:ari"] = emc_ari
        stationarity_history = np.repeat(emc_ins.stationarity_history, prim_len)
        emc_ari_s = cluster.adjusted_rand_score(
            labels_pred=labels_pred[np.argwhere(stationarity_history)].flatten(),
            labels_true=labels_true[np.argwhere(stationarity_history)].flatten()
        )
        result_dict["emc:ari_s"] = emc_ari_s
        # result_dict["drift_ratio"] = np.count_nonzero(stationarity_history==0)/len(stationarity_history)

    # CluStream
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
        for i, (x, _) in enumerate(iter_array(run_params["continuous_sequence"])):
            clustream_ins.learn_one(x)
            labels_pred_clustream.append(clustream_ins.predict_one(x))
        clustream_ari = cluster.adjusted_rand_score(
            labels_pred=labels_pred_clustream,
            labels_true=labels_true
        )
        result_dict["clustream:ari"] = clustream_ari

    # DBStream
    if "dbstream" in methods_to_run:
        dbstream_ins = DBSTREAM(
            clustering_threshold=params["dbstream"]["clustering_threshold"],
            fading_factor=params["dbstream"]["fading_factor"],
            cleanup_interval=params["dbstream"]["cleanup_interval"],
            intersection_factor=params["dbstream"]["intersection_factor"],
            minimum_weight=params["dbstream"]["minimum_weight"]
        )
        labels_pred_dbstream = []
        for i, (x, _) in enumerate(iter_array(run_params["continuous_sequence"])):
            dbstream_ins.learn_one(x)
            labels_pred_dbstream.append(dbstream_ins.predict_one(x))
        dbstream_ari = cluster.adjusted_rand_score(
            labels_pred=labels_pred_dbstream,
            labels_true=labels_true
        )
        result_dict["dbstream:ari"] = dbstream_ari

    # EPSTM (with true change points)
    if "epstm_tcp" in methods_to_run:
        psts = []
        pid_list = set(run_params["discrete_sequence"])
        cursor = 0
        for cp_idx, cp in enumerate(run_params["change_points_scaled"][1:]):
            start = cursor
            end = cp
            pid_seq = run_params["discrete_sequence"][start:end][:500]
            pst = construct_pst(
                pst_id=f"{run_id}:reg_{start}_{end}",
                pid_list=pid_list,
                pid_seq=pid_seq,
                subsequence_minimum_occurrence=2,
                subsequence_length_limit=1,
                smoothing_min_p=0.001,
                to_render=False,
                to_save=False,
                output_dir_path=None,
                log_level="DEBUG"
            )
            psts.append(pst)
            cursor = end
        dist_pairs, dist_matrix = match_pst_pairs(
            pst_list=psts,
            matching_function="epstm_2020",
            matching_parameters={"I": 0.5}
        )
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.1,
            metric="precomputed",
            linkage="single"
        )
        clusterer.fit(dist_matrix)
        epstm_tcp_labels_pred = []
        for regime_idx, regime_length in enumerate(np.diff(run_params["change_points_scaled"])):
            epstm_tcp_labels_pred.extend(np.repeat(clusterer.labels_[regime_idx], regime_length))
        epstm_tcp_labels_pred = np.repeat(epstm_tcp_labels_pred, prim_len)
        length_diff = len(labels_true) - len(epstm_tcp_labels_pred)
        if length_diff > 0:
            labels_true = labels_true[:-length_diff]
        epstm_tcp_ari = cluster.adjusted_rand_score(labels_pred=epstm_tcp_labels_pred, labels_true=labels_true)
        result_dict["epstm_tcp:ari"] = epstm_tcp_ari

    # EPSTM (with imperfect change points)
    if "epstm_icp" in methods_to_run:
        true_cps_scaled = run_params["change_points_scaled"]
        delay_low, delay_high = [70, 160]
        fp, _ = [0, 0]
        rng = np.random.default_rng(int(run_id.split("_")[1][1:]))
        imperfect_cps = []
        cursor = 0
        for true_cp_idx, true_cp in enumerate(true_cps_scaled[1:]):
            if true_cp_idx == len(run_params["regime_lengths"])-1:
                delay = 0 # no delay in the final change point
            else:
                delay = rng.integers(low=delay_low, high=delay_high, endpoint=True)
            start = cursor
            end = true_cp + delay
            cursor = end
            imperfect_cps.append(end)
        if fp > 0:
            imperfect_cps.extend(rng.integers(len(run_params["discrete_sequence"]), size=1))
            imperfect_cps = np.sort(imperfect_cps)
        psts = []
        pid_list = set(run_params["discrete_sequence"])
        cursor = 0
        for cp_idx, cp in enumerate(imperfect_cps):
            start = cursor
            end = cp
            pid_seq = run_params["discrete_sequence"][start:end][:500]
            pst = construct_pst(
                pst_id=f"{run_id}:reg_{start}_{end}",
                pid_list=pid_list,
                pid_seq=pid_seq,
                subsequence_minimum_occurrence=2,
                subsequence_length_limit=1,
                smoothing_min_p=0.001,
                to_render=False,
                to_save=False,
                output_dir_path=None,
                log_level="DEBUG"
            )
            psts.append(pst)
            cursor = end
        dist_pairs, dist_matrix = match_pst_pairs(
            pst_list=psts,
            matching_function="epstm_2020",
            matching_parameters={"I": 0.5}
        )
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.1,
            metric="precomputed",
            linkage="single"
        )
        clusterer.fit(dist_matrix)
        epstm_tcp_labels_pred = []
        for regime_idx, regime_length in enumerate(np.diff(imperfect_cps)):
            epstm_tcp_labels_pred.extend(np.repeat(clusterer.labels_[regime_idx], regime_length))
        epstm_tcp_labels_pred = np.repeat(epstm_tcp_labels_pred, prim_len)
        length_diff = len(labels_true) - len(epstm_tcp_labels_pred)
        if length_diff > 0:
            labels_true = labels_true[:-length_diff]
        epstm_tcp_ari = cluster.adjusted_rand_score(labels_pred=epstm_tcp_labels_pred, labels_true=labels_true)
        result_dict["epstm_icp:ari"] = epstm_tcp_ari

    # HMM
    if "hmm" in methods_to_run:
        n_components = len(set(run_params["subprocess_sequence"]))
        obs_seq, symbol_map = map_symbols(run_params["discrete_sequence"])
        obs_seq = obs_seq.reshape(-1, 1)
        n_features = len(np.unique(obs_seq))
        model = hmm.CategoricalHMM(
            n_components=n_components,
            n_features=n_features,
            n_iter=100,
            tol=0.001,
            init_params="ste",
            params="ste",
            random_state=42,
            verbose=False
        )
        model.fit(obs_seq)
        logprob, hmm_state_seq = model.decode(obs_seq, algorithm="viterbi")
        hmm_state_seq = np.repeat(hmm_state_seq, prim_len)
        hmm_ari = cluster.adjusted_rand_score(labels_true, hmm_state_seq)
        result_dict["hmm:ari"] = hmm_ari

    return result_dict

# run tasks in parallel
tasks = [(k,p) for k,p in test_runs.items()]
with ProcessPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(execute_single_run, task) for task in tasks]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Executing runs"):
        result = future.result()
        results.append(result)

# run logs
results_df = pd.DataFrame.from_records(results)
run_logs_path = os.path.join(results_dir_path, "har_run_logs_emc.csv")
results_df.to_csv(run_logs_path, index=False, header=True, float_format="%.3f")
print(f"run logs saved: {run_logs_path}")

# format results
results_df_frm = results_df\
    .drop(columns=["run_id"])\
    .groupby("subject")\
    .agg(lambda x: fr"${x.mean():.2f}\pm{x.std():.2f}$")\
    .reset_index()\
    .sort_values(by="subject", ascending=True, key=natsort_keygen())
results_df_frm["subject"] = results_df_frm["subject"].str.replace("s", "")
avg_row = results_df.copy()\
    .drop(columns=["run_id"])\
    .apply(pd.to_numeric, errors="coerce")\
    .agg(lambda x: fr"${x.mean():.2f}\pm{x.std():.2f}$")\
    .to_frame()\
    .transpose()
avg_row.at[0, "subject"] = "Overall"
results_df_frm = pd.concat([results_df_frm, avg_row], ignore_index=True, sort=False)

# save results
results_path = os.path.join(results_dir_path, "har_ari_emc.csv")
results_df_frm.to_csv(results_path, index=False, header=True, float_format="%.3f")
print(f"results saved: {results_path}")

print(results_df_frm)

# visualization
# run_id = "s9_r3"
run_id = results_df.loc[results_df["emc:ari"].idxmax()]["run_id"]
run = test_runs[run_id]
print(f"Visualizing {run_id}")
emc_ins = EMC(
    alpha=alp_car,
    order=params["emc"]["order"],
    lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
    beta=params["emc"]["beta"],
    delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
    eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
    tau=params["emc"]["tau"],
)
emc_ins.process_sequence(run["discrete_sequence"], progress=True)
labels_true = np.repeat(run["subprocess_sequence"], run["regime_lengths"])
labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
length_diff = len(labels_true) - len(labels_pred)
if length_diff > 0:
    labels_true = labels_true[:-length_diff]
emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
plot_mode_transition(
    subplots=["true_modes", "discovered_modes"],
    labels_true=labels_true,
    data=None,
    drift_scores=emc_ins.drift_scores,
    labels_pred=labels_pred,
    change_points=run["change_points"],
    cp_scale_coeff=prim_len*emc_ins.tau,
    data_label=None,
    data_yticks=None,
    data_include_modes=False,
    legend_text=f"ARI: {emc_ari:.2f}",
    mode_id_to_label_map={1:"CLIMB↓", 2:"CLIMB↑", 3:"JUMP", 4:"LIE", 5:"RUN", 6:"SIT", 7:"STAND", 8:"WALK"},
    xtick_rotation=45,
    xtick_ids_to_pad=[3,5],
    mpl_conf_path=get_mpl_conf_path("pub"),
    fig_size=set_size(**{"width":"tpami_half", "aspect_ratio": 1.75, "fraction": 1}),
    fig_path=os.path.join(results_dir_path, "har_mt.pdf"),
)

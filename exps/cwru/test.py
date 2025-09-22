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
from tqdm.autonotebook import tqdm

from emc.estimator.EMC import EMCLin as EMC
from emc.estimator.EPSTM import construct_pst, match_pst_pairs
from emc.utils.data import grouped_decompose
from emc.utils.loader import load_cwru_data, load_json_file, load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.plot import get_mpl_conf_path, plot_mode_transition, set_size
from emc.utils.sequence import map_symbols

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "cwru", "output")
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)

# load alphabet
alphabet_path = os.path.join(alphabet_dir_path, "ac:27_pl:2.pkl")
alphabet = load_pickle_file(alphabet_path)
if alphabet is not None:
    print(f"{alphabet_path} loaded.")
    prim_len = len(alphabet.cluster_centers_[0])
    print(f"primitive length: {prim_len}")
    alp_car = len(alphabet.cluster_centers_)
    print(f"alphabet cardinality: {alp_car}")

# load parameters
params_path = os.path.join(output_dir_path, "params.json")
params = load_json_file(params_path)
if params is not None:
    print(f"\n{params_path} loaded.")
    # print(params)

# prepare test runs
test_runs = {}
sample_cnd_seqs = [
    # LOAD 0
    ["L0-OK","L0-IR:07"],
    ["L0-OK","L0-IR:14"],
    ["L0-OK","L0-IR:21"],
    ["L0-OK","L0-IR:28"],
    ["L0-OK","L0-IR:07","L0-IR:14"],
    ["L0-OK","L0-IR:14","L0-IR:21"],
    ["L0-OK","L0-IR:21","L0-IR:28"],
    # LOAD 1
    ["L1-OK","L1-IR:07"],
    ["L1-OK","L1-IR:14"],
    ["L1-OK","L1-IR:21"],
    ["L1-OK","L1-IR:28"],
    ["L1-OK","L1-IR:07","L1-IR:14"],
    ["L1-OK","L1-IR:14","L1-IR:21"],
    ["L1-OK","L1-IR:21","L1-IR:28"],
    # LOAD 2
    ["L2-OK","L2-IR:07"],
    ["L2-OK","L2-IR:14"],
    ["L2-OK","L2-IR:21"],
    ["L2-OK","L2-IR:28"],
    ["L2-OK","L2-IR:07","L2-IR:14"],
    ["L2-OK","L2-IR:14","L2-IR:21"],
    ["L2-OK","L2-IR:21","L2-IR:28"],
    # LOAD 3
    ["L3-OK","L3-IR:07"],
    ["L3-OK","L3-IR:14"],
    ["L3-OK","L3-IR:21"],
    ["L3-OK","L3-IR:28"],
    ["L3-OK","L3-IR:07","L3-IR:14"],
    ["L3-OK","L3-IR:14","L3-IR:21"],
    ["L3-OK","L3-IR:21","L3-IR:28"],
]

for sample_cnd_seq in tqdm(sample_cnd_seqs, desc="Prepare tasks", leave=True):

    # load data
    X_tst, cp_tst, y_true_tst = load_cwru_data(condition_sequence=sample_cnd_seq, points_per_condition=12000)

    # reconstruct
    symbol_sequence_test = alphabet.predict(grouped_decompose(X_tst, prim_len))

    # save test run
    test_runs[f"{sample_cnd_seq}"] = {
        "X": X_tst,
        "symbol_sequence": symbol_sequence_test,
        "change_points": cp_tst,
        "regime_lengths": np.diff(cp_tst),
        "labels_true": y_true_tst,
        "sample_cnd_seq": sample_cnd_seq,
        "continuous_sequence": X_tst,
        "change_points_scaled": [int(cp/prim_len) for cp in cp_tst],
    }

methods_to_run = [
    "emc",
    "clustream",
    "dbstream",
    "epstm_tcp",
    "epstm_icp",
    "hmm",
]

# execute test runs
def execute_single_run(run_desc):

    run_id, run_params = run_desc
    run_id_san = run_id.replace("[","").replace("]","").replace("'","").replace(", ","→")
    labels_true = np.array(run_params["labels_true"])
    # print(run_id_san)

    result = {
        "run_id": run_id_san,
        "load": run_params["sample_cnd_seq"][0][1], # for table
        "floc": run_params["sample_cnd_seq"][1][3:5],
        "num_regs": run_id_san.count("→")+1, # used for sorting rows
    }

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
        emc_ins.process_sequence(run_params["symbol_sequence"], progress=False)

        labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
        emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
        result["emc:ari"] = emc_ari

        stationarity_history = np.repeat(emc_ins.stationarity_history, prim_len)
        emc_ari_s = cluster.adjusted_rand_score(
            labels_pred=labels_pred[np.argwhere(stationarity_history)].flatten(),
            labels_true=labels_true[np.argwhere(stationarity_history)].flatten()
        )
        result["emc:ari_s"] = emc_ari_s
        # result["drift_ratio"] = np.count_nonzero(stationarity_history==0)/len(stationarity_history)

    # HMM
    if "hmm" in methods_to_run:
        n_components = len(set(run_params["labels_true"]))
        obs_seq, symbol_map = map_symbols(run_params["symbol_sequence"])
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
        result["hmm:ari"] = hmm_ari

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
        result["clustream:ari"] = clustream_ari

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
        result["dbstream:ari"] = dbstream_ari

    # EPSTM (with true change points)
    if "epstm_tcp" in methods_to_run:
        psts = []
        pid_list = set(run_params["symbol_sequence"])
        cursor = 0
        for cp_idx, cp in enumerate(run_params["change_points_scaled"][1:]):
            start = cursor
            end = cp
            pid_seq = run_params["symbol_sequence"][start:end]
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
        labels_true = np.array(run_params["labels_true"])
        epstm_tcp_ari = cluster.adjusted_rand_score(labels_pred=epstm_tcp_labels_pred, labels_true=labels_true)
        result["epstm_tcp:ari"] = epstm_tcp_ari

    # EPSTM (with imperfect change points)
    if "epstm_icp" in methods_to_run:
        true_cps_scaled = run_params["change_points_scaled"]
        delay_low, delay_high = [70, 160]
        fp, _ = [1, 0]
        rng = np.random.default_rng(42)
        imperfect_cps = [0]
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
            imperfect_cps.extend(rng.integers(len(run_params["symbol_sequence"]), size=1))
            imperfect_cps = np.sort(imperfect_cps)
        # print(f"{run_id_san} imperfect cps (scaled): {imperfect_cps} | {true_cps_scaled}")
        psts = []
        pid_list = set(run_params["symbol_sequence"])
        cursor = 0
        for cp_idx, cp in enumerate(imperfect_cps[1:]):
            start = cursor
            end = cp
            pid_seq = run_params["symbol_sequence"][start:end]
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
        epstm_icp_labels_pred = []
        for regime_idx, regime_length in enumerate(np.diff(imperfect_cps)):
            epstm_icp_labels_pred.extend(np.repeat(clusterer.labels_[regime_idx], regime_length))
        epstm_icp_labels_pred = np.repeat(epstm_icp_labels_pred, prim_len)
        labels_true = np.array(run_params["labels_true"])
        epstm_tcp_ari = cluster.adjusted_rand_score(labels_pred=epstm_icp_labels_pred, labels_true=labels_true)
        result["epstm_icp:ari"] = epstm_tcp_ari

    # save figure
    if "emc" in methods_to_run:
        R = len(run_params["sample_cnd_seq"]) # number of modes
        mode_id_to_label_map = {k:run_params["sample_cnd_seq"][k].replace("IR","FD") for k in range(0,R)}
        plot_mode_transition(
            subplots=["data", "discovered_modes"],
            labels_true=np.array(run_params["labels_true"]),
            data=run_params["X"],
            drift_scores=emc_ins.drift_scores,
            labels_pred=labels_pred,
            change_points=run_params["change_points"],
            cp_scale_coeff=emc_ins.tau * prim_len,
            data_label="Vibration",
            data_yticks=None,
            data_include_modes=True,
            legend_text=f"ARI: {emc_ari:.2f}",
            legend_loc="upper left",
            mode_id_to_label_map=mode_id_to_label_map,
            xlabel="Time (s)",
            xticklabel_coeff=(1/12000),
            xtick_rotation=0,
            xtick_ids_to_pad=[],
            xtick_ids_to_skip=[],
            mpl_conf_path=get_mpl_conf_path("pub"),
            mpl_conf_override=None,
            fig_size=set_size(**{"width":"tpami_half", "aspect_ratio": 1.75, "fraction": 1}),
            fig_path=os.path.join(results_dir_path, f"{run_id_san}.pdf"),
        )

    return result

# run tasks in parallel
tasks = [(k,p) for k,p in test_runs.items()]
with ProcessPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(execute_single_run, task) for task in tasks]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Execute tasks", leave=True):
        result = future.result()
        results.append(result)

# format results
results_df = pd.DataFrame.from_records(results)
results_df_frm = results_df.copy()
# results_df_frm["load"] = results_df_frm["load"].map(lambda x: f"{x} HP")
results_df_frm = results_df_frm.sort_values(by=["load","num_regs","run_id"], key=natsort_keygen())
# results_df_frm["run_id"] = results_df_frm["run_id"].str.replace("→","/")
results_df_frm["run_id"] = results_df_frm["run_id"].str.replace("IR:","")
results_df_frm["run_id"] = results_df_frm["run_id"].str.replace("L.-","", regex=True)
results_df_frm = results_df_frm.drop(columns=["floc", "num_regs"])
# results_df_frm = results_df_frm[["run_id","load","emc_ari","emc_ari_s"]]
results_df_frm.loc["Avg"] = results_df_frm.mean(numeric_only=True)
results_df_frm.loc["Avg","run_id"] = "Avg."

print(results_df_frm)

# save results
results_path = os.path.join(results_dir_path, "cwru_ari.csv")
results_df_frm.to_csv(results_path, index=False, header=True, float_format="%.2f")
print(f"results saved: {results_path}")

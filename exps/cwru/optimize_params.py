import json
import logging
import multiprocessing
import os

import numpy as np
import optuna
from river.cluster import DBSTREAM, CluStream
from river.stream import iter_array
from sklearn.metrics import cluster
from tqdm.autonotebook import tqdm

from emc.estimator.EMC import EMCLin as EMC
from emc.utils.data import grouped_decompose
from emc.utils.database import create_optuna_db
from emc.utils.loader import load_cwru_data, load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "cwru", "output")
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")

# load alphabet
alphabet_path = os.path.join(alphabet_dir_path, "ac:27_pl:2.pkl")
alphabet = load_pickle_file(alphabet_path)
if alphabet is not None: 
    print(f"{alphabet_path} loaded")
    prim_len = len(alphabet.cluster_centers_[0])
    print(f"  primitive length: {prim_len}")
    alp_car = len(alphabet.cluster_centers_)
    print(f"  alphabet cardinality: {alp_car}")

# load data
X_opt, cp_opt, y_true_opt = load_cwru_data(condition_sequence=["L0-IR:07","L0-IR:28"], points_per_condition=12000)

# discretize
symbol_sequence_opt = alphabet.predict(grouped_decompose(X_opt, prim_len))

# params
params = {}

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('optm.log')
fh.setLevel(logging.DEBUG) # or any level you want
logger.addHandler(fh)

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
for sample_cnd_seq in tqdm(sample_cnd_seqs):
    X_tst, cp_tst, y_true_tst = load_cwru_data(condition_sequence=sample_cnd_seq, points_per_condition=12000)
    symbol_sequence_test = alphabet.predict(grouped_decompose(X_tst, prim_len))
    test_runs[f"{sample_cnd_seq}"] = {
        "X": X_tst,
        "symbol_sequence": symbol_sequence_test,
        "change_points": cp_tst,
        "labels_true": y_true_tst,
        "sample_cnd_seq": sample_cnd_seq,
        "continuous_sequence": X_tst,
    }

# run optimization
def emc_objective(trial):

    # get parameters
    order = trial.suggest_int("order", 2, 2, step=1)
    lambda_f = trial.suggest_float("lambda_f", 0.9, 0.95, step=0.01)
    lambda_s = trial.suggest_float("lambda_s", 0.95, 0.99, step=0.01)
    beta = trial.suggest_float("beta", 0, 0.05, step=0.0001)
    delta_f = trial.suggest_float("delta_f", 0.02, 0.5, step=0.02)
    delta_s = trial.suggest_float("delta_s", 0.02, 0.5, step=0.02)
    eta_f = trial.suggest_float("eta_f", 0.02, 0.5, step=0.02)
    eta_s = trial.suggest_float("eta_s", 0.02, 0.5, step=0.02)
    tau = trial.suggest_int("tau", 10, 100, step=2)
    
    # run
    emc_ins = EMC(
        alpha=alp_car,
        order=order,
        lambda_=[lambda_f, lambda_s],
        beta=beta,
        delta=[delta_f, delta_s],
        eta=[eta_f, eta_s],
        tau=tau
    )
    emc_ins.process_sequence(symbol_sequence_opt, progress=False)

    # evaluate
    labels_true = np.array(y_true_opt)
    labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
    length_diff = len(labels_true) - len(labels_pred)
    if length_diff > 0:
        labels_true = labels_true[:-length_diff]
    emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    return emc_ari

def clustream_objective(trial):

    # get parameters
    n_macro_clusters = trial.suggest_int("n_macro_clusters", 5, 10)
    max_micro_clusters = trial.suggest_int("max_micro_clusters", 50, 150)
    micro_cluster_r_factor = trial.suggest_int("micro_cluster_r_factor", 2, 10)
    time_window = trial.suggest_int("time_window", 500, 1500)
    time_gap = trial.suggest_int("time_gap", 50, 150)
    
    # run
    clustream_ins = CluStream(
        n_macro_clusters=n_macro_clusters,
        max_micro_clusters=max_micro_clusters,
        micro_cluster_r_factor=micro_cluster_r_factor,
        time_window=time_window,
        time_gap=time_gap,
        seed=42,
    )
    labels_pred_clustream = []
    for i, (x, _) in enumerate(iter_array(X_opt)):
        clustream_ins.learn_one(x)
        labels_pred_clustream.append(clustream_ins.predict_one(x))

    # evaluate
    labels_true = np.array(y_true_opt)
    labels_pred = labels_pred_clustream
    length_diff = len(labels_true) - len(labels_pred)
    if length_diff > 0:
        labels_true = labels_true[:-length_diff]
    clustream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
    
    return clustream_ari

def dbstream_objective(trial):

    # get parameters
    clustering_threshold = trial.suggest_float("clustering_threshold", 1, 10)
    fading_factor = trial.suggest_float("fading_factor", 0.001, 0.1)
    cleanup_interval = trial.suggest_int("cleanup_interval", 1, 20)
    intersection_factor = trial.suggest_float("intersection_factor", 0.03, 0.8)
    minimum_weight = trial.suggest_float("minimum_weight", 0.1, 10)
    
    # run
    labels_pred_dbstream = []
    dbstream = DBSTREAM(
        clustering_threshold=clustering_threshold,
        fading_factor=fading_factor,
        cleanup_interval=cleanup_interval,
        intersection_factor=intersection_factor,
        minimum_weight=minimum_weight
    )
    for i, (x, _) in enumerate(iter_array(X_opt)):
        dbstream.learn_one(x)
        labels_pred_dbstream.append(dbstream.predict_one(x))

    # evaluate
    labels_true = np.array(y_true_opt)
    labels_pred = labels_pred_dbstream
    length_diff = len(labels_true) - len(labels_pred_dbstream)
    if length_diff > 0:
        labels_true = labels_true[:-length_diff]
    dbstream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    return dbstream_ari
    
def print_if_best(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    if study.best_trial.number == trial.number:
        fmt_params = {key: (round(value, 4) if isinstance(value, float) else value) for key, value in trial.params.items()}
        print(f"Trial #{trial.number:03d}: {trial.value:.4f}")
        if trial.value >= 0.9:
            print(fmt_params)

def run_study(study_name, storage, objective, n_trials, timeout):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False, callbacks=[print_if_best])

# run optuna
algs = {
    "emc": emc_objective,
    "clustream": clustream_objective,
    "dbstream": dbstream_objective
}
for alg_name, objective in algs.items():

    print(f"optimizing {alg_name} parameters...")

    study_name = f"cwru_{alg_name}"
    create_optuna_db(db_name=study_name)
    storage = f"mysql+pymysql://optuna_user:123@localhost/{study_name}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        load_if_exists=False
    )
    optuna.logging.set_verbosity(optuna.logging.WARN)

    num_workers, trials_per_worker = 20, 25
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=run_study, args=(study_name, storage, objective, trials_per_worker, None))
        processes.append(p)
        p.start()
    print(f"Started {num_workers} workers for {study_name}...")

    for p in processes:
        p.join()

    # after all workers finish, load and print the result
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"{alg_name} - #trials: {len(study.trials)}, best value: {study.best_value:.4f}, best params: {study.best_params}")
    params[alg_name] = study.best_params

# save parameters
params_path = os.path.join(output_dir_path, "params.json")
with open(params_path, "w") as fp:
    json.dump(params, fp)
print(f"params saved: {params_path}")

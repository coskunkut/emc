import json
import multiprocessing
import os

import numpy as np
import optuna
from river.cluster import DBSTREAM, CluStream
from river.stream import iter_array
from sklearn.metrics import cluster

from emc.estimator.EMC import EMCLin as EMC
from emc.estimator.EPSTM import run_epstm
from emc.utils.database import create_optuna_db
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths

# mysqld_safe --bind-address=0.0.0.0 &
# mysql -u root -e "SET GLOBAL max_connections = 500;"

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "eeg", "output")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")

# parameters
params = {}

# run optimization
start_index = 0
opt_index = 2927

optm_ms_seq = data["ms_seq"][start_index:opt_index]
optm_labels = data["Y"][start_index:opt_index]
optm_X = data["X"][start_index:opt_index, :]
cps_optm = np.where(np.diff(optm_labels, prepend=np.nan))[0].tolist()
cps_optm = cps_optm + [len(optm_labels)]

test_ms_seq = data["ms_seq"][opt_index:]
test_labels = data["Y"][opt_index:]
test_X = data["X"][opt_index:, :]
cps_test = np.where(np.diff(test_labels, prepend=np.nan))[0].tolist()
cps_test = cps_test + [len(test_labels)]

def emc_objective(trial):

    # get parameters
    order = trial.suggest_int("order", 1, 1, step=1)
    lambda_f = trial.suggest_float("lambda_f", 0.90, 0.95, step=0.01)
    lambda_s = trial.suggest_float("lambda_s", 0.95, 0.99, step=0.01)
    beta = trial.suggest_float("beta", 0, 0.01, step=0.00001)
    delta_f = trial.suggest_float("delta_f", 0.01, 0.5, step=0.01)
    delta_s = trial.suggest_float("delta_s", 0.01, 0.5, step=0.01)
    eta_f = trial.suggest_float("eta_f", 0.01, 0.5, step=0.01)
    eta_s = trial.suggest_float("eta_s", 0.01, 0.5, step=0.01)
    tau = trial.suggest_int("tau", 10, 100, step=1)
    
    # run
    emc_ins = EMC(
        alpha=data["meta"]["n_clusters"],
        order=order,
        lambda_=[lambda_f, lambda_s],
        beta=beta,
        delta=[delta_f, delta_s],
        eta=[eta_f, eta_s],
        tau=tau
    )
    emc_ins.process_sequence(optm_ms_seq, progress=False)

    # evaluate
    labels_true = optm_labels
    labels_pred = np.array(emc_ins.pred_mode_hist)
    emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    return emc_ari

def clustream_objective(trial):

    # get parameters
    n_macro_clusters = trial.suggest_int("n_macro_clusters", 5, 10)
    max_micro_clusters = trial.suggest_int("max_micro_clusters", 50, 150)
    micro_cluster_r_factor = trial.suggest_int("micro_cluster_r_factor", 2, 10)
    time_window = trial.suggest_int("time_window", 500, 1500)
    time_gap = trial.suggest_int("time_gap", 50, 150)

    # avoid duplicate trials
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value
    
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
    for i, (x, _) in enumerate(iter_array(optm_X)):
        clustream_ins.learn_one(x)
        labels_pred_clustream.append(clustream_ins.predict_one(x))

    # evaluate
    labels_true = optm_labels
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

    # avoid duplicate trials
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value
    
    # run
    labels_pred_dbstream = []
    dbstream = DBSTREAM(
        clustering_threshold=clustering_threshold,
        fading_factor=fading_factor,
        cleanup_interval=cleanup_interval,
        intersection_factor=intersection_factor,
        minimum_weight=minimum_weight
    )
    for i, (x, _) in enumerate(iter_array(optm_X)):
        dbstream.learn_one(x)
        labels_pred_dbstream.append(dbstream.predict_one(x))

    # evaluate
    labels_true = optm_labels
    labels_pred = labels_pred_dbstream
    length_diff = len(labels_true) - len(labels_pred_dbstream)
    if length_diff > 0:
        labels_true = labels_true[:-length_diff]
    dbstream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    return dbstream_ari

def epstm_objective(trial):

    # get parameters
    clustering_threshold = trial.suggest_float("clustering_threshold", 0.01, 1)
    I = trial.suggest_float("I", 0, 1)
    subsequence_minimum_occurrence = trial.suggest_int("subsequence_minimum_occurrence", 2, 10)
    subsequence_length_limit = trial.suggest_int("subsequence_length_limit", 1, 10)

    # avoid duplicate trials
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value

    epstm_tcp_labels_pred = run_epstm(
        pid_list=set(data["ms_seq"]),
        pid_seq=optm_ms_seq,
        change_points=cps_optm,
        subsequence_minimum_occurrence=subsequence_minimum_occurrence,
        subsequence_length_limit=subsequence_length_limit,
        I=I,
        clustering_threshold=clustering_threshold
    )

    # evaluate
    labels_true = optm_labels
    length_diff = len(labels_true) - len(epstm_tcp_labels_pred)
    if length_diff > 0:
        labels_true = labels_true[:-length_diff]
    epstm_tcp_ari = cluster.adjusted_rand_score(labels_pred=epstm_tcp_labels_pred, labels_true=labels_true)

    return epstm_tcp_ari

def print_if_best(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    if study.best_trial.number == trial.number:
        fmt_params = {key: (round(value, 4) if isinstance(value, float) else value) for key, value in trial.params.items()}
        print(f"Trial #{trial.number:03d}: {trial.value:.4f} | params: {fmt_params}")

def run_study(study_name, storage, objective, n_trials, timeout):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False, callbacks=[print_if_best])

algs = {
    "emc": emc_objective,
    "clustream": clustream_objective,
    "dbstream": dbstream_objective,
    "epstm": epstm_objective
}

# run optuna
for alg_name, objective in algs.items():

    print(f"optimizing {alg_name} parameters...")

    study_name = f"eeg_{alg_name}"
    create_optuna_db(db_name=study_name)
    storage = f"mysql+pymysql://optuna_user:123@localhost/{study_name}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        load_if_exists=True
    )
    optuna.logging.set_verbosity(optuna.logging.WARN)

    num_workers, trials_per_worker = 25, 50
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
    json.dump(params, fp, indent=4)
print(f"params saved: {params_path}")

import json
import multiprocessing
import os

import numpy as np
import optuna
from river.cluster import DBSTREAM, CluStream
from river.stream import iter_array
from sklearn.metrics import cluster

from emc.estimator.EMC import EMCLin as EMC
from emc.utils.database import create_optuna_db
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "har", "output")
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")
alp_car = data["meta"]["alp_car"]
prim_len = data["meta"]["prim_len"]
optm_runs = data["optm"]
test_runs = data["test"]

# params
params = {}

def emc_objective(trial):

    # get parameters
    order = trial.suggest_int("order", 1, 1)
    lambda_f = trial.suggest_float("lambda_f", 0.90, 0.95)
    lambda_s = trial.suggest_float("lambda_s", 0.95, 0.99)
    beta = trial.suggest_float("beta", 0, 0.05)
    delta_f = trial.suggest_float("delta_f", 0.02, 0.5)
    delta_s = trial.suggest_float("delta_s", 0.02, 0.5)
    eta_f = trial.suggest_float("eta_f", 0.02, 0.5)
    eta_s = trial.suggest_float("eta_s", 0.02, 0.5)
    tau = trial.suggest_int("tau", 10, 100)

    ari_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

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
        emc_ins.process_sequence(run_params["discrete_sequence"], progress=False)

        # evaluate
        labels_true = np.repeat(run_params["subprocess_sequence"], run_params["regime_lengths"])
        labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
        length_diff = len(labels_true) - len(labels_pred)
        if length_diff > 0:
            labels_true = labels_true[:-length_diff]
        emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
        ari_vec.append(emc_ari)

        # prune
        intermediate_value = np.mean(ari_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(ari_vec)

def clustream_objective(trial):

    # get parameters
    n_macro_clusters = trial.suggest_int("n_macro_clusters", 5, 10)
    max_micro_clusters = trial.suggest_int("max_micro_clusters", 50, 150)
    micro_cluster_r_factor = trial.suggest_int("micro_cluster_r_factor", 2, 10)
    time_window = trial.suggest_int("time_window", 500, 1500)
    time_gap = trial.suggest_int("time_gap", 50, 150)

    ari_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):
        
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
        for i, (x, _) in enumerate(iter_array(run_params["continuous_sequence"])):
            clustream_ins.learn_one(x)
            labels_pred_clustream.append(clustream_ins.predict_one(x))

        # evaluate
        labels_true = np.repeat(run_params["subprocess_sequence"], run_params["regime_lengths"])
        labels_pred = labels_pred_clustream
        length_diff = len(labels_true) - len(labels_pred)
        if length_diff > 0:
            labels_true = labels_true[:-length_diff]
        clustream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
        ari_vec.append(clustream_ari)
        # print(f"{step+1}/{len(optm_runs)}: CluStream ARI: {clustream_ari:.4f}")

        # prune
        intermediate_value = np.mean(ari_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(ari_vec)

def dbstream_objective(trial):

    # get parameters
    clustering_threshold = trial.suggest_float("clustering_threshold", 1, 10)
    fading_factor = trial.suggest_float("fading_factor", 0.001, 0.1)
    cleanup_interval = trial.suggest_int("cleanup_interval", 1, 20)
    intersection_factor = trial.suggest_float("intersection_factor", 0.03, 0.8)
    minimum_weight = trial.suggest_float("minimum_weight", 0.1, 10)

    ari_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):
        
        # run
        labels_pred_dbstream = []
        dbstream = DBSTREAM(
            clustering_threshold=clustering_threshold,
            fading_factor=fading_factor,
            cleanup_interval=cleanup_interval,
            intersection_factor=intersection_factor,
            minimum_weight=minimum_weight
        )
        for i, (x, _) in enumerate(iter_array(run_params["continuous_sequence"])):
            dbstream.learn_one(x)
            labels_pred_dbstream.append(dbstream.predict_one(x))

        # evaluate
        labels_true = np.repeat(run_params["subprocess_sequence"], run_params["regime_lengths"])
        labels_pred = labels_pred_dbstream
        length_diff = len(labels_true) - len(labels_pred_dbstream)
        if length_diff > 0:
            labels_true = labels_true[:-length_diff]
        dbstream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
        ari_vec.append(dbstream_ari)

        # prune
        intermediate_value = np.mean(ari_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(ari_vec)

def print_if_best(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    if study.best_trial.number == trial.number:
        fmt_params = {key: (round(value, 4) if isinstance(value, float) else value) for key, value in trial.params.items()}
        print(f"Trial #{trial.number:04d}: {trial.value:.4f}")
        if trial.value >= 0.65:
            print(f"  params: {fmt_params}")

def run_study(study_name, storage, objective, n_trials, timeout):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False, callbacks=[print_if_best])

algs = {
    "emc": emc_objective,
    "clustream": clustream_objective,
    "dbstream": dbstream_objective,
}
for alg_name, objective in algs.items():
    
    print(f"Optimizing {alg_name} parameters...")

    study_name = f"har_{alg_name}"
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

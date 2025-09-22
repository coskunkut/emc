import copy
import json
import multiprocessing
import os

import numpy as np
import optuna

from emc.estimator.EMC import EMCLin as EMC
from emc.estimator.MC_ADWIN import MC_ADWIN
from emc.estimator.MC_SW import MC_SW
from emc.utils.database import create_optuna_db
from emc.utils.evaluator import evaluate_estimates
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.sparse import sparse_to_dense

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "01_pt", "output")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")
optm_runs = data["optm"]
test_runs = data["test"]

# parameters
params = {}

def emc_objective(trial):

    # get parameters
    lambda_f = trial.suggest_float("lambda_f", 0.90, 0.95, step=0.01)
    lambda_s = trial.suggest_float("lambda_s", 0.95, 0.99, step=0.01)
    beta = trial.suggest_float("beta", 0, 0)
    delta_f = trial.suggest_float("delta_f", 0.05, 0.5, step=0.05)
    delta_s = trial.suggest_float("delta_s", 0.05, 0.5, step=0.05)
    eta_f = trial.suggest_float("eta_f", 0.05, 0.5, step=0.05)
    eta_s = trial.suggest_float("eta_s", 0.05, 0.5, step=0.05)
    tau = trial.suggest_int("tau", 20, 100, step=5)

    mae_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # run
        emc_ins = EMC(
            alpha=data["meta"]["alp_car"],
            order=data["meta"]["k"],
            lambda_=[lambda_f, lambda_s],
            beta=beta,
            delta=[delta_f, delta_s],
            eta=[eta_f, eta_s],
            tau=tau,
        )
        pred_matrices = []
        for s in run_params["symbol_sequence"]:
            emc_ins.process_symbol(s)
            P_pred = sparse_to_dense(copy.deepcopy(emc_ins.P_exp), k=data["meta"]["k"], alpha=data["meta"]["alp_car"])
            pred_matrices.append(P_pred)

        # evaluate
        true_matrices = [run_params["subprocesses"][sid].transition_matrix for sid in run_params["subprocess_sequence"]]
        emc_mae, emc_ae = evaluate_estimates(
            estimates=pred_matrices,
            true_matrices=true_matrices,
            regime_lengths=run_params["regime_lengths"],
            index_symbol_map=emc_ins.index_symbol_map
        )
        mae_vec.append(emc_mae)

        # prune
        intermediate_value = np.mean(mae_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(mae_vec)

def mcadwin_objective(trial):

    # get parameters
    delta = trial.suggest_float("delta", 0.0002, 0.02)
    clock = trial.suggest_int("clock", 3, 320)
    max_buckets = trial.suggest_int("max_buckets", 1, 50)
    min_window_length = trial.suggest_int("min_window_length", 1, 50)
    grace_period = trial.suggest_int("grace_period", 1, 100)

    mae_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # run
        mc_adwin_ins = MC_ADWIN(
            alpha=data["meta"]["alp_car"],
            order=data["meta"]["k"],
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
        )
        mc_adwin_ins.process_sequence(run_params["symbol_sequence"])

        # evaluate
        true_matrices = [run_params["subprocesses"][sid].transition_matrix for sid in run_params["subprocess_sequence"]]
        mc_adwin_mae, mc_adwin_ae = evaluate_estimates(
            estimates=mc_adwin_ins.estimates,
            true_matrices=true_matrices,
            regime_lengths=run_params["regime_lengths"],
            index_symbol_map=mc_adwin_ins.index_symbol_map
        )
        mae_vec.append(mc_adwin_mae)

        # prune
        intermediate_value = np.mean(mae_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(mae_vec)

def mcsw_objective(trial):

    # get parameters
    window_size = trial.suggest_int("window_size", 250, 750)

    mae_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # run
        mc_sw_ins = MC_SW(
            order=data["meta"]["k"],
            alpha=data["meta"]["alp_car"],
            window_size=window_size
        )
        mc_sw_ins.process_sequence(run_params["symbol_sequence"])
        
        # evaluate
        true_matrices = [run_params["subprocesses"][sid].transition_matrix for sid in run_params["subprocess_sequence"]]
        mc_sw_mae, mc_sw_ae = evaluate_estimates(
            estimates=mc_sw_ins.estimates,
            true_matrices=true_matrices,
            regime_lengths=run_params["regime_lengths"],
            index_symbol_map=mc_sw_ins.index_symbol_map
        )
        mae_vec.append(mc_sw_mae)

        # prune
        intermediate_value = np.mean(mae_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(mae_vec)

def print_if_best(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    if study.best_trial.number == trial.number:
        fmt_params = {key: (round(value, 4) if isinstance(value, float) else value) for key, value in trial.params.items()}
        print(f"Trial #{trial.number:04d}: {trial.value:.4f} | {fmt_params}")

def run_study(study_name, storage, objective, n_trials, timeout):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False, callbacks=[print_if_best])

# run optimization for each algorithm
algs = {
    "emc": emc_objective,
    "mc_adwin": mcadwin_objective,
    "mc_sw": mcsw_objective
}
for alg_name, objective in algs.items():

    print(f"Optimizing {alg_name} parameters...")

    study_name = f"syn_pt_{alg_name}"
    create_optuna_db(db_name=study_name)
    storage = f"mysql+pymysql://optuna_user:123@localhost/{study_name}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
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

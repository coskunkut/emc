import json
import multiprocessing
import os

import bocd
import numpy as np
import optuna
from river.drift import ADWIN, KSWIN, PageHinkley

from emc.estimator.EMC import EMCLin as EMC
from emc.utils.database import create_optuna_db
from emc.utils.evaluator import evaluate_cpd
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "02_cpd", "output")

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
    delta_f = trial.suggest_float("delta_f", 0.02, 0.5, step=0.02)
    delta_s = trial.suggest_float("delta_s", 0.02, 0.5, step=0.02)
    eta_f = trial.suggest_float("eta_f", 0.02, 0.5, step=0.02)
    eta_s = trial.suggest_float("eta_s", 0.02, 0.5, step=0.02)
    tau = trial.suggest_int("tau", 10, 100, step=5)

    f1_vec = []
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
        emc_ins.process_sequence(run_params["symbol_sequence"], progress=False)

        emc_cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=np.array(emc_ins.detected_changes),
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(emc_cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

def adwin_objective(trial):

    # get parameters
    delta = trial.suggest_float("delta", 0.0002, 0.02)
    clock = trial.suggest_int("clock", 3, 320)
    max_buckets = trial.suggest_int("max_buckets", 1, 50)
    min_window_length = trial.suggest_int("min_window_length", 1, 50)
    grace_period = trial.suggest_int("grace_period", 1, 100)

    f1_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):
        
        adwin_ins = ADWIN(
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period
        )
        detected_changes = []
        for i, symbol in enumerate(run_params["symbol_sequence"]):
            _ = adwin_ins.update(symbol)
            if adwin_ins.drift_detected:
                detected_changes.append(i)
        adwin_cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=detected_changes,
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(adwin_cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

def kswin_objective(trial):

    # get parameters
    alpha = trial.suggest_float("alpha", 0.0005, 0.05)
    window_size = trial.suggest_int("window_size", 10, 1000)
    stat_size = trial.suggest_int("stat_size", 3, 300)

    f1_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # stat_size must be smaller than window_size
        if not stat_size < window_size:
            raise optuna.TrialPruned()
        
        # Sample larger than population or is negative:
        # self._rng.sample(range(self.window_size - self.stat_size), self.stat_size)
        if not window_size-stat_size >= stat_size:
            raise optuna.TrialPruned()
        
        kswin_ins = KSWIN(
            alpha=alpha,
            window_size=window_size,
            stat_size=stat_size,
            seed=42
        )
        detected_changes = []
        for i, symbol in enumerate(run_params["symbol_sequence"]):
            _ = kswin_ins.update(symbol)
            if kswin_ins.drift_detected:
                detected_changes.append(i)
        kswin_cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=detected_changes,
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(kswin_cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

def pht_objective(trial):

    # get parameters
    min_instances = trial.suggest_int("min_instances", 3, 300)
    delta = trial.suggest_float("delta", 0.0005, 0.05)
    threshold = trial.suggest_float("threshold", 5, 500)
    alpha = trial.suggest_float("alpha", 0.9, 0.99999)
    mode = trial.suggest_categorical("mode", ["up", "down", "both"])

    f1_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        pht_ins = PageHinkley(
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha,
            mode=mode
        )
        detected_changes = []
        for i, symbol in enumerate(run_params["symbol_sequence"]):
            _ = pht_ins.update(symbol)
            if pht_ins.drift_detected:
                detected_changes.append(i)
        pht_cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=detected_changes,
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(pht_cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

def bocd_objective(trial):

    # get parameters
    hazard_lambda = trial.suggest_int("hazard_lambda", 100, 1000, step=100)
    mu = trial.suggest_float("mu", 0.5, 5.0, step=0.5)
    kappa = trial.suggest_float("kappa", 0.1, 10)
    alpha = trial.suggest_float("alpha", 0.1, 10)
    beta = trial.suggest_float("beta", 1.0, 10.0)

    f1_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        bc = bocd.BayesianOnlineChangePointDetection(
            hazard=bocd.ConstantHazard(hazard_lambda), 
            distribution=bocd.StudentT(mu=mu, kappa=kappa, alpha=alpha, beta=beta)
        )
        rt_mle = np.empty(len(run_params["symbol_sequence"]))
        for i, d in enumerate(run_params["symbol_sequence"]):
            bc.update(d)
            rt_mle[i] = bc.rt[0]
        detected_cps = np.where(np.diff(rt_mle) < 0)[0] + 1  # +1 to adjust for the diff operation

        cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=np.array(detected_cps),
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

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
    "adwin": adwin_objective,
    "kswin": kswin_objective,
    "pht": pht_objective,
    "bocd": bocd_objective
}
for alg_name, objective in algs.items():

    print(f"Optimizing {alg_name} parameters...")

    study_name = f"syn_cpd_{alg_name}"
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
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from tqdm.auto import tqdm

from emc.estimator.EMC import EMCLin as EMC
from emc.estimator.MC_ADWIN import MC_ADWIN
from emc.estimator.MC_SW import MC_SW
from emc.utils.evaluator import evaluate_estimates
from emc.utils.loader import load_json_file, load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.plot import get_mpl_conf_path, set_size
from emc.utils.sparse import sparse_to_dense

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "01_pt", "output")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")

# load parameters
params_path = os.path.join(output_dir_path, "params.json")
params = load_json_file(params_path)
if params is not None:
    print(f"{params_path} loaded")

methods_to_run = [
    "emc",
    "mc_adwin",
    "mc_sw"
]

def execute_single_run(run_desc):

    run_id, run_params = run_desc
    result_dict = {"run_id": run_id, "regime_length_bounds": run_id.split("_")[0][3:]}
    true_matrices = [run_params["subprocesses"][sp_id].transition_matrix for sp_id in run_params["subprocess_sequence"]]

    # EMC
    if "emc" in methods_to_run:
        emc_ins = EMC(
            alpha=data["meta"]["alp_car"],
            order=data["meta"]["k"],
            lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
            beta=params["emc"]["beta"],
            delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
            eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
            tau=params["emc"]["tau"],
        )
        emc_ps = []
        for o in run_params["symbol_sequence"]:
            emc_ins.process_symbol(o)
            emc_p_dense = sparse_to_dense(sparse_matrix=emc_ins.P_exp, k=emc_ins.order, alpha=data["meta"]["alp_car"])
            emc_ps.append(emc_p_dense)
        emc_mae, emc_ae = evaluate_estimates(
            estimates=emc_ps,
            true_matrices=true_matrices,
            regime_lengths=run_params["regime_lengths"],
            index_symbol_map=emc_ins.index_symbol_map
        )
        result_dict["emc:mae"] = emc_mae
        result_dict["emc:ae"] = emc_ae

    # run MC-ADWIN
    if "mc_adwin" in methods_to_run:
        mc_adwin_ins = MC_ADWIN(
            alpha=data["meta"]["alp_car"],
            order=data["meta"]["k"],
            delta=params["mc_adwin"]["delta"],
            clock=params["mc_adwin"]["clock"],
            max_buckets=params["mc_adwin"]["max_buckets"],
            min_window_length=params["mc_adwin"]["min_window_length"],
            grace_period=params["mc_adwin"]["grace_period"],
        )
        mc_adwin_ins.process_sequence(run_params["symbol_sequence"])
        mc_adwin_mae, mc_adwin_ae = evaluate_estimates(
            estimates=mc_adwin_ins.estimates,
            true_matrices=true_matrices,
            regime_lengths=run_params["regime_lengths"],
            index_symbol_map=mc_adwin_ins.index_symbol_map
        )
        result_dict["mc_adwin:mae"] = mc_adwin_mae
        result_dict["mc_adwin:ae"] = mc_adwin_ae

    # run MC_SW
    if "mc_sw" in methods_to_run:
        window_sizes = [100, params["mc_sw"]["window_size"], 500]
        for window_size in window_sizes:
            mc_sw_ins = MC_SW(
                order=data["meta"]["k"],
                alpha=data["meta"]["alp_car"],
                window_size=window_size
            )
            mc_sw_ins.process_sequence(run_params["symbol_sequence"])
            mc_sw_mae, mc_sw_ae = evaluate_estimates(
                estimates=mc_sw_ins.estimates,
                true_matrices=true_matrices,
                regime_lengths=run_params["regime_lengths"],
                index_symbol_map=mc_sw_ins.index_symbol_map
            )
            result_dict[f"mc_sw_{window_size}:mae"] = mc_sw_mae
            result_dict[f"mc_sw_{window_size}:ae"] = mc_sw_ae

    return result_dict

# run tasks in parallel
test_runs = data["test"]
tasks = [(k,p) for k,p in test_runs.items()]
with ProcessPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(execute_single_run, task) for task in tasks]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        results.append(result)

# format results table
results_df = pd.DataFrame.from_records(results)
numeric_cols = results_df.select_dtypes(include='number')
results_df_grp = numeric_cols.groupby(results_df["regime_length_bounds"]).agg(['mean', 'std'])
results_df_frm = results_df_grp.apply(
    lambda row: pd.Series(
        {f'{col}': fr"${row[(col, 'mean')]:.3f}\pm{row[(col, 'std')]:.3f}$" for col in numeric_cols.columns},
    ),
    axis=1
).transpose().reset_index().rename(columns={"index":"alg"})
results_df_frm.columns.name = None
results_df_frm = results_df_frm[["alg", *[f"{b[0]}:{b[1]}" for b in data["meta"]["regime_length_bounds_list"]]]]
alg_dict = {
    "emc:mae": "EMC",
    "mc_adwin:mae": "MC-ADWIN",
    "mc_sw_100:mae": "MC-SW ($w=100$)",
    "mc_sw_328:mae": "MC-SW ($w=328$)",
    "mc_sw_500:mae": "MC-SW ($w=500$)",
}
results_df_frm["alg"] = results_df_frm["alg"].replace(alg_dict, regex=True)

# save results
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)
results_path = os.path.join(results_dir_path, "syn_pt_mae.csv")
results_df_frm.to_csv(results_path, index=False, header=True, float_format="%.3f")
print(f"results saved: {results_path}")
print(results_df_frm)

# visualize results
results_df = pd.DataFrame.from_records(results)
run_id = "reg1500:2000_run11"
run = test_runs[run_id]
result = results_df.loc[results_df["run_id"] == run_id]

# initialize figure
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use(get_mpl_conf_path("pub"))
plt.rcParams.update({
    "ytick.minor.visible": True,
    "ytick.direction": "out",
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "axes.prop_cycle": (
        cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color']) +
        cycler(linestyle=["-","--","-.",":","-","--","-.",":","-","--"])
    )
})
size_conf = {"width":"tpami_half", "aspect_ratio":2.25, "fraction": 1}
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(**size_conf))

# plot errors
for alg in result.filter(regex=":ae$").columns:
    ax.plot(np.cumsum(result[alg].values[0]), label=alg.split(":")[0].upper().replace("_", "-"))

xticks = np.insert(run["change_points"], 0, 0)
label_xs = xticks[:-1]+np.diff(xticks)/2
for sp_index, sp_id in enumerate(run["subprocess_sequence"]):
    ax.annotate(
        f"{sp_id+1}",
        xy=(label_xs[sp_index], 0),
        xycoords="data",
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize="x-small",
        fontfamily="monospace",
        color="black",
        bbox=dict(
            boxstyle="Round,pad=0.2",
            facecolor="xkcd:off white",
            edgecolor="black",
            linewidth=0.4
        )
    )
ax.set_xlabel("Time")
ax.set_xticks(xticks)
plt.xticks(rotation=90)
ax.set_ylabel("CAE")
ax.set_ylim(0, None)
ax.legend(loc="upper left", labelspacing=0.1)
ax.margins(x=0.05, y=0.1)

# save the figure
fig_path = os.path.join(results_dir_path, "syn_pt_cae.pdf")
fig.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"figure saved: {fig_path}")

import os
from itertools import pairwise, product

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cycler import cycler
from sklearn.manifold import MDS, TSNE

from emc.utils.compare import hellinger_distance
from emc.utils.paths import get_paths


def get_mpl_conf_path(mpl_conf):
	paths = get_paths()
	return os.path.join(paths["config"]["mpl"], f"{mpl_conf}.mplstyle")

def get_colorscheme(colorscheme):
	if colorscheme == "retro":
		return {"axes.prop_cycle": cycler('color', ['#4165c0','#e770a2','#5ac3be','#696969','#f79a1e','#ba7dcd'])}
	elif colorscheme == "muted":
		return {"axes.prop_cycle": (
			cycler(color=['#332288','#CC6677','#DDCC77','#117733','#88CCEE','#882255','#44AA99','#999933']) +
			cycler(linestyle=["-","--","-.",":","-","--","-.",":"])
		)}
	elif colorscheme == "muted_solid":
		return {"axes.prop_cycle": (
			cycler(color=['#332288','#CC6677','#DDCC77','#117733','#88CCEE','#882255','#44AA99','#999933'])
		)}
	elif colorscheme == "deep":
		return {"axes.prop_cycle": (
			cycler(color=['#4c72b0','#dd8452','#55a868','#c44e52','#8172b3','#937860','#da8bc3','#8c8c8c']) +
			cycler(linestyle=["-","--","-.",":","-","--","-.",":"])
		)}
	elif colorscheme == "pacoty":
		return {"axes.prop_cycle": cycler('color', ['#5A5B9F','#D94F70','#009473','#F0C05A','#7BC4C4','#FF6F61'])}
	elif colorscheme == "tab10":
		return {"axes.prop_cycle": cycler('color', ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])}
	elif colorscheme == "jazzcup":
		return {"axes.prop_cycle": cycler("color", ["#FF6AD5","#C774E8","#AD8CFF","#8795E8","#94D0FF"])}
	elif colorscheme == "science":
		return {"axes.prop_cycle": (
			cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
            cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."])
		)}

def set_size(width, aspect_ratio=1.25, fraction=1):

		if width == "generic_paper":
			width_pt = 472.03123
		elif width == "thesis":
			width_pt = 426.79135
		elif width == "pat_reg":
			width_pt = 468.3324
		elif width == "lncp":
			width_pt = 347.12354
		elif width == "tpami_full":
			width_pt = 516.0
		elif width == "tpami_half":
			width_pt = 252.0
		elif width == "beamer":
			width_pt = 307.28987
		elif width == "beamer_169":
			width_pt = 398.3386
		else:
			width_pt = width

		if aspect_ratio == "golden":
			aspect_ratio = 1.61803398875

		# Width of figure (in pts)
		fig_width_pt = width_pt * fraction

		# Convert from pt to inches
		inches_per_pt = 1 / 72.26999

		# Figure width and height in inches
		fig_width_in = fig_width_pt * inches_per_pt
		fig_height_in = fig_width_in / aspect_ratio

		fig_dim = (fig_width_in, fig_height_in)
		return fig_dim

def plot_cpd(
		p_history,
		condition_symbol_sequences,
		symbol_index_map,
		alphabet_cardinality,
		order=None,
		hit_ratio_history=None
	):

	# ensure that the history hypermatrix is an ndarray
	p_history = np.array(p_history)

	# plt.style.use("seaborn-whitegrid")

	fig, ax = plt.subplots(nrows=1, ncols=1)
	# fig.set_tight_layout(True)
	# plt.rcParams.update(plt.rcParamsDefault)

	for condition_symbol_sequence in condition_symbol_sequences:

		# print(condition_symbol_sequence)

		# convert symbols to indices in given event
		condition_index_sequence = [symbol_index_map[symbol] for symbol in condition_symbol_sequence]

		# print(condition_index_sequence)

		for event_index in range(alphabet_cardinality):

			# print(event_index)

			event_symbol = list(symbol_index_map.keys())[list(symbol_index_map.values()).index(event_index)]
			# print(event_symbol)

			event_index_sequence = condition_index_sequence + [event_index]
			# print(event_index_sequence)

			# prepare index tuple for advanced slicing
			# this way, higher dimensions are also supported
			p_history_index = (slice(None),) + tuple(event_index_sequence)
			cpd_history = p_history[p_history_index]

			if order:
				label = condition_symbol_sequence + ["X"]*(order-1) + [event_symbol]
			else:
				label = condition_symbol_sequence + [event_symbol]
			ax.plot(
				cpd_history,
				label=label,
				# color="black",
				alpha=0.8,
				# linewidth=1,
				# zorder=10
			)

		if hit_ratio_history is not None:
			ax.plot(
				hit_ratio_history,
				label="Hit Ratio",
				# color="black",
				alpha=0.8,
				# linewidth=1,
				# zorder=10
			)

	ax.set_xlabel("Observation Count")
	ax.set_ylabel("Probability")
	ax.set_ylim(-0.05, 1.05)
	ax.set_yticks(np.linspace(0, 1, num=11))
	ax.margins(x=0.05, y=0.05)
	ax.grid(linestyle="dashed", alpha=0.5)

	# sort and show legend entries
	handles, labels = ax.get_legend_handles_labels()
	labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
	ax.legend(handles, labels, loc="best")

	# # fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.05)
	plt.show()

def draw_markov_chain(transition_matrix, fig_path, node_color=None):

	index_symbol_map = {i: str(i) for i in range(len(transition_matrix))}
	mc = nx.DiGraph()

	transitions = list(product(index_symbol_map.keys(), repeat=2))
	for transition in transitions:
		# index_tuple = tuple([symbol_index_map[s] for s in transition])
		index_tuple = tuple([int(s) for s in transition])
		# print(transition, index_tuple)
		mc.add_edge(
			transition[0],
			transition[1],
			# weight=transition_matrix[index_tuple],
			label="{:.2f}".format(transition_matrix[index_tuple])
		)

	pdt = nx.nx_pydot.to_pydot(mc)
	# pdt.set_bgcolor("lightyellow")
	for node in pdt.get_node_list():
		node.set_shape("circle")
		node.set("style", "filled")
		node.set("fillcolor", node_color if node_color else "lightblue")
	for edge in pdt.get_edge_list():
		edge.set("decorate", "true")
	# pdt.set_prog("sfdp")
	pdt.set("layout", "dot")
	# pdt.set("mindist", 2)
	# pdt.set("splines", "curved")
	# pdt.set("overlap", "vpsc")
	# pdt.set("overlap_scaling", "12")
	# pdt.set("nodesep", 18)
	# pdt.set("voro_margin", 0.2)
	# pdt.set("minlen", 3)
	pdt.set("beautify", "true")
	# pdt.set("rankdir", "TB")
	# print(pdt)
	pdt.write(fig_path, format="png")

	# pos = nx.circular_layout(mc)
	# nx.draw_networkx_nodes(mc, pos, node_size=1000)
	# nx.draw_networkx_labels(mc, pos, symbol_index_map, font_size=16)
	# edge_labels = {(str(i), str(j)): f"{transition_matrix[i][j]:.2f}" for i in range(len(transition_matrix)) for j in range(len(transition_matrix[0]))}
	# print(edge_labels)
	# nx.draw_networkx_edges(mc, pos, edge_color='black', arrows=True, connectionstyle="arc3,rad=0.2")
	# nx.draw_networkx_edge_labels(mc, pos, edge_labels=edge_labels, font_size=14)
	# plt.axis('off')
	# plt.show()

def plot_mode_transition(
	subplots,
	labels_true,
	data,
	drift_scores,
	labels_pred,
	change_points,
	cp_scale_coeff=1,
	data_label="Data",
	data_yticks=None,
	data_include_modes=False,
	legend_text=None,
	legend_loc="upper left",
	mode_id_to_label_map=None,
	xlabel="Time",
	xticklabel_coeff=1,
	xtick_rotation=0,
	xtick_ids_to_pad=[],
	xtick_ids_to_skip=[],
	mpl_conf_path=None,
	mpl_conf_override=None,
	fig_size=None,
	fig_path=None,
):

	# load mpl conf and apply overrides
	if mpl_conf_path is not None:
		plt.style.use("default")
		plt.style.use(mpl_conf_path)
	if mpl_conf_override is not None:
		plt.rcParams.update(mpl_conf_override)
	
	# initialize figure
	fig, axs = plt.subplots(nrows=len(subplots), ncols=1, figsize=fig_size)
	subplot_index=0

	# plot the true mode transition
	if "true_modes" in subplots:
		axs[subplot_index].plot(labels_true, color="black")
		axs[subplot_index].set_xticks(change_points)
		axs[subplot_index].set_xticklabels([])
		axs[subplot_index].set_ylabel("True Modes")
		if mode_id_to_label_map is not None:
			axs[subplot_index].set_ylim(0.5, len(mode_id_to_label_map)+0.5)
			axs[subplot_index].set_yticks(list(mode_id_to_label_map.keys()))
			axs[subplot_index].set_yticklabels(list(mode_id_to_label_map.values()))
		subplot_index += 1

	# plot data
	if "data" in subplots:
		axs[subplot_index].plot(data, color="black")
		axs[subplot_index].set_xticks(change_points)
		axs[subplot_index].set_xticklabels([])
		axs[subplot_index].set_ylabel(data_label)
		if data_yticks: axs[subplot_index].set_yticks(data_yticks)
		for i, (reg_str, reg_end) in enumerate(pairwise(change_points)):
			mid_point = int(((reg_str+reg_end)/2))
			mid_point_text = int(mid_point)
			_, ymax = axs[subplot_index].get_ylim()
			axs[subplot_index].annotate(
				# text=fr"\texttt{{{mode_id_to_label_map[labels_true[mid_point]]}}}",
				text=f"{mode_id_to_label_map[labels_true[mid_point]]}",
				xy=(mid_point_text, ymax),
				xycoords="data",
				xytext=(0, 5),
				textcoords="offset points",
				ha="center",
				va="center",
				rotation=0,
				fontsize="6",
				fontfamily="monospace",
				color="black",
				bbox=dict(
					boxstyle="Round,pad=0.2",
					facecolor="xkcd:off white",
					edgecolor="black",
					linewidth=0.4,
					alpha=1
				),
			)
		subplot_index += 1

	# plot drift
	if "drift_scores" in subplots:
		axs[subplot_index].plot(drift_scores, color="black")
		axs[subplot_index].set_ylabel("Drift")
		# axs[subplot_index].set_ylim(-0.1, np.max(drift_scores)+0.1)
		axs[subplot_index].set_xticks([int(cp/cp_scale_coeff) for cp in change_points])
		axs[subplot_index].set_xticklabels([])
		subplot_index += 1

	# plot discovered modes
	if "discovered_modes" in subplots:
		axs[subplot_index].plot(labels_pred, color="black", label=legend_text, zorder=5)
		axs[subplot_index].set_xlabel(xlabel)
		axs[subplot_index].set_xticks(change_points)
		if xtick_rotation in [0, 90]:
			axs[subplot_index].set_xticklabels([cp*xticklabel_coeff for cp in change_points], rotation=xtick_rotation)
		else:
			axs[subplot_index].set_xticklabels([cp*xticklabel_coeff for cp in change_points], rotation=xtick_rotation, rotation_mode="anchor", ha="right")
		axs[subplot_index].set_ylabel("Discovered\nModes")
		axs[subplot_index].set_yticks(list(set(labels_pred)))
		if legend_text is not None: axs[subplot_index].legend(loc=legend_loc)

		# plot drift heatmap across discovered modes
		uncertainty = np.repeat(drift_scores, cp_scale_coeff)
		# uncertainty[:200] *= 0.5 # for eeg
		x = list(range(len(uncertainty)))
		y_margin = np.max(labels_pred)*0.15
		extent = [0, len(uncertainty), 1-y_margin, np.max(labels_pred)+y_margin]
		norm = colors.PowerNorm(gamma=1.5, clip=True)
		# norm = colors.PowerNorm(gamma=0.9, clip=True) # for eeg
		im = axs[subplot_index].imshow(
			uncertainty[np.newaxis,:],
			cmap="Reds",
			aspect="auto",
			extent=extent,
			# norm="linear",
			norm=norm,
			# vmin=0.05,
			# vmax=0.25,
			interpolation="bilinear",
			# interpolation_stage="rgba",
			alpha=0.8,
			# zorder=4
		)
		# Add colorbar
		cbar = fig.colorbar(im, ax=axs[subplot_index], pad=0.01)
		cbar.set_ticks(np.arange(0, np.max(uncertainty), 0.1))
		# cbar.set_ticks(np.arange(0, 0.25, 0.05))
		cbar.ax.tick_params(labelsize=3, pad=1)
		axs[subplot_index].grid()

	# fix overlapping xticks caused by short jump actions
	for tick_ind, tick in enumerate(axs[subplot_index].xaxis.get_major_ticks()):
		if tick_ind-1 in xtick_ids_to_pad:
			tick.set_pad(8)
		if tick_ind-1 in xtick_ids_to_skip:
			axs[subplot_index].xaxis.get_majorticklabels()[tick_ind-1].set_visible(False)
	subplot_index += 1

	# settings for all axes
	for i, ax in enumerate(axs):
		ax.use_sticky_edges = False
		ax.margins(x=0.05, y=0) if i == len(axs)-1 else ax.margins(x=0.05, y=0.2)
		ax.autoscale_view()

	# save the figure
	if fig_path is not None:
		fig.savefig(fig_path, bbox_inches="tight")
	else:
		plt.show()
	plt.close()

def plot_mode_distances(
	learned_modes,
	mpl_conf_path=None,
	mpl_conf_override=None,
	fig_size=None,
	fig_path=None,
):

	dist_matrix = np.zeros((len(learned_modes), len(learned_modes)))
	for i in learned_modes.keys():
		for j in learned_modes.keys():
			if isinstance(learned_modes[i], list):
				hd = np.max(hellinger_distance(np.array(learned_modes[i]), np.array(learned_modes[j])))
			else:
				hd = np.max(hellinger_distance(np.array(learned_modes[i].mean), np.array(learned_modes[j].mean)))
			dist_matrix[int(i)-1, int(j)-1] = hd

	mds = MDS(
		n_components=2,
		# max_iter=3000,
		# eps=1e-9,
		random_state=42,
		dissimilarity="precomputed",
		n_jobs=1,
	)
	pos = mds.fit(dist_matrix).embedding_

	# initialize figure
	if mpl_conf_path is not None:
		plt.style.use(mpl_conf_path)
	if mpl_conf_override is not None:
		plt.rcParams.update(mpl_conf_override)
	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
	plt.gca().set_aspect("equal")

	plt.scatter(
		x=pos[:, 0],
		y=pos[:, 1],
		c="xkcd:dark",
		s=30,
		zorder=5
	)
	for i in learned_modes.keys():
		axs.annotate(
			i,
			xy=(pos[int(i)-1,0], pos[int(i)-1,1]),
			xycoords="data",
			xytext=(0,0.25),
			textcoords="offset points",
			fontsize="xx-small",
			color="white",
			ha="center",
			va="center_baseline",
			zorder=10
		)
	ticks = np.linspace(-1, 1, 11)
	axs.set_xticks(ticks, ticks, rotation=45, rotation_mode="anchor", ha="right")
	axs.xaxis.set_major_formatter("{x:.1f}")
	axs.set_yticks(ticks)
	axs.margins(x=0.15, y=0.15)

	# save the figure
	if fig_path is not None:
		fig.savefig(fig_path, bbox_inches="tight")
	else:
		plt.show()
	plt.close()

def plot_transition_matrices(
	modes,
	compact_mode=False,
	mpl_conf_path=None,
	mpl_conf_override=None,
	fig_size=None,
	fig_dir_path=None
):
	
	textcolors = ("black", "white")
	kw = dict(horizontalalignment = "center", verticalalignment = "center")
	
	# initialize figure
	if mpl_conf_path is not None:
		plt.style.use(mpl_conf_path)
	if mpl_conf_override is not None:
		plt.rcParams.update(mpl_conf_override)
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
	
	for mode_id, mode_tm in modes.items():
		if isinstance(mode_tm, list):
			mode_tm = np.array(mode_tm)
		else:
			mode_tm = mode_tm.mean
		im = ax.imshow(mode_tm, cmap="Blues")

		# colorbar
		# cbar = ax.figure.colorbar(im, ax=ax)

		threshold = im.norm(mode_tm.max())/2.
		for i in range(mode_tm.shape[0]):
			for j in range(mode_tm.shape[1]):
				kw.update(color=textcolors[int(im.norm(mode_tm[i, j]) > threshold)])
				text = ax.text(j, i, f"{mode_tm[i, j]:.2f}", ha="center", va="center", **kw)

		ax.set_xticks(list(range(mode_tm.shape[1])))
		ax.set_yticks(list(range(mode_tm.shape[0])))
		ax.tick_params(axis="y",direction="out")
		ax.tick_params(axis="x",direction="out")
		ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
		
		ax.spines[:].set_visible(False)
		ax.set_xticks(np.arange(mode_tm.shape[1]+1)-.5, minor=True)
		ax.set_yticks(np.arange(mode_tm.shape[0]+1)-.5, minor=True)
		ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
		ax.tick_params(which="minor", bottom=False, left=False)

		ax.grid(visible=False, which="major")

		# compact mode
		if compact_mode:
			plt.axis('off')
		
		# save or show
		if fig_dir_path is not None:
			fig_path = os.path.join(fig_dir_path, f"mode_{mode_id}.pdf")
			fig.savefig(fig_path, bbox_inches="tight")
		else:
			plt.show()

		plt.cla()

	plt.close()
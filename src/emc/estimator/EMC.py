import copy
import logging
from collections import deque
from itertools import product

import numpy as np
from tqdm import tqdm

from emc.utils.compare import hellinger_distance
from emc.utils.stats import ExpectedTensor


class EMC:

	def __init__(
		self,
		alpha, # int, alphabet cardinality, it is needed to initialize the estimate tensor
		order, # int, Markov order of the chain
		lambda_, # list or double [0,1], AKA: learning coefficient
		beta, # double [0,1], entropy regularization rate
		delta, # list or double [0,1], min distance to fire drift detection
		eta, # list or double [0,1], model similarity threshold
		tau, # int, distance between compared estimates in terms of symbols
		live_memory=True, # boolean, whether to update models in memory
		symbols_are_indices=True, # boolean, whether to use symbols as indices in estimate tensor
		log_file_path=None # path, file to write logs
	):

		# initialize EMC instance
		self.alpha = alpha
		self.order = order
		self.lambda_fast, self.lambda_slow = lambda_[:2] if isinstance(lambda_, list) else (lambda_, lambda_)
		self.beta = beta
		self.delta_fast, self.delta_slow = delta[:2] if isinstance(delta, list) else (delta, delta)
		self.eta_fast, self.eta_slow = eta[:2] if isinstance(eta, list) else (eta, eta)
		self.tau = tau
		self.live_memory = live_memory
		self.symbols_are_indices = symbols_are_indices
		self.is_stationary = False # start from non-stationary state
		self.dist = hellinger_distance
		self.obs_count = 0

		# estimate and prediction histories
		self.P0 = 1/self.alpha * np.ones(shape=np.repeat(self.alpha, self.order+1))
		self.P_hist = [self.P0]
		self.P_exp_hist = []
		self.curr_mode_pred = None # id of the most recently predicted mode
		self.pred_mode_hist = [] # predicted mode history

		# symbol to index and index to symbol maps
		self.symbol_index_map = {}
		self.index_symbol_map = {}

		# condition vectors (symbol and index)
		self.cnd_vec = deque(maxlen=self.order)
		self.cndi_vec = deque(maxlen=self.order)

		# detections
		self.detected_drifts = []
		self.detected_changes = []
		self.deviation_history = []
		self.stationarity_history = []

		# the memory
		self.learned_modes = {}

		# initialize logger
		if log_file_path:
			logging.basicConfig(
				filename=log_file_path,
				filemode="w",
				format="%(levelname)s: %(message)s",
				encoding="utf-8",
				level=logging.INFO,
				force=True
			)

	# EMC update (Eq. 2)
	# - updates the probabilities of the observed and non-observed relevant events
	# - probabilities of the irrelevant events are not updated
	def update_estimate(self, symbol_index):
		lambda_ = self.lambda_slow if self.is_stationary else self.lambda_fast
		new_estimate = np.copy(self.P_hist[-1])
		new_estimate[tuple(self.cndi_vec)] *= lambda_
		new_estimate[tuple(self.cndi_vec)][symbol_index] += (1 - lambda_)
		self.P_hist.append(new_estimate)

	# entropy regularization (Eq. 11)
	# - gradually makes the probability tensor closer to uniform distribution
	# - only affects the irrelevant region
	def entropy_regularization(self):
		uniform_tensor = np.copy(self.P0)
		exclude_idx = tuple(self.cndi_vec)
		uniform_tensor[exclude_idx] = self.P_hist[-1][exclude_idx]
		self.P_hist[-1] = (1 - self.beta) * self.P_hist[-1] + self.beta * uniform_tensor
	
	# compares the most recent estimate with a past version, changes steady/drift (stationary/non-stationary) state
	def check_deviation(self):

		# calculate the deviation
		current_deviation = np.max(self.dist(self.P_hist[-1], self.P_hist[-self.tau]))
		self.deviation_history.append(current_deviation)

		if self.is_stationary:

			# S -> NS
			if current_deviation > self.delta_slow: 
				logging.info(f"{self.obs_count}: stationary -> non-stationary (H:{current_deviation:.4f}, Ds:{self.delta_slow})")
				self.detected_drifts.append(self.obs_count)
				self.is_stationary = False
				self.identify_regime()

			# S -> S
			else:
				logging.info(f"{self.obs_count}: stationary (H:{current_deviation:.4f}, Ds:{self.delta_slow})")

		else:

			# NS -> NS
			if current_deviation > self.delta_fast:
				logging.info(f"{self.obs_count}: non-stationary (H:{current_deviation:.4f}, Df:{self.delta_fast})")
				self.identify_regime()

			# NS -> S
			else:
				logging.info(f"{self.obs_count}: non-stationary -> stationary (H:{current_deviation:.4f}), checking memory")
				self.is_stationary = True
				self.identify_regime()

	# compares the current estimate to saved models in the memory
	# updates the memory if needed
	def identify_regime(self):

		# get a copy of the most recent estimate
		current_matrix = np.copy(self.P_hist[-1])

		# memory is empty
		if len(self.learned_modes) == 0:

			# save the mode
			if self.is_stationary:
				self.insert_to_memory(current_matrix)

		# memory is NOT empty
		else:

			mode_dists = self.get_dists_from_memory(self.P_hist[-1])
			closest_model_id, min_distance = list(mode_dists.items())[0]
			logging.info(f"dists: {self.get_dists_from_memory(self.P_hist[-1])}")

			# closest model is close enough
			eta = self.eta_slow if self.is_stationary else self.eta_fast
			if min_distance < eta:
				self.curr_mode_pred = closest_model_id
				logging.info(f"previously seen mode {closest_model_id} is close enough (t:{eta})")

			# closest model is NOT close enough
			else:
				if self.is_stationary:
					logging.info("no mode is similar enough (t: {})".format(eta))
					self.insert_to_memory(current_matrix)

	# inserts a given tensor to the memory, allocates the next available index
	def insert_to_memory(self, tensor_to_insert):
		new_mode_id = len(self.learned_modes)+1
		self.learned_modes[new_mode_id] = ExpectedTensor(initial_tensor=tensor_to_insert)
		self.curr_mode_pred = new_mode_id
		logging.info("{}: new mode {} is saved".format(self.obs_count, new_mode_id))

	# calculates the distance of a given tensor to the elements of the memory
	# returns a sorted dict of distances
	def get_dists_from_memory(self, ref_tensor):
		mode_dists = {}
		for mode_id, mode_transition_matrix in self.learned_modes.items():
			mode_dists[mode_id] = np.max(self.dist(mode_transition_matrix.mean, ref_tensor))
		return dict(sorted(mode_dists.items(), key=lambda item: item[1]))

	# manages symbol->index and index->symbol mappings
	# - assigns a new index to given symbol if it has not been observed before
	# - returns index
	def get_index(self, symbol):
		if self.symbols_are_indices:
			if symbol not in self.symbol_index_map:
				self.symbol_index_map[symbol] = symbol
				self.index_symbol_map[symbol] = symbol
			index = symbol
		else:
			if len(self.symbol_index_map) == 0:
				self.symbol_index_map[symbol] = 0
				self.index_symbol_map[0] = symbol
			elif symbol not in self.symbol_index_map:
				if len(self.symbol_index_map) >= self.alpha:
					print(f"Unexpected symbol observed: {symbol}")
					exit()
				next_available_index = max(self.symbol_index_map.values()) + 1
				self.symbol_index_map[symbol] = next_available_index
				self.index_symbol_map[next_available_index] = symbol
			index = self.symbol_index_map[symbol]
		return index

	# processes a single symbol
	def process_symbol(self, symbol):

		# learn
		index = self.get_index(symbol)
		if len(self.cndi_vec) == self.order:
			self.update_estimate(index)
			if self.beta > 0:
				self.entropy_regularization()
		else:
			self.P_hist.append(self.P0)

		# maintain memory
		if len(self.learned_modes) == 0:
			self.pred_mode_hist.append(1)
			P_exp = np.copy(self.P_hist[-1])
		else:
			if self.curr_mode_pred != self.pred_mode_hist[-1]:
				self.detected_changes.append(self.obs_count)
			self.pred_mode_hist.append(self.curr_mode_pred)
			if self.is_stationary:
				if self.live_memory:
					self.learned_modes[self.curr_mode_pred].update(self.P_hist[-1])
				P_exp = np.copy(self.learned_modes[self.curr_mode_pred].mean)
			else:
				P_exp = np.copy(self.P_hist[-1])
		self.P_exp_hist.append(P_exp)

		self.stationarity_history.append([0,1][self.is_stationary])
		self.obs_count += 1
		self.cnd_vec.append(symbol)
		self.cndi_vec.append(index)
		if self.obs_count % self.tau == 0:
			self.check_deviation()

	# processes each symbol in a given sequence, shows progress information
	def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
		if progress:
			pbar_conf = {
				"total": len(sequence),
				"desc": progress_desc.ljust(progress_desc_width),
				"bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
				"ascii": True
			}
			with tqdm(**pbar_conf) as pbar:
				for symbol in sequence:
					self.process_symbol(symbol)
					pbar.update()
		else:
			for symbol in sequence:
				self.process_symbol(symbol)

class EMCLin:

    def __init__(
        self,
        alpha, # int, alphabet cardinality, it is needed to initialize the estimate tensor
        order, # int, Markov order of the chain
        lambda_, # list or double [0,1], AKA: learning coefficient
        beta, # double [0,1], entropy regularization rate
        delta, # list or double [0,1], min distance to fire drift detection
        eta, # list or double [0,1], model similarity threshold
        tau, # int, distance between compared estimates in terms of symbols
        live_memory=True, # boolean, whether to update models in memory
        symbols_are_indices=True, # boolean, whether to use symbols as indices in estimate tensor
        log_file_path=None # path, file to write logs
    ):

        # initialize EMC instance
        self.alpha = alpha
        self.order = order
        self.lambda_fast, self.lambda_slow = lambda_[:2] if isinstance(lambda_, list) else (lambda_, lambda_)
        self.beta = beta
        self.delta_fast, self.delta_slow = delta[:2] if isinstance(delta, list) else (delta, delta)
        self.eta_fast, self.eta_slow = eta[:2] if isinstance(eta, list) else (eta, eta)
        self.tau = tau

        # settings
        self.live_memory = live_memory
        self.symbols_are_indices = symbols_are_indices
        self.is_stationary = False # start from non-stationary state
        self.dist = hellinger_distance
        self.obs_count = 0

        # estimate and prediction histories
        self.P = {}
        self.P_ref = {}
        self.P_exp = self.P
        self.updated_context_keys = []
        self.distances_to_modes = {}

        self.uniform_cpd = np.full(shape=self.alpha, fill_value=1/self.alpha, dtype=np.float64)
        self.all_context_keys = product(*[range(self.alpha)] * self.order)

        self.curr_mode_pred = None # id of the most recently predicted mode
        self.pred_mode_hist = [] # predicted mode history

        # symbol to index and index to symbol maps
        self.symbol_index_map = {}
        self.index_symbol_map = {}

        # condition vectors (symbol and index)
        self.cnd_vec = deque(maxlen=self.order)
        self.cndi_vec = deque(maxlen=self.order)

        # detections
        self.detected_drifts = []
        self.detected_changes = []
        self.drift_scores = []
        self.stationarity_history = []

        # the memory
        self.learned_modes = {}
        self.n_memory_updates = {}

    # manages symbol->index and index->symbol mappings
    # - assigns a new index to given symbol if it has not been observed before
    # - returns index
    def get_index(self, symbol):
        if self.symbols_are_indices:
            if symbol not in self.symbol_index_map:
                self.symbol_index_map[symbol] = symbol
                self.index_symbol_map[symbol] = symbol
            index = symbol
        else:
            if len(self.symbol_index_map) == 0:
                self.symbol_index_map[symbol] = 0
                self.index_symbol_map[0] = symbol
            elif symbol not in self.symbol_index_map:
                if len(self.symbol_index_map) >= self.alpha:
                    print(f"Unexpected symbol observed: {symbol}")
                    exit()
                next_available_index = max(self.symbol_index_map.values()) + 1
                self.symbol_index_map[symbol] = next_available_index
                self.index_symbol_map[next_available_index] = symbol
            index = self.symbol_index_map[symbol]
        return index

    def update_reference(self, update_type="partial"):

        # full copy
        if update_type == "full":
            self.P_ref = {key: np.copy(value) for key, value in self.P.items()}

        # partial copy
        elif update_type == "partial":
            for key in self.updated_context_keys:
                self.P_ref[key] = np.copy(self.P[key])

        # deep copy
        else:
            self.P_ref = copy.deepcopy(self.P)

        # reset
        self.updated_context_keys = []

    def update_estimate(self, symbol_index):

        # get lambda depending on stationarity
        lambda_ = self.lambda_slow if self.is_stationary else self.lambda_fast

        # get the key for the current condition vector
        key = tuple(self.cndi_vec)

        # update P
        if key not in self.P:
            self.P[key] = np.full(shape=self.alpha, fill_value=1/self.alpha, dtype=np.float64)
        self.P[key] *= lambda_
        self.P[key][symbol_index] += (1 - lambda_)

        # keep track of indices of updated CPDs
        if key not in self.updated_context_keys:
            self.updated_context_keys.append(key)

    def entropy_regularization(self, type="partial"):

        # update all contexts that exist in P
        if type == "full":
            for k, v in self.P.items():
                if k == tuple(self.cndi_vec):
                    continue
                self.P[k] = v * (1 - self.beta) + self.uniform_cpd * self.beta

        # update all contexts that have not been updated in current batch and exist in P
        elif type == "partial":
            for context_key in self.all_context_keys:
                if context_key in self.updated_context_keys or context_key not in self.P:
                    continue
                self.P[context_key] = self.P[context_key] * (1 - self.beta) + self.uniform_cpd * self.beta

    # compute the distance between P and P_ref
    def compute_drift_score(self):

        drift_score = 0
        for context_key in self.updated_context_keys:
            if context_key in self.P_ref:
                cpd_dist = self.dist(self.P[context_key], self.P_ref[context_key])
            else:
                cpd_dist = self.dist(self.P[context_key], self.uniform_cpd)
            if cpd_dist > drift_score:
                drift_score = cpd_dist
        self.drift_scores.append(drift_score)

    # compute the distance between P_ref and learned modes
    def compute_mode_distances(self):

        if len(self.learned_modes) > 0:

            for mode_id, mode_repr in self.learned_modes.items():
                max_context_dist = 0
                max_context_key = None
                for context_key in self.updated_context_keys:
                    if context_key in mode_repr:
                        if context_key not in self.P:
                            print(f"{self.obs_count} Context key {context_key} not found in P")
                        context_dist = self.dist(self.P[context_key], mode_repr[context_key].mean)
                    else:
                        context_dist = self.dist(self.P[context_key], self.uniform_cpd)
                    if context_dist > max_context_dist:
                        max_context_dist = context_dist
                        max_context_key = context_key
                self.distances_to_modes[mode_id] = (max_context_key, max_context_dist)

        # print(f"Mode distances ({len(self.learned_modes)}) at {self.obs_count}: {self.distances_to_modes}")

    def check_drift(self):

        if self.is_stationary:

            # S -> NS
            if self.drift_scores[-1] > self.delta_slow:
                self.detected_drifts.append(self.obs_count)
                self.is_stationary = False
                self.predict_regime()
                # print(f"Drift detected at {self.obs_count} with score {self.cpd_dist_max:.4f}")
            
            # S -> S
            else:
                pass

        else:

            # NS -> NS
            if self.drift_scores[-1] > self.delta_fast:
                self.predict_regime()
                # pass

            # NS -> S
            else:
                self.is_stationary = True
                self.predict_regime()
                # print(f"Stationarity detected at {self.obs_count} with score {self.cpd_dist_max:.4f}")

    def save_mode(self):
        new_mode_id = len(self.learned_modes)+1
        self.learned_modes[new_mode_id] = {key: ExpectedTensor(copy.deepcopy(value)) for key, value in self.P.items()}
        self.curr_mode_pred = new_mode_id
        # print(f"Mode {new_mode_id} saved at {self.obs_count}")

    # live memory update
    def update_mode(self):
         
        context_key = tuple(self.cndi_vec)
        if context_key not in self.learned_modes[self.curr_mode_pred]:
            uniform_cpd = np.full(shape=self.alpha, fill_value=1/self.alpha, dtype=np.float64)
            self.learned_modes[self.curr_mode_pred][context_key] = ExpectedTensor(initial_tensor=uniform_cpd)
        if context_key in self.P:
            val = self.P[context_key]
            self.learned_modes[self.curr_mode_pred][context_key].update(val)

        if self.curr_mode_pred in self.distances_to_modes:
            known_max_context_dist = self.distances_to_modes[self.curr_mode_pred][1]
            if context_key in self.learned_modes[self.curr_mode_pred]:
                context_dist = self.dist(self.P[context_key], self.learned_modes[self.curr_mode_pred][context_key].mean)
            else:
                context_dist = self.dist(self.P[context_key], self.uniform_cpd)
            if context_dist > known_max_context_dist:
                self.distances_to_modes[self.curr_mode_pred] = (context_key, context_dist)

    def check_memory(self):

        mode_dists = {}
        for mode_id, mode_repr in self.learned_modes.items():

            # prepare a list of context keys to check
            context_keys_to_check = self.updated_context_keys.copy()
            if mode_id in self.distances_to_modes:
                max_context_key = self.distances_to_modes[mode_id][0]
                if max_context_key not in context_keys_to_check and max_context_key is not None:
                    context_keys_to_check.append(max_context_key)

            # compute the maximum context distance
            max_context_dist = 0
            for context_key in context_keys_to_check:
                if context_key in mode_repr:
                    context_dist = self.dist(self.P[context_key], mode_repr[context_key].mean)
                else:
                    context_dist = self.dist(self.P[context_key], self.uniform_cpd)
                if context_dist > max_context_dist:
                    max_context_dist = context_dist
            mode_dists[mode_id] = max_context_dist

        return dict(sorted(mode_dists.items(), key=lambda item: item[1]))

    def predict_regime(self):
        
        # mode memory is empty
        if len(self.learned_modes) == 0:
            
            # save the mode
            if self.is_stationary:
                self.save_mode()
                # print(f"EMCF: Mode {self.curr_mode_pred} saved at {self.obs_count}")

        # mode memory is not empty
        else:

            closest_model_id = min(self.distances_to_modes, key=lambda k: self.distances_to_modes[k][1])
            min_distance = self.distances_to_modes[closest_model_id][1]

            # closest model is sufficiently close
            eta = self.eta_slow if self.is_stationary else self.eta_fast
            if min_distance < eta:
                self.curr_mode_pred = closest_model_id
                # print(f"Mode {closest_model_id} predicted at {self.obs_count} with distance {min_distance:.4f}")

            # closest model is not sufficiently close
            else:

                # print(f"No close mode found at {self.obs_count} with distance {min_distance:.4f}")

                # save the mode
                if self.is_stationary:
                    self.save_mode()
                    # print(f"EMCF: Mode {self.curr_mode_pred} saved at {self.obs_count}, d:{min_distance:.4f}, cm:{closest_model_id}")

    # processes a single symbol
    def process_symbol(self, symbol):

        index = self.get_index(symbol)

        if len(self.cndi_vec) == self.order:

            # learn
            self.update_estimate(index)

            # entropy regularization
            if self.beta > 0:
                self.entropy_regularization(type="full")

            if self.obs_count % self.tau == 0:
                self.compute_drift_score()
                self.compute_mode_distances()
                self.check_drift()
                self.update_reference()

            if self.is_stationary and self.live_memory:
                self.update_mode()

        # maintain memory
        if len(self.learned_modes) == 0:
            self.pred_mode_hist.append(1)
            self.P_exp = self.P.copy()

        else:

            # a change is detected if the current mode prediction is different from the previous one
            if self.curr_mode_pred != self.pred_mode_hist[-1]:
                self.detected_changes.append(self.obs_count)

            # append the current mode prediction to the history
            self.pred_mode_hist.append(self.curr_mode_pred)

            # update the expected predictions
            if self.is_stationary:
                self.P_exp = self.learned_modes[self.curr_mode_pred].copy()
            else:
                self.P_exp = self.P.copy()

        self.stationarity_history.append([0,1][self.is_stationary])
        self.obs_count += 1
        self.cnd_vec.append(symbol)
        self.cndi_vec.append(index)

    # processes each symbol in a given sequence, shows progress information
    def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
        if progress:
            pbar_conf = {
                "total": len(sequence),
                "desc": progress_desc.ljust(progress_desc_width),
                "bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
                "ascii": True
            }
            with tqdm(**pbar_conf) as pbar:
                for symbol in sequence:
                    self.process_symbol(symbol)
                    pbar.update()
        else:
            for symbol in sequence:
                self.process_symbol(symbol)
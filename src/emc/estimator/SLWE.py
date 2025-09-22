import numpy as np
from tqdm import tqdm


class SLWE:

	def __init__(self, alpha, lambda_):

		# alphabet cardinality must be known to initialize estimate vectors
		self.alpha = alpha

		# AKA: lambda
		self.lambda_ = lambda_

		# estimate history
		self.initial_estimate = np.ones(self.alpha) / self.alpha
		self.estimates = [self.initial_estimate]

		# symbol to index and index to symbol maps
		self.symbol_index_map = {}
		self.index_symbol_map = {}

	# SLWE update
	def update_estimate(self, symbol_index):

		# multiply the estimate vector with scalar lambda
		new_estimate = self.estimates[-1] * self.lambda_

		# apply rewarding to relevant probability
		new_estimate[symbol_index] += (1 - self.lambda_)

		# append the current estimate to estimate history
		self.estimates.append(new_estimate)

	# manages symbol->index and index->symbol mappings
	# - assigns a new index to given symbol if it has not been observed before
	# - returns index
	def get_index(self, symbol):
		if len(self.symbol_index_map) == 0:
			self.symbol_index_map[symbol] = 0
			self.index_symbol_map[0] = symbol
		elif symbol not in self.symbol_index_map:
			if len(self.symbol_index_map) >= self.alpha:
				print("Unexpected symbol observed: {}".format(symbol))
				exit()
			next_available_index = max(self.symbol_index_map.values()) + 1
			self.symbol_index_map[symbol] = next_available_index
			self.index_symbol_map[next_available_index] = symbol
		index = self.symbol_index_map[symbol]
		return index

	# processes a single symbol
	def process_symbol(self, symbol):
		index = self.get_index(symbol)
		self.update_estimate(index)

	# processes each symbol in a given sequence, shows progress information
	def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
		if progress:
			pbar_conf = {
				"total": len(sequence),
				"desc": progress_desc.ljust(progress_desc_width),
				"bar_format": "{l_bar}{bar:10}{r_bar}{bar:-10b}",
				"ascii": True
			}
			with tqdm(**pbar_conf) as pbar:
				for symbol in sequence:
					self.process_symbol(symbol)
					pbar.update()
		else:
			for symbol in sequence:
				self.process_symbol(symbol)

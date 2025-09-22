import numpy as np


def map_symbols(seq):
    unique_symbols = np.unique(seq)
    symbol_map = {old: new for new, old in enumerate(unique_symbols)}
    new_seq = np.array([symbol_map[x] for x in seq])
    return new_seq, symbol_map

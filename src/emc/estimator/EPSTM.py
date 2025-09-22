import logging
import os
import time
from collections import defaultdict
from itertools import product
from math import log
from pathlib import Path

import more_itertools as mit
import numpy as np
from natsort import natsorted
from sklearn.cluster import AgglomerativeClustering
from treelib import Tree


class PST:

    def __init__(self):

        self.id = ""

        self.alphabet = []
        self.alphabet_cardinality = -1
        self.sequence = []
        self.subsequence_list = []
        self.node_list = []

        # aka L
        self.subsequence_length_limit = 3

        # aka t
        self.subsequence_minimum_occurrence = 2

        # pointer to the tree instance
        self.tree = Tree()

        # logs
        self.se_call_count = 0
        self.ct_call_count = 0

    def initialize_logger(self, log_level, log_file_path):

        if log_file_path:

            # create a custom logger
            self.logger = logging.getLogger(self.id)
            self.logger.setLevel(log_level)

            # create handlers
            f_handler = logging.FileHandler(log_file_path, mode="w")

            # create formatters and add it to handlers
            # f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            f_format = logging.Formatter("%(asctime)s: %(message)s")
            f_handler.setFormatter(f_format)

            # add handlers to the logger
            self.logger.addHandler(f_handler)

        else:

            # create a custom logger
            self.logger = logging.getLogger()
            self.logger.setLevel(log_level)

            self.logger.disabled = True

    def process_sequence(self, sequence, pid_list=None):

        self.sequence = sequence
        if len(self.sequence) == 0:
            print("Initialize sequence first.")
            return

        if pid_list:
            self.alphabet = pid_list
        else:
            self.alphabet = list(set(self.sequence))
        self.logger.debug("Alphabet: {}".format(self.alphabet))
        self.alphabet_cardinality = len(set(self.sequence))
        self.logger.debug("Alphabet Cardinality: {}".format(self.alphabet_cardinality))
        sequence_length = len(self.sequence)
        self.logger.debug("Sequence Length: {}".format(sequence_length))

        # initially, SIL includes all indexes (0,1,...,k) and LCP is empty
        sil = np.arange(sequence_length)
        previous_lcp = []

        # subsequence extraction
        se_start = time.time()
        self.extract_subsequences(sil, previous_lcp)
        se_duration = time.time() - se_start
        self.logger.debug("SE Duration: {}s".format(se_duration))

        # tree construction
        ct_start = time.time()
        self.construct_tree(self.node_list)
        ct_duration = time.time() - ct_start
        self.logger.debug("CT Duration: {}s".format(ct_duration))

    # subsequence extraction phase
    def extract_subsequences(self, sil, previous_lcp):

        self.se_call_count += 1
        self.logger.debug("SE Rec. Call #{}".format(self.se_call_count))
        self.logger.debug("SIL: {}".format(sil))

        # get LCP from the current SIL
        lcp = self.get_lcp(self.sequence, sil)

        # increment SIL according to the length of calculated LCP
        sil = [index+len(lcp) for index in sil]

        # insert LCP from the previous call to the beginning of current LCP
        # a decreasing index is used to maintain order
        # NOTE: insert(0,li) is faster than li=[a]+li
        for i in range(len(previous_lcp)-1, -1, -1):
            lcp.insert(0, previous_lcp[i])

        self.logger.debug("LCP: {}".format(lcp))

        # group SIL according to corresponding tokens
        grouped_sil = self.group_sil(self.sequence, sil)

        self.logger.debug("GSIL: {}".format(grouped_sil))

        # insert subsequences to subsequence list
        # 1 is added to len() to make :i correspond to the whole list
        for i in range(len(lcp)+1):
            current_subsequence = lcp[:i]

            # subsequences that already exist in node list are discarded
            # TODO: ensure that in such cases p does not change
            # NOTE: existence check could be done by iterating over node instances, but will cause more overhead
            if current_subsequence in self.subsequence_list:
                self.logger.debug("Subsequence {} is already in node list.".format(current_subsequence))
                continue

            # subsequences that are longer than L are discarded
            if len(current_subsequence) > self.subsequence_length_limit:
                self.logger.debug("Subsequence {} is discarded. (L={})".format(current_subsequence, self.subsequence_length_limit))
                continue

            # probabilities are stored in a dictionary, 0 probabilities are not included
            p = {}

            # in this case, probabilities are calculated by counting symbols in GIL
            # NOTES:
            # 	- probably this could be done more efficiently, but not less than O(n)
            # 	- total_length is not always equal to len(SIL)
            # 	- total_length (or even p) could be calculated at group_sil function
            if i == len(lcp):
                total_length = 0
                for token in grouped_sil:
                    p[token] = len(grouped_sil[token])
                    total_length += len(grouped_sil[token])
                for token in grouped_sil:
                    p[token] /= total_length

            # in this case, we are sure that there is only one possible next symbol
            else:
                p[lcp[i]] = 1

            # append obtained subsequence and probability vector to node list
            self.subsequence_list.append(current_subsequence)
            new_node = PST_Node(current_subsequence, p)
            self.node_list.append(new_node)

            self.logger.debug("Subsequence {} is added to node list, p={}.".format(current_subsequence, p))

        # depth first recursive call for each group in GSIL
        for token in grouped_sil:
            sil = grouped_sil[token]
            if len(sil) < self.subsequence_minimum_occurrence:
                self.logger.debug("SIL {} is discarded. (t={})".format(sil, self.subsequence_minimum_occurrence))
                continue
            self.extract_subsequences(sil, lcp)

    # finds the longest common prefix of the suffixes starting with indices in SIL
    # NOTE: os.path.commonprefix(LIST) could be used, but it requires all suffixes to be loaded at once
    def get_lcp(self, sequence, sil):
        lcp = []
        for suffix_index_count in range(len(sequence)):
            common_prefix_found = True
            for cur, next in mit.pairwise(sil):
                if cur > len(sequence)-1 or next > len(sequence)-1:
                    common_prefix_found = False
                    break
                if sequence[cur] != sequence[next]:
                    common_prefix_found = False
                    break
            if common_prefix_found:
                lcp.append(sequence[sil[0]])
                sil = [index+1 for index in sil]
                # print("LCP: {}".format(lcp))
            else:
                break
        return lcp

    # groups indices in SIL according to the corresponding character in sequence
    # returns grouped suffix index list as a dictionary
    def group_sil(self, sequence, sil):
        grouped_sil = {}
        for suffix_index in sil:
            if suffix_index > len(sequence)-1:
                continue
            token = sequence[suffix_index]
            if token not in grouped_sil:
                grouped_sil[token] = []
            grouped_sil[token].append(suffix_index)
        return grouped_sil

    # tree construction phase
    def construct_tree(self, node_list, last_child=None):

        self.ct_call_count += 1
        self.logger.debug("CT Rec. Call #{}".format(self.ct_call_count))

        # node with the shortest subsequence is obtained
        nwss = self.get_nwss(node_list)
        self.logger.debug("NwSS: {}".format(nwss.subsequence))

        # node with the shortest subsequence is added to tree
        parent_id = [id(last_child), None][last_child is None]
        # p_str = "\n".join("{}: {:.2f}".format(t, p) for t, p in nwss.p.items())
        self.tree.create_node(
            tag = "{}".format(nwss.subsequence),
            # tag = "{}".format(''.join(map(str, nwss.subsequence))),
            # tag = "{}\\n\\n{}".format(nwss.subsequence, p_str),
            identifier = id(nwss),
            parent = parent_id,
            data = nwss
        )
        self.logger.debug("Node with subsequence {} is added to tree as a children of {}.".format(nwss.subsequence, last_child))

        # node with the shortest subsequence is removed from the node list
        node_list = self.remove_node_from_list(node_list, nwss)
        self.logger.debug("Node with subsequence {} is removed from node list.".format(nwss.subsequence))

        # nodes are grouped according to the nth last character
        n = self.tree.depth(id(nwss)) + 1
        grouped_nl = self.group_nodes(node_list, n)
        self.logger.debug("n: {}".format(n))
        self.logger.debug("GNL: {}".format(grouped_nl))

        # depth first recursive call for each group in GNL
        for token in grouped_nl:
            node_list = grouped_nl[token]
            self.construct_tree(node_list, nwss)

    # typical O(n) min/max calculation to get the node with the shortest subsequence
    def get_nwss(self, node_list):
        shortest_length = self.subsequence_length_limit + 1
        nwss = None
        for node in node_list:
            if len(node.subsequence) < shortest_length:
                shortest_length = len(node.subsequence)
                nwss = node
        return nwss

    # returns a copy of given list without a specific element
    def remove_node_from_list (self, node_list, node):
        return [n for n in node_list if n.subsequence != node.subsequence]

    # groups nodes in the node list according to nth last character
    def group_nodes(self, node_list, n):
        grouped_nl = {}
        for node in node_list:
            # print("SS: {}, nth: {}".format(node.subsequence, node.subsequence[-n]))
            if node.subsequence[-n] not in grouped_nl:
                grouped_nl[node.subsequence[-n]] = []
            # grouped_nl[node.subsequence[-n]].append(node.subsequence)
            grouped_nl[node.subsequence[-n]].append(node)
        return grouped_nl

    # adds min_p to all zero probabilities
    # - second axiom is satisfied by decreasing nonzero probabilities according to their values
    def apply_smoothing(self, min_p=0.000001):
        for node in self.tree.all_nodes():
            zero_p_tokens = []
            for token in self.alphabet:
                if token not in node.data.p:
                    zero_p_tokens.append(token)
            zero_p_token_count = len(zero_p_tokens)
            total_p_to_add = zero_p_token_count * min_p
            for token in node.data.p.keys():
                node.data.p[token] -= node.data.p[token] * total_p_to_add
            for token in zero_p_tokens:
                node.data.p[token] = min_p
            self.logger.debug("p after smoothing: {}".format(node.data.p))

    # calculates the probability of a given sequence being generated from the process modeled with current PST
    # NOTES:
    # - this approach is (among MInD researchers) also called the "Schulz's method"
    # - due to numerical precision issues with products of probabilities, this function follows common practice 
    # 	and use summation of log probabilities
    def predict_sequence(self, sequence):
        self.logger.debug("Predicting sequence {}.".format(sequence))
        window_size = self.subsequence_length_limit
        sum_log_p = 0
        for index in range(len(sequence)):
            start = 0 if index < window_size else (index - window_size)
            self.logger.debug("Searching for P({}|{})".format(sequence[index], sequence[start:index]))
            given, match = self.find_node_with_subsequence(sequence[start:index])
            if sequence[index] in given.data.p:
                p = given.data.p[sequence[index]]
            else:
                # this never happens if smoothing is active
                p = 0
            sum_log_p += log(p)
            self.logger.debug("Found P({}|{}): {}, log: {}".format(sequence[index], given.data.subsequence, p, log(p)))
            self.logger.debug("Total P: {}".format(sum_log_p))
        return sum_log_p

    def iteratively_predict_sequence(self, sequence):
        window_size = self.subsequence_length_limit
        logp = []
        for index in range(len(sequence)):
            start = 0 if index < window_size else (index - window_size)
            given, match = self.find_node_with_subsequence(sequence[start:index])
            if sequence[index] in given.data.p:
                p = given.data.p[sequence[index]]
            else:
                # this never happens if smoothing is active
                p = 0
            logp.append(log(p))
        return logp

    # finds the node with given subsequence
    # returns:
    # - the closest node
    # - a boolean representing if exact match is found
    def find_node_with_subsequence(self, subsequence):
        current_node = self.tree.get_node(self.tree.root)
        match = True
        for i in range(len(subsequence)-1, -1, -1):
            if not match:
                break
            current_subsequence = subsequence[i:len(subsequence)]
            match = False
            for node in self.tree.children(current_node.identifier):
                if node.data.subsequence == current_subsequence:
                    current_node = node
                    match = True
                    break
        return current_node, match

    def print_suffixes(self, sequence, sil):
        for suffix_index in sil:
            print("{}: {}".format(suffix_index, sequence[suffix_index:]))

    def print_node_list(self, node_list):
        for node in node_list:
            print(node.subsequence, node.p)

class PST_Node:

    def __init__ (self, subsequence, p):
        self.subsequence = subsequence
        self.p = p
        # self.children = []

    def add_child (self, child):
        self.children.append(child)

    def __repr__ (self):
        return str(self.subsequence)

    def print_children (self):
        for child in self.children:
            print(child.subsequence, child.p)

# constructs a pst from given primitive id sequence
def construct_pst(
    pst_id, # string, unique ID for the PST
    pid_list, # list, a list of all primitive IDs (to ensure that all symbols exist in PST)
    pid_seq, # list, input symbols (sequence of alphabet primitive ids)
    subsequence_minimum_occurrence, # int, pruning parameter t
    subsequence_length_limit, # int, pruning parameter L
    smoothing_min_p, # double, a small value to add all 0 probabilities
    to_render, # boolean, whether to render the resulting PST
    to_save, # boolean, whether to save the resulting PST with jsonpickle
    output_dir_path, # path, where to save output files
    log_level # string, level of logging [DEBUG, INFO, WARNING] https://docs.python.org/3/library/logging.html#levels
):

    # instantiate pst
    pst_ins = PST()
    pst_ins.id = pst_id
    pst_ins.subsequence_minimum_occurrence = subsequence_minimum_occurrence
    pst_ins.subsequence_length_limit = subsequence_length_limit

    # instantiate logger
    if output_dir_path:
        log_dir_path = os.path.join(output_dir_path, "logs")
        Path(log_dir_path).mkdir(parents=True, exist_ok=True)
        pst_ins.initialize_logger(
            log_level=log_level,
            log_file_path=os.path.join(log_dir_path, "{}.log".format(pst_ins.id))
        )
    else:
        pst_ins.initialize_logger(
            log_level=log_level,
            log_file_path=None
        )
    
    # construct pst from pid_seq
    pst_ins.process_sequence(pid_seq, pid_list)

    # apply smoothing
    pst_ins.apply_smoothing(smoothing_min_p)

    # render tree
    if to_render:
        pass
        # tree_fig_dir_path = os.path.join(output_dir_path, "figures", "trees")
        # Path(tree_fig_dir_path).mkdir(parents=True, exist_ok=True)
        # tree_fig_file_path = "{}/{}".format(tree_fig_dir_path, pst_ins.id)
        # src = Source(pst_ins.to_graphviz())
        # src.render(tree_fig_file_path, format="pdf", cleanup=True, view=False)

    # save tree with jsonpickle
    if to_save:
        pass
        # pst_dir_path = os.path.join(output_dir_path, "pst")
        # Path(pst_dir_path).mkdir(parents=True, exist_ok=True)

        # # pst_json = jsonpickle.encode(pst_ins, keys=True)
        # pst_min_ins = PST_min(
        # 	id=pst_ins.id,
        # 	tree=pst_ins.tree,
        # 	subsequence_minimum_occurrence=pst_ins.subsequence_minimum_occurrence,
        # 	subsequence_length_limit=pst_ins.subsequence_length_limit
        # )
        # pst_json = jsonpickle.encode(pst_min_ins, keys=True)

        # pst_file_path = os.path.join(pst_dir_path, "{}.json".format(pst_ins.id))
        # with open(pst_file_path, "w") as outfile:
        # 	json.dump(pst_json, outfile)
    
    return pst_ins

# this is the matching function from 2020 paper
    # I allows for scaling between two types of matching costs: Dissimilarity & Probability. 
    # The closer I is to 1, the higher the contribution of dissimilarity cost to the total cost and vice versa.
def match_psts(t1, t2, I=0.5):
            
    # gather all the matchings
    # - a matching is a tuple of two nodes
    # - order is important, first element is always from the first tree
    # - from a node, it is possible to access its subsequence and probability vector
    # - these loops could be run inside another loop but it significantly reduces readability
    matchings = []
    for t1_node in t1.tree.all_nodes():
        t2_node, if_exact_match = t2.find_node_with_subsequence(t1_node.data.subsequence)
        matching = (t1_node.data, t2_node.data)
        if matching not in matchings:
            matchings.append(matching)
    for t2_node in t2.tree.all_nodes():
        t1_node, if_exact_match = t1.find_node_with_subsequence(t2_node.data.subsequence)
        matching = (t1_node.data, t2_node.data)
        if matching not in matchings:
            matchings.append(matching)

    # x is not explicitly calculated, instead matchings are iterated 
    # since each matching is composed of subsequences that are already the closest nodes
    matching_cost = 0
    contribution_count = 0
    
    for matching in matchings:
        
        l1 = len(matching[0].subsequence)
        l2 = len(matching[1].subsequence)
        d = np.abs(l1 - l2)
        L = np.max([l1, l2])
        if L == 0: # this only happens while comparing root nodes, where L is irrelevant
            L = 1
        # print("d: {}".format(d))
        # print("L: {}".format(L))

        # epsilon is not calculated explicitly, instead delta is set to 0 if lengths are not equal
        # this way, unnecessary calculation of token set and delta is avoided
        delta = 0
        if l1 == l2:
            k1 = list(matching[0].p.keys())
            k2 = list(matching[1].p.keys())
            k = set(k1 + k2)
            for token in k:
                pt1 = matching[0].p[token] if token in matching[0].p else 0
                pt2 = matching[1].p[token] if token in matching[1].p else 0
                delta += np.abs(pt1 - pt2)
        # print("delta: {}".format(delta))

        contribution = d * I / L + (1 - I) * delta / 2
        contribution_count += 1

        matching_cost += contribution
        
    if contribution_count > 0:
        matching_cost /= contribution_count

    return matching_cost

# calculates pairwise distances from a given list of psts
# returns distances in both dictionary and matrix forms
def match_pst_pairs(pst_list, matching_function, matching_parameters):

    pst_ids = natsorted([pst.id for pst in pst_list])
    pst_id_pairs = list(product(pst_ids, repeat=2))
    dist_pairs = defaultdict(dict)
    for pair in pst_id_pairs:
        dist_pairs[pair[0]][pair[1]] = -1
    for pst1 in pst_list:
        for pst2 in pst_list:
            if matching_function == "epstm_2020":
                dist_pairs[pst1.id][pst2.id] = match_psts(pst1, pst2, I=matching_parameters["I"])
            else:
                print("unknown matching function")
                # pbar.update()

    # prepare distance matrix
    dist_matrix = np.empty([len(pst_ids)+1, len(pst_ids)+1], dtype=object)
    dist_matrix[0,1:] = natsorted([pst.id for pst in pst_list])
    dist_matrix[1:,0] = natsorted([pst.id for pst in pst_list])
    for row_ind, row_id in enumerate(dist_matrix[1:,0]):
        for col_ind, col_id in enumerate(dist_matrix[0,1:]):
            dist_matrix[row_ind+1][col_ind+1] = dist_pairs[row_id][col_id]
    dist_matrix = np.array(dist_matrix[1:,1:], dtype=float)

    return dist_pairs, dist_matrix

def run_epstm(
    pid_list,
    pid_seq,
    change_points,
    subsequence_minimum_occurrence,
    subsequence_length_limit,
    I,
    clustering_threshold,
):
    
    psts = []
    cursor = 0
    for cp_idx, cp in enumerate(change_points[1:]):
        start = cursor
        end = cp
        reg_pid_seq = pid_seq[start:end]
        pst = construct_pst(
            pst_id=f"reg_{start}_{end}",
            pid_list=pid_list,
            pid_seq=reg_pid_seq,
            subsequence_minimum_occurrence=subsequence_minimum_occurrence,
            subsequence_length_limit=subsequence_length_limit,
            smoothing_min_p=0.001,
            to_render=False,
            to_save=False,
            output_dir_path=None,
            log_level="DEBUG"
        )
        psts.append(pst)
        cursor = end
    dist_pairs, dist_matrix = match_pst_pairs(
        pst_list=psts,
        matching_function="epstm_2020",
        matching_parameters={"I": I}
    )
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold= clustering_threshold,
        metric="precomputed",
        linkage="single"
    )
    clusterer.fit(dist_matrix)

    labels_pred = []
    for regime_idx, regime_length in enumerate(np.diff(change_points)):
        labels_pred.extend(np.repeat(clusterer.labels_[regime_idx], regime_length))
    
    return labels_pred
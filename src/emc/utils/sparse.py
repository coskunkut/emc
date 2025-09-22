import numpy as np

def sparse_to_dense(sparse_matrix, k, alpha):

    # dense matrix is initialized with uniform probabilities
    dense_matrix = np.full(shape=np.repeat(alpha, k+1), fill_value=1/alpha)

    # iterate over the keys of the sparse matrix and fill the dense matrix
    for key in sparse_matrix:
        dense_matrix[key] = sparse_matrix[key]

    return dense_matrix
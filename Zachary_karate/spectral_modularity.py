import sys
import snap
import numpy
sys.path.append("/courses/TSKS33/ht2023/common-functions")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from matplotlib import pyplot as plt
from load_data import get_graph
from snap_scipy import to_sparse_mat
from save_Gephi_gexf import save_csrmatrix_Gephi_gexf_twocolors
import time
from gen_stochblock import *

G = get_graph(3) # graph
N = G.GetNodes() # number of nodes
I = np.eye(N) # identity matrix

def power_method(Z):
    """
    Returns dominant eigenvector ψ1 and dominant eigenvalue λ
    """
    eig_vector = np.ones((N,1))

    for _ in range(250):
        eig_vector = Z @ eig_vector # update
        eig_vector = eig_vector / np.linalg.norm(eig_vector) # normalize

    eig_vector = -eig_vector
    eig_value = (eig_vector.T @ Z @ eig_vector).item() 

    return eig_value, eig_vector
    
def spectral_modularity_maximization(Z):
    """
    spectral modularity maximization algorithm using power method.
    Returns eigenvector to largest positive eigenvalue 
    """
    eig_value, eig_vector = power_method(Z)

    if eig_value > 0:
        return eig_value, eig_vector 
    else:
        shifted_Z = Z - eig_value * I 
        eig_value_shifted, eig_vector = power_method(shifted_Z)
        eig_value = eig_value + eig_value_shifted 
        return eig_value, eig_vector

if __name__ == "__main__": 
    start = time.time()
    A = to_sparse_mat(G) # adjacency matrix 
    M = G.GetEdges() # sets of links 
    k = A.sum(axis=1) # degree vector
    Z = A - 1 / (2 * M) * k @ k.T # Z-matrix

    # Z = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         Z[i, j] = A[i, j] - (k[i] * k[j]) / (2 * M)

    dom_eig_value, dom_eig_vector = power_method(Z)
    eig_value, eig_vector = spectral_modularity_maximization(Z)

    # Modularity 
    s = np.where(dom_eig_vector >= 0, 1, -1)
    Q = (1/(4*M)) * (s.T @ Z @ s)

    n1 = np.count_nonzero(s == 1)  # Nodes in community 1 
    n2 = np.count_nonzero(s == -1) # Nodes in community 2 

    # Print result
    print(f"largest-magnitude eigenvalue: {dom_eig_value}")
    print(f"largest-positive eigenvalue: {eig_value}")
    print(f"largest-positive eigenvector: \n {eig_vector}")
    print(f"modularity score: {Q.item():.3f}")
    end = time.time()
    print(f"time: {(end - start):.3f}")
    print(f"number of nodes in community 1: {n1}")
    print(f"number of nodes in community 2: {n2}")
    # Set 
    save_csrmatrix_Gephi_gexf_twocolors(A,"samll.gexf",s)


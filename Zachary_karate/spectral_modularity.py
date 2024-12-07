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

G = get_graph(1) # graph
N = G.GetNodes() # number of nodes
I = np.eye(N) # Identity matrix

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
    A = to_sparse_mat(G) # adjacency matrix 
    M = G.GetEdges() # sets of links 
    k = A.sum(axis=1) # degree vector
    Z = A - 1 / (2 * M) * k @ k.T # Z-matrix

    dom_eig_value, dom_eig_vector = power_method(Z)
    eig_value, eig_vector = spectral_modularity_maximization(Z)

    # Print result
    print(f"largest-magnitude eigenvalue: {dom_eig_value}")
    print(f"largest-positive eigenvalue: {eig_value}")
    print(f"largest-positive eigenvector: \n {eig_vector}")
    
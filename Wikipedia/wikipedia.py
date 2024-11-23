import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from matplotlib import pyplot as plt


"""
Read nodes and edges
"""
def read(file):
    nodes = open("titles/"+file, "r").read().strip().splitlines()
    edges = np.loadtxt("links/"+file, dtype=int)
    return nodes, edges


"""
Create a adjacency matrix given the edges and nr of nodes
"""
def getA(edges, n):
    A = np.zeros((n, n),dtype=int)
    for (i, j) in edges:
        A[j-1][i-1] = 1
    return A

"""
Returns the in and out degree for given adjacency matrix A
"""
def get_degree(A,n):
    u = np.ones(n)
    k_in = np.dot(A, u)
    k_out = np.dot(A.T,u)

    # Normalize 
    k_in /= np.sum(k_in)
    k_out /= np.sum(k_out) 

    return k_in,k_out

if __name__ == "__main__": 

    # Read data 
    file = "2.txt"
    nodes, edges = read(file)
    nr_nodes = len(nodes)
    A = getA(edges, nr_nodes)

    # Task 1
    k_in, k_out = get_degree(A,nr_nodes)
    print(np.sort(k_in)[-5:])
    print(np.argsort(k_in)[-5:])
    # 
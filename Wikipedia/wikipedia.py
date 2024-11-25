import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy import linalg
from matplotlib import pyplot as plt
import pandas as pd


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
    A = np.zeros((n, n))
    edges -= 1
    for (i, j) in edges:
        A[j][i] = 1
    print(A.shape)
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
    pd.set_option('display.float_format', '{:.6f}'.format)
    # Read data 
    file = "2.txt"
    nodes, edges = read(file)
    nr_nodes = len(nodes)
    A = getA(edges, nr_nodes)
    nodes = np.array(nodes)
    # Task 1
    k_in, k_out = get_degree(A,nr_nodes)
    print("--------------Task 1------------")
    in_indices = np.argsort(k_in)[-5:]
    out_indices = np.argsort(k_out)[-5:]
    df1 = pd.DataFrame({
        "Top in-degrees": nodes[in_indices][::-1],
        "In-degree centrality": k_in[in_indices][::-1],
        "Out-degree centrality": k_out[in_indices][::-1]
    })

    print(df1.to_string(index=False))
    print("-"*50) 
    df2 = pd.DataFrame({
        "Top out-degrees": nodes[out_indices][::-1],
        "Out-degree centrality": k_out[out_indices][::-1],
        "In-degree centrality": k_in[out_indices][::-1],
    })
    print(df2.to_string(index=False))
    # Task 2
    print("--------------Task 2------------")

    hub_val, hub_vec = np.linalg.eigh(np.dot(A.T, A))
    auth_val, auth_vec = np.linalg.eigh(np.dot(A, A.T))

    # The dominant eigenvector gives the hub/auth centrality
    hub_centrality = hub_vec[:,-1] / np.sum(hub_vec[:,-1])
    auth_centrality = auth_vec[:,-1] / np.sum(auth_vec[:,-1])

    top_hub_indices = np.argsort(hub_centrality)[-5:]
    df3 = pd.DataFrame({
        "Top hubs": nodes[top_hub_indices][::-1],
        "Hub centrality": hub_centrality[top_hub_indices][::-1],
        "Authority centrality": auth_centrality[top_hub_indices][::-1],
    })

    top_hub_indices = np.argsort(auth_centrality)[-5:]
    df4 = pd.DataFrame({
        "Top authorities": nodes[top_hub_indices][::-1],
        "Authority centrality": auth_centrality[top_hub_indices][::-1],
        "Hub centrality": hub_centrality[top_hub_indices][::-1],
    })

    print(df3.to_string(index=False))
    print("-"*50)
    print(df4.to_string(index=False))
    
    print("--------------Task 3------------")

    eig_val, eig_vec = np.linalg.eig(A)
    idx = np.argmax(eig_val.real)
    eig_centrality = eig_vec.real[:,idx] / np.sum(np.abs(eig_vec.real[:,idx]))
    top_eig_indices = np.argsort(eig_centrality)[-5:]

    df5 = pd.DataFrame({
        "Top eigenvector centrality": nodes[top_eig_indices][::-1],
        "Eigenvector centrality": eig_centrality[top_eig_indices][::-1],
    })
    print(df5.to_string(index=False))

    
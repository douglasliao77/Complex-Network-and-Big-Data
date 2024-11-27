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
    for (i, j) in edges:
        A[j-1][i-1] = 1
    print(A.shape)
    return A

"""
Returns the in and out degree for given adjacency matrix A
"""
def get_degree(A,n):
    u = np.ones(n)
    k_in = np.dot(A, u)
    k_out = np.dot(A.T,u)

    # Normalize the vector
    k_in /= np.sum(k_in)
    k_out /= np.sum(k_out) 

    return k_in,k_out

if __name__ == "__main__": 
    pd.set_option('display.float_format', '{:.6f}'.format)
    # Read data 
    file = "1.txt"
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
    N = A.shape[0]
    eig_val, eig_vec = np.linalg.eig(A)
    idx = np.argmax(eig_val.real)
    # eig_centrality = eig_vec.real[:,idx] / np.sum(eig_vec.real[:,idx])
    # top_eig_indices = np.argsort(eig_centrality)[-5:]

    alpha = 1/eig_val[idx]
    eig_centrality = np.power((alpha * A), 150) @ np.ones((N,))
    eig_centrality = eig_centrality.real / np.sum(eig_centrality.real)
    top_eig_indices = np.argsort(eig_centrality)[-5:]

    df5 = pd.DataFrame({
        "Top eigenvector centrality": nodes[top_eig_indices][::-1],
        "Eigenvector centrality": eig_centrality[top_eig_indices][::-1],
    })
    print(df5.to_string(index=False))

    print("--------------Task 4------------")
    lambda_max = np.abs(eig_val[idx])
    alpha = 0.85 / lambda_max 

    N = A.shape[0] # eigenvector size
    I = np.eye(N) # Identity matrix
    U = np.ones(N) # Vector of ones

    # Ingoing influence - based on the number and importance of nodes pointing
    katz_centrality = (1/N)*np.linalg.inv(I - alpha * A).dot(U) 
    katz_centrality = katz_centrality / np.sum(katz_centrality)
    top_katz_indices = np.argsort(katz_centrality)[-5:]

    df6 = pd.DataFrame({
        "Top Katz": nodes[top_katz_indices][::-1],
        "Katz centrality": katz_centrality[top_katz_indices][::-1],
    })

    print(df6.to_string(index=False))
    print("--------------Task 5------------")
    N = A.shape[0]
    alphas = [0.3, 0.99, 0.85]

    H = A / np.sum(A, axis=0)  
    H[:, np.sum(A, axis=0) == 0] = 1 / N # Nodes with outgoing links set to 1/N

    I = np.eye(N)
    U = np.ones(N)
    _PR = 0 # Used for the comparison in the next task
    for alpha in alphas:
        PR = ((1 - alpha) / N) * np.linalg.inv(I - alpha * H) @ U
        PR = PR / np.sum(PR)
        top_pr_indices = np.argsort(PR)[-5:]
        if alpha == 0.85:
            _PR = PR

        df7 = pd.DataFrame({
            f"Top PageRank, alpha={alpha}": nodes[top_pr_indices][::-1],
            "PageRank": PR[top_pr_indices][::-1],
        })
        print(df7.to_string(index=False))
        print("-"*50)

    print("--------------Task 6------------")
    top_three = top_pr_indices[::-1][:3]
    print("Top-three articles are: " + str(top_three))


    alpha = 0.85
    G = alpha*H + ((1 - alpha) / N) * np.ones((N,N))
    r = np.ones(N) / N
    
    history = np.zeros((101, N))
    history[0] = r
    for i in range(100):
        r = G @ r
        r = r / np.sum(r)
        history[i+1] = r
        if i+1 in [1,2,5,10,100]:
            top_pr_indices = np.argsort(r)[-5:]
            df8 = pd.DataFrame({
                f"Top iterative PageRank, alpha={alpha}": nodes[top_pr_indices][::-1],
                f"PageRank, iteration {i+1}": r[top_pr_indices][::-1],
            })
            print(df8.to_string(index=False))
            print("-"*50)

    # Plot the centrality of top three articles in the iterative method
    plt.figure(1)
    for idx in top_three:
        plt.hlines(
            _PR[idx],
            colors='green',
            xmin=0, xmax=100, 
            label=f"PageRank task 5")

    plt.plot(
        range(101), 
        history[:, top_three[0]], 
        'k--' ,
        label=f"{nodes[top_three[0]]}",
    )
    plt.plot(
        range(101), 
        history[:, top_three[1]], 
        'b--' ,
        label=f"{nodes[top_three[1]]}",
    )
    plt.plot(
        range(101), 
        history[:, top_three[2]], 
        'r--' ,
        label=f"{nodes[top_three[2]]}",
    )
    
    plt.xlabel('Iterations')
    plt.ylabel('PageRank')
    plt.title('Top 3 Articles')
    plt.legend()
    plt.show()


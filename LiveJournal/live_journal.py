#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import snap
from gen_data import genmod10star
from gen_data import genLiveJournal
import random

S = 100000
G, h = genLiveJournal()
# G, h = genmod10star()
"""
Returns the true average degree for graph G 
"""
def exact_average(G, h):
    N = G.GetNodes() 
    res = 0
    for NI in G.Nodes():
        n = NI.GetId()
        res += h[n]

    return res / N

"""
Returns the true random_connection for graph G 
"""
def exact_random_connection(G, h):
    N = G.GetNodes() 
    res = 0
    for NI in G.Nodes():
        n = NI.GetId()
        x = h[n]
        tmp = 0
        for NId in range(NI.GetDeg()):
            n_prime = G.GetNI(NI.GetNbrNId(NId))
            k_prime = n_prime.GetDeg()
            tmp += (1 / k_prime) 
        
        res += tmp*x

    return res / N


"""
Returns the true uniform random walk for graph G 
"""
def exact_uniform_rnd_walk(G, h):
    M = G.GetEdges()
    N = G.GetNodes() 
    res = 0
    for NI in G.Nodes():
        n = NI.GetId()
        k = NI.GetDeg()
        x = h[n]
        res += k*x

    return res / (2*M) 

"""
Return Uniform sampling S_n/S = 1/N when S-> ∞
"""
def uniform_sampling(G, h, seed=1234):
    rnd = snap.TRnd(seed)
    rnd.Randomize()
    res = 0
    for _ in range(S):
        NId = G.GetRndNId(rnd)
        res += h[NId]

    return res / S

def random_connection(G, h, seed=1234):
    rnd = snap.TRnd(seed)
    rnd.Randomize()
    res = 0
    for _ in range(S):
        # Random starting node
        NId = G.GetRndNId(rnd) 
        n_prime = G.GetNI(NId)
        k = n_prime.GetDeg() 
        # Get random neighbor node of n_prime
        n = n_prime.GetNbrNId(random.randint(0, k-1))
        res += h[n]

    return res / S

def uniform_rnd_walk(G, h, seed=1234):
    rnd = snap.TRnd(seed)
    rnd.Randomize()
    # Random starting node
    NId = G.GetRndNId(rnd) 
    n_prime = G.GetNI(NId)

    # Ensure steady state
    try:
        for _ in range(S):
            k = n_prime.GetDeg() 
            n = n_prime.GetNbrNId(random.randint(0, k-1))
            n_prime = G.GetNI(n)
    except:
        print("Could not find a steady state") 
        return None

    res = 0
    for _ in range(S):
        k = n_prime.GetDeg() 
        n = n_prime.GetNbrNId(random.randint(0, k-1))
        n_prime = G.GetNI(n)
        res += h[n]
    
    return res / S

def MH_rnd_walk(G, h, seed):
    rnd = snap.TRnd(seed)
    rnd.Randomize()
    # Random starting node
    NId = G.GetRndNId(rnd) 
    n_prime = G.GetNI(NId)
    res = 0

    for _ in range(S):
        k_prime = n_prime.GetDeg() 
        NId = n_prime.GetNbrNId(random.randint(0, k_prime-1))
        n = G.GetNI(NId)
        k = n.GetDeg() 
        p = random.uniform(0, 1)
        if p < (k_prime/k):
            n_prime = n
        
        res += h[n_prime.GetId()]
    
    return res / S


if __name__ == "__main__": 
    print("-- expected values of <x>-hat -----")
    print(f"expected average: {exact_average(G,h)}")
    print(f"random connection of random node: {exact_random_connection(G,h):.3f} ")
    print(f"uniform random walk: {exact_uniform_rnd_walk(G, h):.3f}")
    print("---estimated <x> -----")

    for i in range(5):
        print(f"uniform sampling: {uniform_sampling(G,h,i):.3f}")

    for i in range(5):
        print(f"random connection of random node: {random_connection(G,h,i):.3f}")

    for i in range(5):
        print(f"uniform random walk: {uniform_rnd_walk(G,h,i):.3f}")

    for i in range(5):
        print(f"M-H random walk: {MH_rnd_walk(G,h,i):.3f}")

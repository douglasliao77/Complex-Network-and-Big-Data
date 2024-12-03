#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Data generation for hands-on session 4

Erik G. Larsson 2020
"""

#import sys
import snap
import numpy

TPATH = "/courses/TSKS33/ht2024/data/"
#TPATH = "./"

def genmod10star():   
    #G1 = snap.GenPrefAttach(1000, 10)
    #G1 = snap.GenRndPowerLaw(100000,2.2)
    G = snap.GenStar(snap.PUNGraph,10,False)
    G.AddEdge(4,5)
    #G = snap.GetMxScc(G1)
    N = G.GetNodes()
    
    # node numbering in the mat file is sequential n=1,2,... following the node iterator
    # assign the degree as attribute
    x = snap.TIntFltH(N)

    for NI in G.Nodes():
        n=NI.GetId()
        x[n]=NI.GetDeg()

    return G, x
 
def genLiveJournal():   
    G1 = snap.LoadEdgeList(snap.PUNGraph, TPATH + "soc-LiveJournal1.txt", 0, 1)
    snap.DelSelfEdges(G1)
    G = snap.GetMxScc(G1)
    N = G.GetNodes()
    x = snap.TIntFltH(N)

    for NI in G.Nodes():
        n=NI.GetId()
        k=NI.GetDeg()
        x[n]=50000*(1+k/10.+0.1*numpy.sin(n))

    return G, x


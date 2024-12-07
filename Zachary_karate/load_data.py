#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Load data for hands-on session 5

Erik G. Larsson 2020
"""

import sys
import snap
import numpy
sys.path.append("/courses/TSKS33/ht2023/common-functions")
from save_Gephi_gexf import saveGephi_gexf
from save_Gephi_gexf import saveGephi_gexf_twocolors
from gen_stochblock import gen_stoch_block_2comm

### GENERATE

#G1=gen_stoch_block_2comm(700,300,3.0/700,3.0/300,0.1/(700+300))
G1=gen_stoch_block_2comm(700,300,10.0/700,10.0/300,0.3/(700+300))
G=snap.GetMxScc(G1)
snap.SaveEdgeList(G, "SB-small-network.txt")

G1=gen_stoch_block_2comm(10000,5000,3.0/10000,3.0/5000,0.1/(10000+5000))
G=snap.GetMxScc(G1)
snap.SaveEdgeList(G, "SB-large-network.txt")

def get_graph(nr):
    if nr == 1:
        return snap.LoadEdgeList(snap.PUNGraph, "test-network.txt", 0, 1)
    elif nr == 2:
        return snap.LoadEdgeList(snap.PUNGraph, "karate-network.txt", 0, 1)
    elif nr == 3:
        return snap.LoadEdgeList(snap.PUNGraph, "SB-small-network.txt", 0, 1)
    else:
        return snap.LoadEdgeList(snap.PUNGraph, "SB-large-network.txt", 0, 1)

from utils import node2vec 
import torch
import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import utils.node2vec
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms import bipartite
from numpy import *
import matplotlib.pyplot as plt 
from itertools import chain
# from args import *
from utils.process import *

#Reachability Computation Function
def get_target_random_walks(args, adj):
    r_adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    edge_index = r_adj._indices()

    graph = nx.Graph()
    edge_list = edge_index.transpose(1,0).tolist()
    #print(edge_list)

    graph.add_edges_from(edge_list)

    
    st_node,walks = graph_random_walks(args ,graph)
    
    return walks


def graph_random_walks(args, graph): #这个函数涉及随机游走操作

    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = 1

    G = node2vec.Graph(graph, args.directed, args.p, args.q,args.fastRandomWalk)
    G.preprocess_transition_probs()
    theta=2.0
    st_node, walks = G.simulate_walks(args.num_walks, args.walk_length,theta)

    return st_node, walks
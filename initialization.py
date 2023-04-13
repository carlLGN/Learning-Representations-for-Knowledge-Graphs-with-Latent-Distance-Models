from load_data import load_subgraph
import numpy as np
import scipy
import networkx as nx


#Function uses subgraph for testing - remember to update!
def initialize(k=2, mode="paper2paper"):

    G = load_subgraph(mode)

    lap=nx.directed_laplacian_matrix(G)
    _, eigenvectors = scipy.linalg.eigh(a=lap, subset_by_index=[0, k-1])

    return eigenvectors


initialize()
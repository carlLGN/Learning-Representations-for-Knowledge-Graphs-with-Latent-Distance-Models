from load_data import load_subgraph
import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt


#Function uses subgraph for testing - remember to update!
def initialize(k=2, mode="author2paper"):

    G = load_subgraph(mode)

    order = [p for p in list(G.nodes) if p.startswith("W")]+[a for a in list(G.nodes) if a.startswith("A")]


    adj = nx.adjacency_matrix(G, nodelist = order).todense()

    A = adj + adj.T

    D = np.diag(np.sum(A, axis=1))

    L = D - A

    Lsym = np.sqrt(np.linalg.inv(D)) @ L @ (np.sqrt(np.linalg.inv(D)))
    _, eigenvectors = scipy.linalg.eigh(a=Lsym, subset_by_index=[1,k])

    return eigenvectors


print(initialize())
plt.scatter(initialize()[:,0],initialize()[:,1])
plt.show()
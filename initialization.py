from load_data import load_subgraph
import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt


#Function uses subgraph for testing - remember to update!
def initialize(k=2, combined=True):
    G1 = load_subgraph(mode="paper2paper")
    G2 = load_subgraph(mode="author2paper")

    if combined:
        #We order by papers in the adjacency matrices so we can easily concatenate.

        paperorder = [p for p in list(G1.nodes)]
        n = len(paperorder)

        adjp2p = nx.adjacency_matrix(G1, nodelist = paperorder).todense()
        adja2p = nx.adjacency_matrix(G2, nodelist = paperorder).todense()

        #We concatenate the adjacency matrices into one adjacency matrix.

        adj = np.concatenate((adjp2p, adja2p), axis=1)

        #We now perform SVD to get u and v

        p_star, s,V = scipy.sparse.linalg.svds(adj, k=k)


        #U is the initialization of citing papers (p*)
        #V is the initalization of both cited papers (p) and authors (a)

        p = V[:n]
        a = V[a:]


        return p_star, p, a


    #This should maybe use SVD instead??
    else:
        Graphs = [G1, G2]

        inits = []
        for G in graphs:
            papers = [p for p in list(G.nodes) if p.startswith("W")]
            authors = [a for a in list(G.nodes) if a.startswith("A")]

            order = papers + authors

            adj = nx.adjacency_matrix(G, nodelist = order).todense()

            A = adj + adj.T

            D = np.diag(np.sum(A, axis=1))

            L = D - A

            Lsym = np.sqrt(np.linalg.inv(D)) @ L @ (np.sqrt(np.linalg.inv(D)))
            _, eigenvectors = scipy.linalg.eigh(a=Lsym, subset_by_index=[1,k])
            inits.append(eigenvectors)

        return inits[0], inits[1]


eigenvectors, npaper, nauthor = initialize()

plt.scatter(eigenvectors[:npaper,0],eigenvectors[:npaper,1], c="blue", marker = ".", alpha=0.5)
plt.scatter(eigenvectors[npaper:,0],eigenvectors[npaper:,1], c="orange", marker = "^", alpha = 0.5)
plt.show()
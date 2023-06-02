from load_data import load_subgraph
from load_data import load_data
import numpy as np
import scipy
import networkx as nx


#Function uses subgraph for testing - remember to update!
def initialize(k=2, combined=True):
    G1 = load_data(path="./Data/paper2paper.gml")
    print("First Graph Loaded")
    G2 = load_data(path="./Data/author2paper.gml")

    print("Data Loaded")


    nodelist=list(G1.nodes())
    N_nodes = len(nodelist)
    for i, node in enumerate(nodelist):
        print(f'{i/N_nodes*100} %')
        if not G2.has_node(node):
            G1.remove_node(node)


    if combined:
        #We order by papers in the adjacency matrices so we can easily concatenate.

        paperorder = [p for p in list(G1.nodes)]
        n = len(paperorder)

        print("Constructing Adjacency Matrices")
        adjp2p = nx.adjacency_matrix(G1, nodelist = paperorder)
        adja2p = nx.adjacency_matrix(G2, nodelist = paperorder)

        #We concatenate the adjacency matrices into one adjacency matrix.

        print("Concatenating")
        adj = scipy.sparse.hstack([adjp2p,adja2p])

        #We now perform SVD to get u and v

        print("performing SVD")
        p_star, s,V = scipy.sparse.linalg.svds(adj, k=k)

        print("SVD Done")
        #U is the initialization of citing papers (p*)
        #V is the initalization of both cited papers (p) and authors (a)

        p = V[:n]
        a = V[n:]


        return p_star, p, a


    #This should maybe use SVD instead??
    else:
        Graphs = [G1, G2]

        inits = []
        for G in Graphs:
            papers = [p for p in list(G.nodes) if p.startswith("W")]
            authors = [a for a in list(G.nodes) if a.startswith("A")]

            order = papers + authors

            adj = nx.adjacency_matrix(G, nodelist = order)

            A = adj + adj.T

            D = np.diag(np.sum(A, axis=1))

            L = D - A

            Lsym = np.sqrt(np.linalg.inv(D)) @ L @ (np.sqrt(np.linalg.inv(D)))
            _, eigenvectors = scipy.linalg.eigh(a=Lsym, subset_by_index=[1,k])
            inits.append([eigenvectors])

        return inits[0], inits[1]


def save_initializations(k=2, combined=True):

    if combined:
        p_star, p, a = initialize(k=k, combined=True)

        inits = [p_star, p, a]

    else:
        p2p, a2p = initialize(k=k, combined=False)

        inits = [p2p, a2p]
    #this is a bug:
    print("Saving Embeddings")
    for values in inits:
        with open(f"{values}_init.txt", 'w',encoding="uft-8") as f:
            f.write(f"{np.shape(values)}\n")
            for i in range(len(values)):
                f.write(f"{i}" + " " + f"{values[i]}\n")

save_initializations(k=2, combined=True)
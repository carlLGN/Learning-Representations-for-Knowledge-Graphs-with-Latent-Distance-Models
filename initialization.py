from load_data import load_subgraph
from load_data import load_data
import numpy as np
import scipy
import networkx as nx
from tqdm import tqdm



def initialize(k=2, combined=True):
    G1 = load_data(path="./Data/paper2paper.gml")
    print("First Graph Loaded")
    G2 = load_data(path="./Data/author2paper.gml")

    #The Author to Paper graph is loaded as a directed graph.

    G2 = G2.to_undirected()


    print("Data Loaded")

    #This piece of code handles papers that appear in the citation network,
    #but not in the author-paper netowrk
    nodelist=list(G1.nodes())
    for node in tqdm(nodelist):
        if not G2.has_node(node):
            G1.remove_node(node)


    if combined:
        #We order by papers in the adjacency matrices so we can easily concatenate.

        paperorder = [paper for paper in list(G1.nodes)]
        authororder = [author for author in list(G2.nodes) if author.startswith('A')]
        n = len(paperorder)


        print("Constructing Adjacency Matrices")
        adjp2p = nx.adjacency_matrix(G1, nodelist = paperorder)
        adja2p = nx.bipartite.biadjacency_matrix(G2, row_order = paperorder+authororder, column_order = paperorder+authororder)[:n,n:]

        #We concatenate the adjacency matrices into one adjacency matrix.

        print("Concatenating")
        adj = scipy.sparse.hstack([adjp2p,adja2p], dtype=np.single)

        #We now perform SVD to get u (p*) and v

        print("performing SVD")
        #DATATYPE PROBLEMS?
        p_star, _, V = scipy.sparse.linalg.svds(adj, k=k)

        print("SVD Done")
        #U is the initialization of citing papers (p*)
        #V is the initalization of both cited papers (p) and authors (a)

        p = V.T[:n]
        a = V.T[n:]


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
        name = ['p_star','p','a']

    else:
        p2p, a2p = initialize(k=k, combined=False)

        inits = [p2p, a2p]
        name = ['p2p', 'a2p']

    print("Saving Embeddings")
    for i, values in enumerate(inits):
        with open(f"./Embeddings/{name[i]}_init.emb", 'w',encoding="utf-8") as f:
            f.write(f"{np.shape(values)}\n")
            for i in tqdm(range(len(values))):
                f.write(f"{i}" + " " + f"{values[i][0]}"+" "+f"{values[i][0]}\n")


#Tilføjet in case man vil kalde save initialization fra andre filer
if __name__ == '__main__':
    save_initializations(k=2, combined=True)
    
    print('debug')
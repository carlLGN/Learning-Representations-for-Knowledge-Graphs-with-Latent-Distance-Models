from load_data import load_data
import numpy as np
import scipy
import networkx as nx
from tqdm import tqdm
from operator import itemgetter



def initialize(k=2, combined=True):
    G1 = load_data(path="./Data/paper2paper_2000_gcc.gml")
    print("First Graph Loaded")
    G2 = load_data(path="./Data/author2paper_2000_gcc.gml")

    #The Author to Paper graph is loaded as a directed graph.

    print("Fixing Second Graph Structure")
    G2 = G2.to_undirected()


    print("Data Loaded")

    #This piece of code handles papers that appear in the citation network,
    #but not in the author-paper netowrk
    nodelist=list(G1.nodes())
    for node in tqdm(nodelist):
        if not G2.has_node(node):
            G1.remove_node(node)

    dates = set([(lis[0], lis[2]) for lis in G1.edges(data='date')])
    dates_sorted = sorted(dates, key=itemgetter(1))
    paperorder = list(np.asarray(dates_sorted)[:, 0][::-1])



    #Some papers are never cited
    paperset = set(paperorder)
    for node in G1.nodes:
        if node not in paperset:
            paperorder.append(node)

    if combined:
        #We order by papers in the adjacency matrices so we can easily concatenate.

        authororder = [author for author in list(G2.nodes) if author.startswith('A')]
        n = len(paperorder)
        m = len(authororder)


        print("Constructing Adjacency Matrices")
        adjp2p = nx.adjacency_matrix(G1, nodelist = paperorder)
        adja2p = nx.bipartite.biadjacency_matrix(G2, row_order = paperorder+authororder, column_order = paperorder+authororder)[:n,n:]

        #We concatenate the adjacency matrices into one adjacency matrix.

        print("Concatenating")
        adj = scipy.sparse.hstack([adjp2p,adja2p], dtype = np.single)


        print("Creating Laplacian Matrix")
        ul = scipy.sparse.csr_matrix((n,n))
        lr = scipy.sparse.csr_matrix((n+m, n+m))

        top_row = scipy.sparse.hstack([ul, adj], dtype = np.single)
        bot_row = scipy.sparse.hstack([adj.T, lr], dtype = np.single)

        L = scipy.sparse.vstack([top_row, bot_row], dtype = np.single)


        #We now perform SVD to get u (p*) and v

        print("Computing Eigenvectors")

        _, eigenvectors = scipy.sparse.linalg.eigsh(L, k=k+1)

        print("Eigenvectors Computed")

        p_star = eigenvectors[:n, 1:k+1]
        p = eigenvectors[n:2*n, 1:k+1]
        a = eigenvectors[2*n:, 1:k+1]

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
                f.write(f"{i}" + " " + f"{values[i][0]}"+" "+f"{values[i][1]}\n")


#Tilføjet in case man vil kalde save initialization fra andre filer
if __name__ == '__main__':
    save_initializations(k=2, combined=True)
    
    print('debug')
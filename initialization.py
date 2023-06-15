from load_data import load_data
import numpy as np
import scipy
import networkx as nx
from tqdm import tqdm
from operator import itemgetter
from paper_size import read_emb3

def initialize(k=2, mode="paper2paper"):
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
    for node in tqdm(list(G1.nodes)):
        if node not in paperset:
            paperorder.append(node)

    #We order by papers in the adjacency matrices so we can easily concatenate.

    authororder = [author for author in list(G2.nodes) if author.startswith('A')]
    n = len(paperorder)
    m = len(authororder)


    print("Constructing Adjacency Matrices")
    adjp2p = nx.adjacency_matrix(G1, nodelist = paperorder)
    adja2p = nx.bipartite.biadjacency_matrix(G2, row_order = paperorder+authororder, column_order = paperorder+authororder)[:n,n:]

    if mode == "paper2paper":
        testset = read_emb3("./Data/test_edgelist_pp")
        for edge in testset:
            sending = int(edge[0])
            receiving = int(edge[1])
            adjp2p[sending, receiving] = 0
        adjp2p.eliminate_zeros()

    elif mode == "author2paper":
        testset = read_emb3("./Data/test_edgelist_ap")


    #We concatenate the adjacency matrices into one adjacency matrix.

    print("Concatenating")
    adj = scipy.sparse.hstack([adjp2p,adja2p], dtype = np.single)


    print("Creating Laplacian Matrix")
    ul = scipy.sparse.csr_matrix((n,n))
    lr = scipy.sparse.csr_matrix((n+m, n+m))

    top_row = scipy.sparse.hstack([ul, adj], dtype = np.single)
    bot_row = scipy.sparse.hstack([adj.T, lr], dtype = np.single)

    A = scipy.sparse.vstack([top_row, bot_row], dtype = np.single)
    D = scipy.sparse.diags([d[0] for d in A.sum(axis=1).A])
    Dinv = scipy.sparse.diags([d[0]**-1 if d[0] != 0 else d[0] for d in A.sum(axis=1).A])
    L = D - A

    print("Getting eigenvectors for L")
    eigenvalues_L, eigenvectors_L = scipy.sparse.linalg.eigsh(L, k=k, sigma=0,tol=1e-4, which="LM")


    print("Creating L_sym")
    sqrtDeg = Dinv.sqrt()
    L_sym =  sqrtDeg @ L @ sqrtDeg


    #We now perform SVD to get u (p*) and v

    print("Computing Eigenvectors")

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L_sym, k=k, sigma=0, tol=1e-4, which="LM")

    print("Eigenvectors Computed")

    p_star = eigenvectors[:n, :]
    p = eigenvectors[n:2*n, :]
    a = eigenvectors[2*n:, :]

    return p_star, p, a, eigenvalues, eigenvectors_L, eigenvalues_L

def save_initializations(k=2, mode="paper2paper"):


    p_star, p, a, eigenvalues, evec_L, eval_L = initialize(k=k, mode=mode)

    inits = [p_star, p, a]
    name = ['p_star','p','a']


    print("Saving Embeddings")
    for k, values in enumerate(inits):
        with open(f"./Embeddings/{name[k]}_init.emb", 'w',encoding="utf-8") as f:
            f.write(f"{np.shape(values)}\n")
            for i in tqdm(range(len(values))):
                f.write(f"{i}")
                for j in range(np.shape(values)[1]):
                    f.write(" " + f"{values[i,j]}")
                f.write("\n")
    print("Eigenvalues for L_sym: " + f"{eigenvalues}")
    print("Embeddings saved")

    print("Saving L's Eigenvectors")
    with open(f"./Embeddings/eigenvectors_L.emb", 'w',encoding="utf-8") as f:
        f.write(f"{np.shape(e_L)}\n")
        for i in tqdm(range(len(e_L))):
            f.write(f"{i}")
            for j in range(np.shape(e_L)[1]):
                f.write(" " + f"{e_L[i,j]}")
            f.write("\n")
    print("Eigenvalues for L: "+f"{eval_L}")

if __name__ == '__main__':



    save_initializations(k=10, mode="paper2paper")
    
    print('debug')
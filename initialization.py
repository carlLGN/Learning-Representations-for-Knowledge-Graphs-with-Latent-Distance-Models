import numpy as np
import scipy
from tqdm import tqdm
from paper_size import read_emb3

def initialize(k=2, mode="paper2paper"):
    print("Loading Data")
    if mode == "paper2paper":
        pp = np.array(read_emb3("./Data/train_edgelist_pp"))
        ap = np.array(read_emb3("./Data/author2paper_edgelist"))

    elif mode =="author2paper":
        pp = np.array(read_emb3("./Data/paper2paper_edgelist"))
        ap = np.array(read_emb3("./Data/train_edgelist_ap"))

    print("Creating Adjacency Matrices")

    n = int(np.max(pp[:,1]))+1
    m = int(np.max(ap[:,1]))+1

    data_pp = pp[:,2]
    row_pp = pp[:,0]
    col_pp = pp[:,1]

    adjp2p = scipy.sparse.csr_matrix((data_pp, (row_pp, col_pp)), shape=(n,n))

    data_ap = ap[:,2]
    row_ap = ap[:,0]
    col_ap = ap[:,1]-n

    adja2p = scipy.sparse.csr_matrix((data_ap, (row_ap, col_ap)), shape=(n, m-n))


    #We concatenate the adjacency matrices into one adjacency matrix.

    print("Concatenating")
    adj = scipy.sparse.hstack([adjp2p,adja2p], dtype = np.single)


    print("Creating Laplacian Matrix")
    ul = scipy.sparse.csr_matrix((n,n))
    lr = scipy.sparse.csr_matrix((m, m))

    top_row = scipy.sparse.hstack([ul, adj], dtype = np.single)
    bot_row = scipy.sparse.hstack([adj.T, lr], dtype = np.single)

    A = scipy.sparse.vstack([top_row, bot_row], dtype = np.single)
    D = scipy.sparse.diags([d[0] for d in A.sum(axis=1).A])
    Dinv = scipy.sparse.diags([d[0]**-1 if d[0] != 0 else d[0] for d in A.sum(axis=1).A])
    L = D - A

    print("Getting eigenvectors for L")

    maxeval = scipy.sparse.linalg.eigsh(L, k=1)[0]
    Lflip = maxeval[0]*scipy.sparse.eye(L.shape[0]) - L

    bigevals, evec_L = scipy.sparse.linalg.eigsh(Lflip, k=k,tol=1e-3, which="LM")
    eval_L = maxeval - bigevals
    # eigenvalues_L, eigenvectors_L = scipy.sparse.linalg.eigsh(L, k=k, sigma=0,tol=1e-3, which="LM")


    print("Creating L_sym")
    sqrtDeg = Dinv.sqrt()
    L_sym =  sqrtDeg @ L @ sqrtDeg


    #We now perform SVD to get u (p*) and v

    print("Computing Eigenvectors")

    maxeval = scipy.sparse.linalg.eigsh(L_sym, k=1)[0]
    L_symflip = maxeval[0] * scipy.sparse.eye(L.shape[0]) - L_sym

    bigevals, evec_L_sym = scipy.sparse.linalg.eigsh(L_symflip, k=k, tol=1e-3, which="LM")
    eval_L_sym = maxeval - bigevals

    print("Eigenvectors Computed")

    p_star = evec_L_sym[:n, :]
    p = evec_L_sym[n:2*n, :]
    a = evec_L_sym[2*n:, :]

    return p_star, p, a, eval_L_sym, evec_L, eval_L

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
        f.write(f"{np.shape(evec_L)}\n")
        for i in tqdm(range(len(evec_L))):
            f.write(f"{i}")
            for j in range(np.shape(evec_L)[1]):
                f.write(" " + f"{evec_L[i,j]}")
            f.write("\n")
    print("Eigenvalues for L: "+f"{eval_L}")

if __name__ == '__main__':



    save_initializations(k=5, mode="author2paper")
    
    print('debug')
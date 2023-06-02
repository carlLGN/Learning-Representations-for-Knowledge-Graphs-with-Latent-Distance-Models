import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



#plotting functions:

def read_emb(path):
    points=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens)==2:
                pass
            else:
                u, v = float(tokens[1]),float(tokens[2])

                points.append((u,v))
    return points

def visualize(embs):

    # Dimension must be 2
    assert embs.shape[1] == 2, "The dimension of embeddings must be 2"

    plt.figure()
    plt.scatter(embs[:, 0], embs[:, 1], s=1)
    plt.show()


#embs = np.asarray(read_emb("ldm_paper2papertest.emb"))
#print(embs)
#visualize(embs)


import matplotlib.pyplot as plt
import numpy as np


# Read embeddings
def read_emb(path):
    
    edges = []
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            u, v = int(tokens[0]), int(tokens[1])
            if v > u:
                temp = v
                v = u
                u = temp
            edges.append((u, v))
            
    return edges

# Visualize the embeddings
def visualize(embs):

    # Dimension must be 2
    assert embs.shape[1] == 2, "The dimension of embeddings must be 2"

    plt.figure()
    plt.scatter(embs[:, 0], embs[:, 1], s=1)
    plt.show()
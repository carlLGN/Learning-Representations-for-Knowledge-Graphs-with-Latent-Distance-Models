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
    
    
def read_embeddings(files):
    #Files must be in correct order
    #P-star, P, A
    embs = [[], [], []]
    
    for i, file in enumerate(files):
        with open(file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                line = line.split()
                if j == 0:
                    #(_, dim) = line
                    continue
                else:
                    embs[i].append([float(line[k]) for k in range(1, len(line))])
        
        embs[i] = np.array(embs[i])
        
    return embs[0], embs[1], embs[2]

if __name__ == '__main__':
    read_embeddings(["./Embeddings/p_star_init.emb", "./Embeddings/p_init.emb", "./Embeddings/a_init.emb"])

            
            
            
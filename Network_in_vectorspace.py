from Visualization import read_emb, visualize
from load_data import load_subgraph
import numpy as np
import matplotlib.pyplot as plt

ldm_paper2paper = np.asarray(read_emb('./Embeddings/a_init.emb'))

def read_edges(path):
    edges=[]
    with open(path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            u, v = float(tokens[0]),float(tokens[1])

            edges.append((u,v))
    return np.asarray(edges)


def paper2paper_in_network(path):
    edges=read_edges(path)

    xval=edges[:,0]
    yval=edges[:,1]

    for i in range(len(xval)):
        xvalues=ldm_paper2paper[int(xval[i])]
        yvalues=ldm_paper2paper[int(yval[i])]

        plt.plot(xvalues,yvalues)

    plt.show()



def paper2paper_in_network_specific_points(path, nodes_looked_at): 

    edges=read_edges(path)

    xval=edges[:,0]
    yval=edges[:,1]

    plt.scatter(ldm_paper2paper[:,0], ldm_paper2paper[:,1], alpha=0.1)

    for i in range(len(xval)):
        if xval[i] in nodes_looked_at or yval[i] in nodes_looked_at:
            xvalues=ldm_paper2paper[int(xval[i])]
            yvalues=ldm_paper2paper[int(yval[i])]

            plt.plot(xvalues,yvalues, color='red', alpha=0.6)
            plt.scatter(xvalues, yvalues, alpha=0.6, color='red')

        else:
            pass

    plt.show()



if __name__ == '__main__':
    paper2paper_in_network_specific_points('./Data/paper2paper_edgelist', [6, 54, 388])
    #plt.scatter(ldm_paper2paper[:,0], ldm_paper2paper[:,1])
    #plt.show()









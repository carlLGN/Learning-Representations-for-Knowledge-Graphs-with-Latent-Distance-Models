from Visualization import read_emb
from Network_in_vectorspace import read_edges
import numpy as np
import matplotlib.pyplot as plt

ldm_citingpapers = np.asarray(read_emb('./Embeddings/p_init.emb'))
ldm_citedpapers = np.asarray(read_emb('./Embeddings/p_star_init.emb'))
ldm_authors = np.asarray(read_emb('./Embeddings/a_init.emb'))


def read_edges(path):
    edges=[]
    with open(path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            u, v = float(tokens[0]),float(tokens[1])

            edges.append((u,v))
    return np.asarray(edges)


def all_together():
    
    plt.scatter(ldm_citingpapers[:,0],ldm_citingpapers[:,1], alpha=0.5, color='red')

    plt.scatter(ldm_citedpapers[:,0], ldm_citedpapers[:,1], alpha=0.5, color='blue')

    plt.scatter(ldm_authors[:,0],ldm_authors[:,1], alpha=0.5, color='green')

    plt.show()


def all_together_nodes(nodes):

    for i in range(len(nodes)):

        i=nodes[i]

        plt.scatter(ldm_citingpapers[i,:][0],ldm_citingpapers[i,:][1], alpha=0.5, color='red')
        plt.scatter(ldm_citedpapers[i,:][0], ldm_citedpapers[i,:][1], alpha=0.5, color='blue')
        plt.scatter(ldm_authors[i,:][0],ldm_authors[i,:][1], alpha=0.5, color='green')

    plt.show()


def all_together_edges(nodes):

    a2p=read_edges('./Data/paper2paper_edgelist')
    p2p=read_edges('./Data/author2paper_edgelist')

    citing1=a2p[:,0]
    author=a2p[:,1]

    citing2=p2p[:,0]
    cited=p2p[:,1]

    for j in range(len(citing1)):

       #nodes are thus refering to citing papers
        if citing1[j] in nodes:

            citingnode1=ldm_citingpapers[int(citing1[j])]
            authornode=ldm_authors[int(author[j])]

            plt.plot(citingnode1[0],citingnode1[1], marker='o', color='red', alpha=0.5)
            plt.plot(authornode[0],authornode[1], marker='o', color='green', alpha=0.5)
            plt.plot([citingnode1[0],authornode[0]],[citingnode1[1],authornode[1]], linestyle='-', alpha=0.5, color='green')

    for k in range(len(citing2)):
        
        if citing2[k] in nodes:

            citingnode2=ldm_citingpapers[int(citing2[k])]
            citednode=ldm_citedpapers[int(cited[k])]

            plt.plot(citingnode2[0], citingnode2[1], marker='o', color='red', alpha=0.5)
            plt.plot(citednode[0], citednode[1], marker='o', color='blue', alpha=0.5)
            plt.plot([citingnode2[0],citednode[0]],[citingnode2[1],citednode[1]], linestyle='-', alpha=0.5, color='blue')


    
    plt.show()



if __name__ == '__main__':
    all_together_edges([5])



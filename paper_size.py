import numpy as np
import matplotlib.pyplot as plt
from Visualization import read_emb
import matplotlib.cm as cm
import matplotlib as mpl


# read_emb edited from Visualization to import all 3 columns
def read_emb3(path):
    points=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens)==2:
                pass
            else:
                s, u, v = float(tokens[0]),float(tokens[1]),float(tokens[2])

                points.append((s,u,v))
    return points

# debuging
#print(np.shape(np.asarray(read_emb('./Embeddings/a_init.emb'))))
#print(len(np.unique(np.asarray(read_emb3('Data/author2paper_edgelist'))[:,0])))


def paper_size(data):
    # Papers cited by other papers. Index occurs every time a paper is cited.
    #all_paper_citations = np.concatenate((data[:,0],data[:,1]))
    all_paper_citations = data[:,1]

    amount_of_citations = np.zeros(np.max(all_paper_citations).astype(int)+1)
    for i in range(len(all_paper_citations)):
        amount_of_citations[all_paper_citations[i].astype(int)] += 1
    return amount_of_citations


def plot_paper_size(path):
    data = np.asarray(read_emb(path))

    x = data[:, 0]
    y = data[:, 1]

    s = paper_size(paper2paper_edgelist).astype(float)*2
    color = 1/np.sum(s)*s
    plt.scatter(x, y, s=s, c=color, alpha=.8,cmap=mpl.colormaps['winter_r'],edgecolors='black', linewidth=0.3)
    plt.colorbar()
    plt.show()


def author_size(data):
    # Index occurs every time a auhtor has written a paper.
    all_authors = data[:,0]

    amount_of_papers_written = np.zeros(np.max(all_authors).astype(int) + 1)
    for i in range(len(all_authors)):
        amount_of_papers_written[all_authors[i].astype(int)] += 1
    return amount_of_papers_written

def plot_author_size(path):
    data = np.asarray(read_emb(path))

    x = data[:, 0]
    y = data[:, 1]

    s = paper_size(author2paper_edgelist).astype(float)*2
    color = 1/np.sum(s)*s
    plt.scatter(x, y, s=s, c=color, alpha=.8,cmap=mpl.colormaps['winter_r'],edgecolors='black', linewidth=0.3)
    plt.colorbar()
    plt.show()




if __name__ == '__main__':
    author2paper_edgelist = np.asarray(read_emb3('Data/author2paper_edgelist'))
    # print(author2paper_edgelist)

    paper2paper_edgelist = np.asarray(read_emb3('Data/paper2paper_edgelist'))
    # print(paper2paper_edgelist)

    print(plot_paper_size('./Embeddings/p_init.emb'))
    #print(plot_paper_size('./Embeddings/a_init.emb'))

#def author_size():
#    count_papers_pr_author = np.zeros()
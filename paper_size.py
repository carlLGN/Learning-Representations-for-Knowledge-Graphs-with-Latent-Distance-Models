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

author2paper_edgelist = np.asarray(read_emb3('Data/author2paper_edgelist'))
#print(author2paper_edgelist)

paper2paper_edgelist = np.asarray(read_emb3('Data/paper2paper_edgelist'))
#print(paper2paper_edgelist)

ldm_paper2paper = np.asarray(read_emb('LDM/ldm_paper2paper.emb'))

def paper_size(data):
    # Papers cited by other papers. Index occurs every time a paper cited.
    #all_paper_citations = np.concatenate((data[:,0],data[:,1]))
    all_paper_citations = data[:,1]

    amount_of_citations = np.zeros(np.max(all_paper_citations).astype(int)+1)
    for i in range(len(all_paper_citations)):
        amount_of_citations[all_paper_citations[i].astype(int)] += 1
    return amount_of_citations

#print(paper_size(paper2paper_edgelist))

def plot_paper_size(data):
    x = data[:, 0]
    y = data[:, 1]

    s = paper_size(paper2paper_edgelist).astype(float)*2
    color = s/np.max(s)
    plt.scatter(x, y, s=s, c=1-color, alpha=.8,cmap=mpl.colormaps['winter'],edgecolors='black', linewidth=0.3)
    plt.colorbar()
    plt.show()

print(plot_paper_size(ldm_paper2paper))

#def author_size():
#    count_papers_pr_author = np.zeros()
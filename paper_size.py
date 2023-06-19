import numpy as np
import matplotlib.pyplot as plt
from Visualization import read_emb
import matplotlib.cm as cm
import matplotlib as mpl
import torch
from LDM.src.multimodal_ldm import Multimodal_LDM


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


def read_emb_general(path):
    points=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens)==2:
                pass
            else:
                points.append([float(tokens[i]) for i in range(len(tokens))])
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
        amount_of_citations[int(all_paper_citations[i])] += 1
    return amount_of_citations


def plot_paper_size(path):
    #data = np.asarray(read_emb_general(path))
    data = path

    x = data[:, 0]
    y = data[:, 1]

    s = paper_size(paper2paper_edgelist).astype(float)*2
    color = 1/np.sum(s)*s
    plt.scatter(x, y, s=s, c=color, alpha=.8,cmap=mpl.colormaps['winter_r'],edgecolors='black', linewidth=0.3)
    plt.colorbar()
    #ax = plt.gca()
    #ax.set_xlim([-1, 1])
    #ax.set_ylim([-1, 1])
    plt.show()


def author_size(data):
    # Index occurs every time a auhtor has written a paper.
    all_authors = data[:,1] - np.max(data[:,0]+1)

    amount_of_authors = np.zeros(np.max(all_authors).astype(int) + 1)
    for i in range(len(all_authors)):
        amount_of_authors[all_authors[i].astype(int)] += 1
    return amount_of_authors

def plot_author_size(path):
    #data = np.asarray(read_emb_general(path))
    data = path

    x = data[:, 0]
    y = data[:, 1]

    s = author_size(author2paper_edgelist).astype(float)*2
    color = 1/np.sum(s)*s
    plt.scatter(x, y, s=s, c=color, alpha=.8,cmap=mpl.colormaps['winter_r'],edgecolors='black', linewidth=0.3)
    plt.colorbar()
    #ax = plt.gca()
    #ax.set_xlim([-7, -5])
    #ax.set_ylim([-1.5, 1.5])
    plt.show()




if __name__ == '__main__':
    def read_edge(path):
        points = []
        with open(path, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split()
                s, u = float(tokens[0]), float(tokens[1])

                points.append((s, u))
        return points

    device = torch.device('cpu')
    args = ['./Data/train_edgelist_pp', './Data/train_edgelist_ap', './ldm_paper2papertest2.emb', 2, 5000, 1, 0.5, 500,
            0.1, 19, 1, 0]
    dataset_pp_path = args[0]
    dataset_ap_path = args[1]
    emb_path = args[2]
    dim = args[3]
    epoch_num = args[4]
    steps_per_epoch = args[5]
    alpha = args[6]
    batch_size = args[7]
    lr = args[8]

    seed = args[9]
    verbose = args[10]
    visualize = args[11]

    edges_pp = read_edge(dataset_pp_path)
    edges_pp = torch.as_tensor(edges_pp, dtype=torch.int, device=torch.device("cpu")).T

    edges_ap = read_edge(dataset_ap_path)
    edges_ap = torch.as_tensor(edges_ap, dtype=torch.int, device=torch.device("cpu")).T

    model = Multimodal_LDM(edges_pp=edges_pp, edges_ap=edges_ap, dim=dim, lr=lr, epoch_num=epoch_num,
                           batch_size=batch_size, spe=steps_per_epoch, device=torch.device(device), verbose=verbose,
                           seed=seed)
    model.load_state_dict(torch.load(r'./Embeddings/0.5_0', map_location=device))





    author2paper_edgelist = np.asarray(read_emb3('Data/train_edgelist_ap'))
    # print(author2paper_edgelist)

    paper2paper_edgelist = np.asarray(read_emb3('Data/paper2paper_edgelist'))
    # print(paper2paper_edgelist)

    #print(plot_paper_size('./Embeddings/p_init_p2p.emb'))
    #print(plot_author_size('./Embeddings/a_init_p2p.emb'))

    #print(plot_author_size(model.a.detach().numpy()))
    print(plot_paper_size(model.p_star.detach().numpy()))
#def author_size():
#    count_papers_pr_author = np.zeros()
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


'''Functions for reading different files'''
'''This function, contrary to its name, reads edgelists
   - returns a Nx3 list of lists, where the first column is indexes'''
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

'''Reads embedding files
   - returns list of lists'''
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


'''Reads edgelists'''
def read_edges(path):
    edges=[]
    with open(path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            u, v = float(tokens[0]),float(tokens[1])

            edges.append((u,v))
    return np.asarray(edges)

'''Loads text document of training loss
   - Takes .txt
   - Returns two lists'''
def load_training_loss(random_path='Data/training_loss.txt', init_path='Data/training_loss_init_embeddings.txt'):
    data_random = []
    with open(random_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            line = [float(x) for x in line]
            data_random.append(line)
    data_random = np.array(data_random)
    for i in range(10000):
        for j in range(1, 11, 1):
            if data_random[i, j] >= 350000:
                data_random[i, j] = min(data_random[i, 1:])

    data_init = []
    with open(init_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            line = [float(x) for x in line]
            data_init.append(line)
    data_init = np.array(data_init)
    for i in range(10000):
        for j in range(1, 11, 1):
            if data_init[i, j] >= 850000:
                data_init[i, j] = min(data_init[i, 1:])

    return data_random, data_init




'''techniques used for determining outliers: https://towardsdatascience.com/outlier-detection-python-cd22e6a12098'''
'''Finds outliers
   - Takes embeddings
   - returns indexes for outliers'''
def find_outliers(data):
    model = DBSCAN(eps=6, min_samples=5).fit(data)
    colors = model.labels_
    outliers_index = (colors == -1).astype(int)

    #check the amount of outliers (-1's) - for tuning the hyperparameters
    unique, counts = np.unique(colors, return_counts=True)
    labels = dict(zip(unique, counts))

    return np.where(outliers_index==1)[0]





'''Following code is used to scale authors by size according to papers released.
   - Takes data in the form of an edge-list.
   - Counts amount of times authors have formed links with papers.
   - Returns a list of edge counts for each author index'''
def author_size(data):
    # Index occurs every time a auhtor has written a paper.
    all_authors = data[:,1] - np.max(data[:,0]+1)

    amount_of_authors = np.zeros(np.max(all_authors).astype(int) + 1)
    for i in range(len(all_authors)):
        amount_of_authors[all_authors[i].astype(int)] += 1
    return amount_of_authors

'''Same for papers and how many times they've been cited.
    - Takes data in the form of an edgelist
    - Counts amount of times papers are cited
    - returns list of citation counts for each paper'''
def paper_size(data):
    # Papers cited by other papers. Index occurs every time a paper is cited.
    #all_paper_citations = np.concatenate((data[:,0],data[:,1]))
    all_paper_citations = data[:,1]

    amount_of_citations = np.zeros(np.max(all_paper_citations).astype(int)+1)
    for i in range(len(all_paper_citations)):
        amount_of_citations[int(all_paper_citations[i])] += 1
    return amount_of_citations


if __name__ == '__main__':

    ldm_paper2paper = np.asarray(read_emb('LDM/ldm_paper2paper.emb'))
    #print(ldm_paper2paper)

    #paper2paper_edgelist = np.asarray(read_emb('Data/paper2paper_edgelist'))
    #print(paper2paper_edgelist)

    #print(find_outliers(ldm_paper2paper))
    print(plot_outliers(ldm_paper2paper))





    # paper2paper_in_network_specific_points('./Data/paper2paper_edgelist', [6, 54, 388])
    #plt.scatter(ldm_paper2paper[:,0], ldm_paper2paper[:,1])
    #plt.show()

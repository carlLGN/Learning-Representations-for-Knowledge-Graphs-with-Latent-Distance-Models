import numpy as np
from Visualization import read_emb
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


ldm_paper2paper = np.asarray(read_emb('LDM/ldm_paper2paper.emb'))
#print(ldm_paper2paper)

#paper2paper_edgelist = np.asarray(read_emb('Data/paper2paper_edgelist'))
#print(paper2paper_edgelist)

# techniques used for determining outliers: https://towardsdatascience.com/outlier-detection-python-cd22e6a12098
def find_outliers(data):
    model = DBSCAN(eps=6, min_samples=5).fit(data)
    colors = model.labels_
    outliers_index = (colors == -1).astype(int)

    #check the amount of outliers (-1's) - for tuning the hyperparameters
    unique, counts = np.unique(colors, return_counts=True)
    labels = dict(zip(unique, counts))

    return np.where(outliers_index==1)[0]


def plot_outliers(data):
    x = data[:, 0]
    y = data[:, 1]

    model = DBSCAN(eps=6, min_samples=5).fit(data)
    colors = model.labels_

    plt.scatter(x, y, c=colors, marker='o')
    plt.show()

#print(find_outliers(ldm_paper2paper))
print(plot_outliers(ldm_paper2paper))

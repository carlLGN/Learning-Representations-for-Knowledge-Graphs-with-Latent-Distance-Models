import numpy as np
import matplotlib.pyplot as plt
from Visualization import read_emb
import matplotlib.cm as cm
import matplotlib as mpl
import torch
from LDM.src.multimodal_ldm import Multimodal_LDM


def plot_joint_embedding(a,p,pstar):
    a = a.detach().numpy()
    p = p.detach().numpy()
    pstar = pstar.detach().numpy()

    fig, ax = plt.subplots()
    ax.scatter(a[:20000,0],a[:20000,1],color='blue',label='Authors',s=0.2)
    ax.scatter(p[:20000,0],p[:20000,1],color='red',label='Cited papers',s=0.2)
    ax.scatter(pstar[:20000,0],pstar[:20000,1],color='orange',label='Citing papers',s=0.2)

    lgnd=plt.legend(loc='upper left')
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    lgnd.legendHandles[2]._sizes = [30]

    plt.title('Joint embedding of authors, cited-, and citing papers for normalized initialization')
    plt.show()

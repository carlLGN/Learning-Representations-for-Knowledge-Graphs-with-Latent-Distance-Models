import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from paper_size import read_emb_general

#data = np.array([np.arange(1,6),np.arange(2,7),np.arange(4,9),np.arange(6,11),np.arange(8,13)])
#data = np.array([[1,4,6,9,24],[2,5,7,17,45],[3,4,25,45,46],[4,8,23,12,16],[5,2,8,16,34]])

def plot_training_loss(data):
    mean_loss = [np.mean(data[i,1:]) for i in range(len(data[:,0]))]
    std_loss = [data[i,1:].std() for i in range(len(data[:,0]))]
    CI = st.t.interval(confidence=0.95, df=len(data[:,0])-1, loc=mean_loss, scale=std_loss)

    fig, ax = plt.subplots()
    ax.plot(data[:,0],mean_loss, color='blue')
    ax.fill_between(data[:,0], CI[0], CI[1], alpha=.4, color='red')
    #ax.set_title('Training loss: Random initialization')
    plt.show()

def plot_training_loss_combined(data_random,data_init):
    mean_loss_random = [np.mean(data_random[i,1:]) for i in range(len(data_random[:,0]))]
    std_loss_random = [data_random[i,1:].std() for i in range(len(data_random[:,0]))]
    CI_random = st.t.interval(confidence=0.95, df=len(data_random[:,0])-1, loc=mean_loss_random, scale=std_loss_random)

    mean_loss_init = [np.mean(data_init[i, 1:]) for i in range(len(data_init[:, 0]))]
    std_loss_init = [data_init[i, 1:].std() for i in range(len(data_init[:, 0]))]
    CI_init = st.t.interval(confidence=0.95, df=len(data_init[:, 0]) - 1, loc=mean_loss_init,
                              scale=std_loss_init)

    fig, ax = plt.subplots()

    ax.plot(data_random[:,0],mean_loss_random, color='blue',label='Random initialization')
    ax.fill_between(data_random[:,0], CI_random[0], CI_random[1], alpha=.4, color='purple',label='CI random initialization')

    ax.plot(data_init[:, 0], mean_loss_init, color='red',label='Spectral relaxation initialization')
    ax.fill_between(data_init[:, 0], CI_init[0], CI_init[1], alpha=.4, color='orange',label='CI spectral relaxation initialization')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    data_random = []
    with open('Data/training_loss.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            line = [float(x) for x in line]
            data_random.append(line)
    data_random = np.array(data_random)
    for i in range(5000):
        for j in range(1, 11, 1):
            if data_random[i, j] >= 800:
                data_random[i, j] = min(data_random[i, 1:])

    data_init = []
    with open('Data/training_loss_init.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            line = [float(x) for x in line]
            data_init.append(line)
    data_init = np.array(data_init)
    for i in range(5000):
        for j in range(1, 11, 1):
            if data_init[i, j] >= 800:
                data_init[i, j] = min(data_init[i, 1:])

    #plot_training_loss(data)
    plot_training_loss_combined(data_random,data_init)

    #print('debug')
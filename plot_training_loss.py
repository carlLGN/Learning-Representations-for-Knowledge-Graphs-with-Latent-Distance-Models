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
    plt.show()


if __name__ == '__main__':
    data = []
    with open('Data/training_loss.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            line = [float(x) for x in line]
            data.append(line)
    data = np.array(data)
    #data[4119][8] = data[4119][7]
    #data[4529][8] = data[4529][7]
    #data[4483][8] = data[4483][7]
    #data = np.delete(data, 8, 1)
    for i in range(5000):
        for j in range(1, 11, 1):
            if data[i, j] >= 300:
                data[i, j] = min(data[i, 1:])


    plot_training_loss(data[:4000])

    print('debug')
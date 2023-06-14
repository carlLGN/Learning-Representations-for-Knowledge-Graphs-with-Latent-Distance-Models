from paper_size import read_emb_general
import numpy as np
import matplotlib.pyplot as plt

#p_star + p + a
data_p_star = read_emb_general('./Embeddings/p_star_init_full.emb')
data_p = read_emb_general('./Embeddings/p_init_full.emb')
data_a = read_emb_general('./Embeddings/a_init_full.emb')

data_merged = np.vstack((data_p_star,data_p,data_a))
#test = data_merged[:100]

#plt.scatter(data_merged[:,0],data_merged[:,1])
#plt.plot(data_merged[:,0],abs(data_merged[:,1]))

fig, axs = plt.subplots(4, 5)
axs[0,0].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,1]),c='blue')
axs[0,0].set_title('Eigenvector 1')
axs[0,1].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,2]),c='blue')
axs[0,1].set_title('Eigenvector 2')
axs[0,2].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,3]),c='blue')
axs[0,2].set_title('Eigenvector 3')
axs[0,3].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,4]),c='blue')
axs[0,3].set_title('Eigenvector 4')
axs[0,4].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,5]),c='blue')
axs[0,4].set_title('Eigenvector 5')
axs[1,0].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,6]),c='blue')
axs[1,0].set_title('Eigenvector 6')
axs[1,1].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,7]),c='blue')
axs[1,1].set_title('Eigenvector 7')
axs[1,2].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,8]),c='blue')
axs[1,2].set_title('Eigenvector 8')
axs[1,3].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,9]),c='blue')
axs[1,3].set_title('Eigenvector 9')
axs[1,4].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,10]),c='blue')
axs[1,4].set_title('Eigenvector 10')
axs[2,0].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,11]),c='blue')
axs[2,0].set_title('Eigenvector 11')
axs[2,1].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,12]),c='blue')
axs[2,1].set_title('Eigenvector 12')
axs[2,2].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,13]),c='blue')
axs[2,2].set_title('Eigenvector 13')
axs[2,3].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,14]),c='blue')
axs[2,3].set_title('Eigenvector 14')
axs[2,4].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,15]),c='blue')
axs[2,4].set_title('Eigenvector 15')
axs[3,0].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,16]),c='blue')
axs[3,0].set_title('Eigenvector 16')
axs[3,1].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,17]),c='blue')
axs[3,1].set_title('Eigenvector 17')
axs[3,2].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,18]),c='blue')
axs[3,2].set_title('Eigenvector 18')
axs[3,3].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,19]),c='blue')
axs[3,3].set_title('Eigenvector 19')
axs[3,4].plot(np.asarray(data_merged)[:,0],abs(np.asarray(data_merged)[:,20]),c='blue')
axs[3,4].set_title('Eigenvector 20')


#plt.plot(np.asarray(data_a)[:,0],abs(np.asarray(data_a)[:,i]),c='blue')
#plt.title('Eigenvector 1')
#plt.hist(data_merged_one_list[1])
plt.show()


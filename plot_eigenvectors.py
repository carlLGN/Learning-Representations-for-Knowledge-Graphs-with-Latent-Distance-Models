from paper_size import read_emb_general
import numpy as np
import matplotlib.pyplot as plt

#p_star + p + a
data_p_star = read_emb_general('./Embeddings/p_star_init_p2p.emb')
data_p = read_emb_general('./Embeddings/p_init_p2p.emb')
data_a = read_emb_general('./Embeddings/a_init_p2p.emb')

data_merged = np.vstack((data_p_star,data_p,data_a))
#data_merged = read_emb_general('./Embeddings/eigenvectors_L_p2p.emb')
eigenvalues_p2p = np.array([10.87016213,6.56190205,3.32007754,1.08764416,0.06888143])
eigenvalues_Lsym_p2p = np.array([5.61192289e-03,4.56078461e-03,3.852744514e-03,3.62295623e-03,2.22044605e-15])
eigenvalues_a2p = np.array([10.44176416,6.30759595,3.18412723,1.06012389,0.06225148])

#test = data_merged[:100]

#plt.scatter(data_merged[:,0],data_merged[:,1])
#plt.plot(data_merged[:,0],abs(data_merged[:,1]))

fig, axs = plt.subplots(2, 3)
axs[0,0].plot(np.asarray(data_merged)[:,0],np.asarray(data_merged)[:,-1],c='blue')
axs[0,0].set_title('Eigenvector 1')
axs[0,1].plot(np.asarray(data_merged)[:,0],np.asarray(data_merged)[:,-2],c='blue')
axs[0,1].set_title('Eigenvector 2')
axs[0,2].plot(np.asarray(data_merged)[:,0],np.asarray(data_merged)[:,-3],c='blue')
axs[0,2].set_title('Eigenvector 3')
axs[1,0].plot(np.asarray(data_merged)[:,0],np.asarray(data_merged)[:,-4],c='blue')
axs[1,0].set_title('Eigenvector 4')
axs[1,1].plot(np.asarray(data_merged)[:,0],np.asarray(data_merged)[:,-5],c='blue')
axs[1,1].set_title('Eigenvector 5')
axs[1,2].scatter(np.arange(len(eigenvalues_Lsym_p2p))+1,eigenvalues_Lsym_p2p[::-1],c='blue')
axs[1,2].set_title('Eigenvalues')



#plt.plot(np.asarray(data_a)[:,0],abs(np.asarray(data_a)[:,i]),c='blue')
#plt.title('Eigenvector 1')
#plt.hist(data_merged_one_list[1])
plt.show()


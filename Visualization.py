import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from visualization_utils import read_edges, read_emb, author_size, paper_size, load_training_loss, read_emb_general, read_emb3
from sklearn.cluster import DBSCAN
import matplotlib as mpl


'''Simple function that plots embeddings
   - takes matrix of embeddings'''
def visualize(embs):

    # Dimension must be 2
    assert embs.shape[1] == 2, "The dimension of embeddings must be 2"

    plt.figure()
    plt.scatter(embs[:, 0], embs[:, 1], s=1)
    plt.show()


'''Function that plots edges between points
   - Takes path to edgelist'''
def paper2paper_in_network(path):
    edges=read_edges(path)

    ldm_paper2paper = np.asarray(read_emb('./Embeddings/a_init.emb'))

    xval=edges[:,0]
    yval=edges[:,1]

    for i in range(len(xval)):
        xvalues=ldm_paper2paper[int(xval[i])]
        yvalues=ldm_paper2paper[int(yval[i])]

        plt.plot(xvalues,yvalues)

    plt.show()


'''Function that plots specific points in network
   - Takes edgelist path
   - Takes list of nodes to consider'''
def paper2paper_in_network_specific_points(path, nodes_looked_at):

    ldm_paper2paper = np.asarray(read_emb('./Embeddings/a_init.emb'))

    edges=read_edges(path)

    xval=edges[:,0]
    yval=edges[:,1]

    plt.scatter(ldm_paper2paper[:,0], ldm_paper2paper[:,1], alpha=0.1)

    for i in range(len(xval)):
        if xval[i] in nodes_looked_at or yval[i] in nodes_looked_at:
            xvalues=ldm_paper2paper[int(xval[i])]
            yvalues=ldm_paper2paper[int(yval[i])]

            plt.plot(xvalues,yvalues, color='red', alpha=0.6)
            plt.scatter(xvalues, yvalues, alpha=0.6, color='red')

        else:
            pass

    plt.show()


'''Following visualizations plot embeddings for individual parts with fancy scaling technology'''
def plot_author_size(path):
    data = np.asarray(read_emb_general(path))

    x = data[:, -2]
    y = data[:, -3]

    s = author_size(author2paper_edgelist).astype(float)*2
    color = 1/np.sum(s)*s
    plt.scatter(x, y, s=s, c=color, alpha=.8,cmap=mpl.colormaps['winter_r'],edgecolors='black', linewidth=0.3)
    plt.colorbar()
    plt.show()

def plot_paper_size(path):
    data = np.asarray(read_emb_general(path))

    x = data[:, -2]
    y = data[:, -3]

    s = paper_size(paper2paper_edgelist)[:100000].astype(float)*2
    color = 1/np.sum(s)*s
    plt.scatter(x, y, s=s, c=color, alpha=.8,cmap=mpl.colormaps['winter_r'],edgecolors='black', linewidth=0.3)
    plt.colorbar()
    plt.show()

'''Function also plots all embeddings, but a little cleaner
   - Takes all 3 embeddings as np.arrays'''
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


'''Finds outliers and plots them. 
   - Takes embeddings
   - Selects outliers and plots'''
def plot_outliers(data):
    x = data[:, 0]
    y = data[:, 1]

    model = DBSCAN(eps=6, min_samples=5).fit(data)
    colors = model.labels_

    plt.scatter(x, y, c=colors, marker='o')
    plt.show()



'''The following function plots training loss from a text file output of the LDM model'''
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


'''The following three functions plot the entirety of the embeddings together'''

def all_together():
    ldm_citingpapers = np.asarray(read_emb('./Embeddings/p_init_p2p.emb'))
    ldm_citedpapers = np.asarray(read_emb('./Embeddings/p_star_init_p2p.emb'))
    ldm_authors = np.asarray(read_emb('./Embeddings/a_init_p2p.emb'))

    plt.scatter(ldm_citingpapers[:, 0], ldm_citingpapers[:, 1], alpha=0.5, color='red')

    plt.scatter(ldm_citedpapers[:, 0], ldm_citedpapers[:, 1], alpha=0.5, color='blue')

    plt.scatter(ldm_authors[:, 0], ldm_authors[:, 1], alpha=0.5, color='green')

    plt.show()


def all_together_nodes(nodes):
    ldm_citingpapers = np.asarray(read_emb('./Embeddings/p_init_p2p.emb'))
    ldm_citedpapers = np.asarray(read_emb('./Embeddings/p_star_init_p2p.emb'))
    ldm_authors = np.asarray(read_emb('./Embeddings/a_init_p2p.emb'))
    for i in range(len(nodes)):
        i = nodes[i]

        plt.scatter(ldm_citingpapers[i, :][0], ldm_citingpapers[i, :][1], alpha=0.5, color='red')
        plt.scatter(ldm_citedpapers[i, :][0], ldm_citedpapers[i, :][1], alpha=0.5, color='blue')
        plt.scatter(ldm_authors[i, :][0], ldm_authors[i, :][1], alpha=0.5, color='green')

    plt.show()


def all_together_edges(nodes):
    ldm_citingpapers = np.asarray(read_emb('./Embeddings/p_init_p2p.emb'))
    ldm_citedpapers = np.asarray(read_emb('./Embeddings/p_star_init_p2p.emb'))
    ldm_authors = np.asarray(read_emb('./Embeddings/a_init_p2p.emb'))

    a2p = read_edges('./Data/paper2paper_edgelist')
    p2p = read_edges('./Data/author2paper_edgelist')

    citing1 = a2p[:, 0]
    author = a2p[:, 1]

    citing2 = p2p[:, 0]
    cited = p2p[:, 1]

    for j in range(len(citing1)):

        # nodes are thus refering to citing papers
        if citing1[j] in nodes:
            citingnode1 = ldm_citingpapers[int(citing1[j])]
            authornode = ldm_authors[int(author[j])]

            plt.plot(citingnode1[0], citingnode1[1], marker='o', color='red', alpha=0.5)
            plt.plot(authornode[0], authornode[1], marker='o', color='green', alpha=0.5)
            plt.plot([citingnode1[0], authornode[0]], [citingnode1[1], authornode[1]], linestyle='-', alpha=0.5,
                     color='green')

    for k in range(len(citing2)):

        if citing2[k] in nodes:
            citingnode2 = ldm_citingpapers[int(citing2[k])]
            citednode = ldm_citedpapers[int(cited[k])]

            plt.plot(citingnode2[0], citingnode2[1], marker='o', color='red', alpha=0.5)
            plt.plot(citednode[0], citednode[1], marker='o', color='blue', alpha=0.5)
            plt.plot([citingnode2[0], citednode[0]], [citingnode2[1], citednode[1]], linestyle='-', alpha=0.5,
                     color='blue')

    plt.show()

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




'''Specific function to visualize the 5 smallest eigenvectors of the laplacian / symmetric laplacian
   as well as the eigenvalues. Was never generalized as it was not important.
   The eigenvectors and eigenvalues came in reverse order, hence the indexing.'''
def plot_eigenvectors():
    # p_star + p + a
    data_p_star = read_emb_general('./Embeddings/p_star_init_p2p.emb')
    data_p = read_emb_general('./Embeddings/p_init_p2p.emb')
    data_a = read_emb_general('./Embeddings/a_init_p2p.emb')

    data_merged = np.vstack((data_p_star, data_p, data_a))
    # data_merged = read_emb_general('./Embeddings/eigenvectors_L_p2p.emb')
    eigenvalues_p2p = np.array([10.87016213, 6.56190205, 3.32007754, 1.08764416, 0.06888143])
    eigenvalues_Lsym_p2p = np.array([5.61192289e-03, 4.56078461e-03, 3.852744514e-03, 3.62295623e-03, 2.22044605e-15])
    eigenvalues_a2p = np.array([10.44176416, 6.30759595, 3.18412723, 1.06012389, 0.06225148])


    fig, axs = plt.subplots(2, 3)
    axs[0, 0].scatter(np.arange(len(data_merged)), np.asarray(data_merged)[:, -1], c='blue', s=0.1)
    axs[0, 0].set_title('Eigenvector 1')
    axs[0, 1].scatter(np.arange(len(data_merged)), np.asarray(data_merged)[:, -2], c='blue', s=0.1)
    axs[0, 1].set_title('Eigenvector 2')
    axs[0, 2].scatter(np.arange(len(data_merged)), np.asarray(data_merged)[:, -3], c='blue', s=0.1)
    axs[0, 2].set_title('Eigenvector 3')
    axs[1, 0].scatter(np.arange(len(data_merged)), np.asarray(data_merged)[:, -4], c='blue', s=0.1)
    axs[1, 0].set_title('Eigenvector 4')
    axs[1, 1].scatter(np.arange(len(data_merged)), np.asarray(data_merged)[:, -5], c='blue', s=0.1)
    axs[1, 1].set_title('Eigenvector 5')
    axs[1, 2].scatter(np.arange(len(eigenvalues_p2p)) + 1, eigenvalues_Lsym_p2p[::-1], c='blue')
    axs[1, 2].set_title('Eigenvalues')

    plt.show()

'''Function that plots ROC_AUC and PR_AUC with confidence intervals for different multimodal ldms 
   at different values of alpha
   - Takes concatenated data for ROC and PR (see bottom of script)'''
def plot_AUC_combined(ROC,PR):
    mean_ROC= [np.mean(ROC[i,:]) for i in range(len(ROC))]
    std_ROC = [ROC[i,:].std() for i in range(len(ROC))]
    CI_ROC = st.t.interval(confidence=0.95, df=len(ROC[0,:])-1, loc=mean_ROC, scale=std_ROC)

    mean_PR = [np.mean(PR[i, :]) for i in range(len(PR))]
    std_PR = [PR[i, :].std() for i in range(len(PR))]
    CI_PR = st.t.interval(confidence=0.95, df=len(PR[0, :]) - 1, loc=mean_PR, scale=std_PR)

    fig, ax = plt.subplots()

    ax.plot(np.arange(0,1.1,0.1),mean_ROC, color='blue',label='ROC')
    ax.fill_between(np.arange(0,1.1,0.1), CI_ROC[0], CI_ROC[1], alpha=.4, color='purple',label='CI ROC')

    ax.plot(np.arange(0, 1.1, 0.1), mean_PR, color='red', label='PR')
    ax.fill_between(np.arange(0, 1.1, 0.1), CI_PR[0], CI_PR[1], alpha=.4, color='orange', label='CI PR')
    plt.legend(loc='upper left')
    plt.show()



'''Simple function that plots roc_auc and pr_auc as a function of epochs. Not particularly general,
   but data was generated through multimodal_ldm models'''
def auc_over_epoch():
    # chose number of epochs
    epoch_numbers = np.arange(1000, 16000, 1000)
    roc_auc_test = np.array(
        [0.29034500799077234, 0.3847267268160957, 0.45123621825146243, 0.4957287905883826, 0.5291590522696532,
         0.5522746610252844, 0.5621663525959977, 0.5638053308053357, 0.5571976716959265, 0.5440252220569973,
         0.5266243187962738, 0.5107797364965616, 0.49861772617885686, 0.49533809253012584, 0.49403783286827063])
    roc_auc_train = np.array(
        [0.185217220463138, 0.34861175992438564, 0.45930773842155015, 0.5329761637523629, 0.5928756767485823,
         0.6326168175803403, 0.6530295368620037, 0.6638204529773156, 0.6655197334593572, 0.656055768194707,
         0.6447879853497165, 0.6304135590737241, 0.6150738060018903, 0.6143516935255199, 0.6147049068998109])
    pr_auc_test = np.array(
        [0.380564518725965, 0.413444990685923, 0.4486045565876781, 0.48000615450154455, 0.5072291675827374,
         0.5294440479184629, 0.5420511387991905, 0.548363020623874, 0.5444162306694073, 0.5302148128162326,
         0.5134105600939998, 0.49892380163745925, 0.48542226897777135, 0.4787742668396906, 0.47640511439324096])
    pr_auc_train = np.array(
        [0.33960763018487056, 0.3910294806118196, 0.4462588034786169, 0.49750917260817135, 0.5505871291947154,
         0.5954012265186666, 0.624373248576923, 0.6414562582706796, 0.6405204680610089, 0.6206449582949245,
         0.6008215843114599, 0.5781007533400856, 0.5575559087998063, 0.5499739093014275, 0.5482431485861213])

    fig, ax = plt.subplots()
    ax.plot(epoch_numbers, pr_auc_test, color='blue', label='Test set')
    ax.plot(epoch_numbers, pr_auc_train, color='red', label='Training set')
    plt.legend(loc='lower right')
    plt.title('PR-AUC')
    plt.show()

if __name__ == '__main__':

    '''Plot authors scaled by amount of papers'''
    author2paper_edgelist = np.asarray(read_emb3('Data/train_edgelist_ap'))
    # print(author2paper_edgelist)

    paper2paper_edgelist = np.asarray(read_emb3('Data/paper2paper_edgelist'))
    # print(paper2paper_edgelist)

    #print(plot_paper_size('./Embeddings/p_init_p2p.emb'))
    print(plot_author_size('./Embeddings/a_init_p2p.emb'))


    #Plot training loss convergence
    plot_training_loss_combined(load_training_loss())


    '''The following is the data for plotting ROC-AUC and PR-AUC. It was generated using multimodal ldm models
       at different values of alpha'''
    alpha0_ROC = np.array([0.5139942504180098, 0.5129974611502941, 0.5126721524022754, 0.5104210402738061])
    alpha0_PR = np.array([0.5100390276908593, 0.5099537342252727, 0.5095157928941763, 0.5075276938967743])

    alpha01_ROC = np.array([0.5392807594222249, 0.5366336654243505, 0.5392426651938667, 0.5380915588683567])
    alpha01_PR = np.array([0.5021775478557212, 0.4993998856635743, 0.5030544973621494, 0.5010069555408898])

    alpha02_ROC = np.array([0.5388720377568939, 0.5372465995338935, 0.5393936636201688, 0.5379187939129974])
    alpha02_PR = np.array([0.5014847441212367, 0.499359731352131, 0.5022005886789667, 0.5003596766858132])

    alpha03_ROC = np.array([0.5396144682973034, 0.5385998690636926, 0.5401718211077174, 0.5389202644666113])
    alpha03_PR = np.array([0.5012263915962345, 0.4997036198790358, 0.5019039105028505, 0.5003858536680583])

    alpha04_ROC = np.array([0.5415652181575057, 0.5398345358859973, 0.5407575607217756, 0.5400115711698242])
    alpha04_PR = np.array([0.5015050614121187, 0.499742667215619, 0.5010616397389794, 0.4998290288865371])

    alpha05_ROC = np.array([0.5433431701256592, 0.5407354909440404, 0.5423915148703019, 0.542072439477444])
    alpha05_PR = np.array([0.5021168765387254, 0.4995653981665293, 0.5013374478071241, 0.5003995136850902])

    alpha06_ROC = np.array([0.5452066488531859, 0.544226874332557, 0.5413606012276335, 0.5437195695730055])
    alpha06_PR = np.array([0.5019894687983361, 0.502769272545439, 0.4991299267581509, 0.5017529503341484])

    alpha07_ROC = np.array([0.5466371720237961, 0.5463159588664396, 0.543913899507996, 0.5448708821862023])
    alpha07_PR = np.array([0.5022988440091156, 0.5035365915191942, 0.5001767035260741, 0.5016427786403401])

    alpha08_ROC = np.array([0.5500307229279107, 0.5495923874805766, 0.5473200790322637, 0.5475481430713284])
    alpha08_PR = np.array([0.5041376452392207, 0.5051129974149426, 0.5023002794223805, 0.5030205661922255])

    alpha09_ROC = np.array([0.5503401700596766, 0.5549406758956679, 0.5566398698830557, 0.5530312391607088])
    alpha09_PR = np.array([0.503728154626985, 0.5068414989620861, 0.508509936956099, 0.5068895232028332])

    alpha1_ROC = np.array([0.5593023754779938, 0.5626639631162553, 0.5630675821371972, 0.5622581534009768])
    alpha1_PR = np.array([0.508916482762843, 0.5115968515338944, 0.512051142903845, 0.512020086549525])
    ROC = np.array(
        [alpha0_ROC, alpha01_ROC, alpha02_ROC, alpha03_ROC, alpha04_ROC, alpha05_ROC, alpha06_ROC, alpha07_ROC,
         alpha08_ROC, alpha09_ROC, alpha1_ROC])
    PR = np.array(
        [alpha0_PR, alpha01_PR, alpha02_PR, alpha03_PR, alpha04_PR, alpha05_PR, alpha06_PR, alpha07_PR, alpha08_PR,
         alpha09_PR, alpha1_PR])

    '''do the plot'''
    plot_AUC_combined(ROC, PR)
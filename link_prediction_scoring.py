import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import random
from LDM.src.ldm import LDM
from LDM.src.multimodal_ldm import Multimodal_LDM


if __name__=='__main__':

    def read_edge(path):
        points=[]
        with open(path, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split()
                s, u = float(tokens[0]),float(tokens[1])

                points.append((s,u))
        return points


#skal også importere rigtig ap edgelist måske

    device = torch.device('cpu')
    args = ['./Data/train_edgelist_pp', './Data/train_edgelist_ap', './ldm_paper2papertest2.emb', 2, 5000, 1, 0.5, 500, 0.1, 19, 1, 0]
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

    model = Multimodal_LDM(edges_pp = edges_pp, edges_ap=edges_ap, dim=dim, lr=lr, epoch_num=epoch_num, batch_size=batch_size, spe=steps_per_epoch, device=torch.device(device), verbose=verbose, seed=seed)
    model.load_state_dict(torch.load(r'./Embeddings/test_author2paper_0.0_15000', map_location = device))









    lenpapers=len(model.p_star.detach().numpy()[:,0])
    print('works1')

    #hvis det skal køres på test
    # pos_samples=[]
    # with open('./Data/test_edgelist_ap', 'r') as f:
    #         for line in f.readlines():
    #             tokens = line.strip().split()
    #             s, u= float(tokens[0]),float(tokens[1])
    #             pos_samples.append((s,u))
    # print('works2')
    # train_edgelist=set(read_edge('Data/train_edgelist_ap'))
  
    #hvis det skal køres på train

    pos_samples=[]
    with open('./Data/train_edgelist_ap', 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split()
                s, u= float(tokens[0]),float(tokens[1])
                pos_samples.append((s,u))

    pos_samples=pos_samples[0:46000]
    train_edgelist=set(read_edge('Data/train_edgelist_ap'))

    print('works3')
    neg_samples=set()
    z=0
    print('startingloop')
    while z<len(pos_samples):
        a=random.randint(0,lenpapers-2)
        b=random.randint(a+1,lenpapers-1)
        edge=(a,b)
        if edge in train_edgelist or edge in neg_samples:
            continue
        else:
            neg_samples.add(edge)
            if len(neg_samples) % 1000 == 0:
                print(len(neg_samples))
            z+=1


    neg_samples=list(neg_samples)

    labels = [1]*len(pos_samples) + [0]*len(pos_samples)
    samples = pos_samples + neg_samples
    print('works4')

    #open and read beta, embeddingsp and embeddingspstar
    #kun for samples, read som array


    pdist = torch.nn.PairwiseDistance(p=2)


    def get_intensity_for(i, j):
        beta_sum = model.beta_ap.detach().numpy()[int(i)] + model.beta_ap.detach().numpy()[int(j)]
        z_dist=pdist(model.p_star.detach()[int(i),:], model.a.detach()[int(j-lenpapers),:])
        return torch.exp(beta_sum - z_dist)


    pred_scores = []
    for sample in samples:
        #calculate intensity of edge, så input skal være node1, node2. get intensity skal være den rigtige - tjek dette.
        pred_scores.append(
            #lm.get_intensity_for(i=sample[0], j=sample[1]).detach().numpy()
            get_intensity_for(i=sample[0], j=sample[1]).detach().numpy()
        )

    pred_median=np.median(pred_scores)
    pred_binary_scores = pred_scores.copy()
    for i in range(len(pred_scores)):
        if pred_scores[i] >= pred_median:
            pred_binary_scores[i] = 1
        else:
            pred_binary_scores[i] = 0


    #bare skriv output_path?
    with open(emb_path, 'w') as f:
        roc_auc = roc_auc_score(y_true=labels, y_score=pred_scores)
        f.write(f"Roc_AUC: {roc_auc}\n")
        pr_auc = average_precision_score(y_true=labels, y_score=pred_scores)
        f.write(f"PR_AUC: {pr_auc}\n")

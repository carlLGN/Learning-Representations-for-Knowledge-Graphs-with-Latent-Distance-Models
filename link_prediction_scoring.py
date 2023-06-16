import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import random
from LDM.src.multimodal_ldm import Multimodal_LDM

def read_edge(path):
    points=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            s, u = float(tokens[0]),float(tokens[1])

            points.append((s,u))
    return points



device = torch.device('cpu')
args = ['./Data/train_edgelist_pp', './Data/author2paper_edgelist', './ldm_paper2papertest.emb', 2, 200, 1, 0.5, 500, 0.1, 19, 1]
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

edges_pp = read_edge(dataset_pp_path)
edges_pp = torch.as_tensor(edges_pp, dtype=torch.int, device=torch.device("cpu")).T

edges_ap = read_edge(dataset_ap_path)
edges_ap = torch.as_tensor(edges_ap, dtype=torch.int, device=torch.device("cpu")).T

model = Multimodal_LDM(edges_pp=edges_pp, edges_ap=edges_ap, dim=dim, lr=lr, epoch_num=epoch_num, batch_size=batch_size, spe=steps_per_epoch, alpha=alpha, device=torch.device(device), verbose=verbose, seed=seed)
model.load_state_dict(torch.load(r'test', map_location = device))









lenpapers=len(model.p_star.detach().numpy()[:,0])
print('works1')

pos_samples=[]
with open('./Data/test_edgelist_pp', 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            s, u= float(tokens[0]),float(tokens[1])
            pos_samples.append((s,u))
print('works2')
train_edgelist=set(read_edge('Data/train_edgelist_pp'))
#train_edgelist=set(train_edgelist)

#pos_samples=np.asarray(pos_samples)
print('works3')
#det her tager lidt tid, så overvej at køre en gang i anden fil og læs.
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
    beta_sum = model.gamma_pp.detach().numpy()[int(i)] + model.gamma_pp.detach().numpy()[int(j+lenpapers)]
    z_dist=pdist(model.p_star.detach()[int(i),:], model.p.detach()[int(j),:])
    return torch.exp(beta_sum - z_dist)


pred_scores = []
for sample in samples:
    #calculate intensity of edge, så input skal være node1, node2. get intensity skal være den rigtige - tjek dette.
    pred_scores.append(
        #lm.get_intensity_for(i=sample[0], j=sample[1]).detach().numpy()
        get_intensity_for(i=sample[0], j=sample[1]).detach().numpy()
    )

#bare skriv output_path?
with open(output_path, 'w') as f:
    roc_auc = roc_auc_score(y_true=labels, y_score=pred_scores)
    f.write(f"Roc_AUC: {roc_auc}\n")
    pr_auc = average_precision_score(y_true=labels, y_score=pred_scores)
    f.write(f"PR_AUC: {pr_auc}\n")

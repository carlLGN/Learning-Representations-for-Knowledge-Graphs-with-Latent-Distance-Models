import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


pos_samples=[]
with open('./Data/test_edgelist_pp', 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            s, u= float(tokens[0]),float(tokens[1])
            pos_samples.append((s,u))

neg_samples=[]
#lav edgelist på ikke eksistrende edges
with open('./Data/fake_edges', 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            s, u= float(tokens[0]),float(tokens[1])
            neg_samples.append((s,u))


labels = [1]*len(pos_samples) + [0]*len(neg_samples)
samples = pos_samples + neg_samples


#open and read beta, embeddingsp and embeddingspstar
#kun for samples, read som array
beta=[]
with open('./Data/beta', 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            s= float(tokens[0])
            beta.append((s))

embeddingsp=[]
with open('./Data/embeddingsp', 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            s,u = float(tokens[0]), float(tokens[1])
            embeddingsp.append((s,u))

embeddingspstar=[]
with open('./Data/embeddingspstar', 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            s,u = float(tokens[0]), float(tokens[1])
            embeddingspstar.append((s,u))

pos_samples=np.asarray(pos_samples)
neg_samples=np.asarray(neg_samples)
beta=np.asarray(beta)
embeddingsp=np.asarray(embeddingsp)
embeddingspstar=np.asarray(embeddingspstar)


pdist = torch.nn.PairwiseDistance(p=2)

lenpapers=len(embeddingsp[:,0])

def get_intensity_for(i, j):
    beta_sum = beta[i] + beta[j+lenpapers]
    z_dist=pdist(embeddingsp[i,:], embeddingspstar[j,:])
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

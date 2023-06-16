import torch
from multimodal_ldm import Multimodal_LDM
import utils

device = torch.device('cpu')
base_folder = 'C:/Users/carl/Documents/DTU/Fjerde_semester/Fag_Projekt/Projekt/Learning-Representations-for-Knowledge-Graphs-with-Latent-Distance-Models'
args = [r'C:/Users/carl/Documents/DTU/Fjerde_semester/Fag_Projekt/Projekt/Learning-Representations-for-Knowledge-Graphs-with-Latent-Distance-Models/train_edgelist_pp', base_folder+'/Data/author2paper_edgelist', './ldm_paper2papertest.emb', 2, 200, 1, 0.5, 500, 0.1, 19, 1]
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

edges_pp = utils.read_emb(dataset_pp_path)
edges_pp = torch.as_tensor(edges_pp, dtype=torch.int, device=torch.device("cpu")).T

edges_ap = utils.read_emb(dataset_ap_path)
edges_ap = torch.as_tensor(edges_ap, dtype=torch.int, device=torch.device("cpu")).T

model = Multimodal_LDM(edges_pp=edges_pp, edges_ap=edges_ap, dim=dim, lr=lr, epoch_num=epoch_num, batch_size=batch_size, spe=steps_per_epoch, alpha=alpha, device=torch.device(device), verbose=verbose, seed=seed)
model.load_state_dict(torch.load(r'C:/Users/carl/Documents/DTU/Fjerde_semester/Fag_Projekt/Projekt/Learning-Representations-for-Knowledge-Graphs-with-Latent-Distance-Models/Transfer/test', map_location = device))


print('debug')
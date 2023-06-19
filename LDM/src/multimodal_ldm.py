#Multimodal LDM model - Carl Nørlund, Malthe Bresler, Sara Andersen & Jacob Corkill
# @2023

import torch
from torch_sparse import spspmm
import LDM.src.utils as utils
import math
import time
import sys
import random



class Multimodal_LDM(torch.nn.Module):
    def __init__(self,  edges_pp, edges_ap, dim, lr=0.1, epoch_num=100, batch_size = 0, spe=1, alpha=0.5, device=torch.device("cpu"),
                 verbose=False, seed=0):
        super(Multimodal_LDM, self).__init__()

        #Edges
        self.edges_pp = edges_pp.to(device)[[1,0], :]
        self.edges_ap = edges_ap.to(device)[[1,0], :]
        
        #Nodes
        self.nodes_num_pp = torch.max(self.edges_pp) + 1
        self.nodes_num_ap = torch.max(self.edges_ap) + 1
        
        #Alpha
        self.alpha = alpha
        
        #Number nodes and dimension of embedding
        self.edges_num_pp = self.edges_pp.shape[1]
        self.edges_num_ap = self.edges_ap.shape[1]
        self.dim = dim
        
        #Sampling weights
        self.sampling_weights = torch.ones(self.nodes_num_pp, dtype=torch.float, device=device)
        
        self.training_loss = []
        self.device = device
        self.seed = seed
        self.verbose = verbose

        # Set the seed
        self.__set_seed(self.seed)

        #Embedding files
        #emb_file = ["/zhome/c2/3/167669/GRL/Embeddings/p_star_init.emb", "/zhome/c2/3/167669/GRL/Embeddings/p_init.emb", "/zhome/c2/3/167669/GRL/Embeddings/a_init.emb"]
        
        #Get initial embeddings
        #p_star, p, a = utils.read_embeddings(emb_file)
        
        # Initialize the parameters
        self.beta_ap = torch.nn.Parameter(
            2 * torch.rand(size=(self.nodes_num_ap,), device=self.device) - 1, requires_grad=True
        )
        self.gamma_pp = torch.nn.Parameter(
            2 * torch.rand(size=(2*self.nodes_num_pp,), device=self.device) - 1, requires_grad=True
        )
        self.p = torch.nn.Parameter(
            2 * torch.rand(size=(self.nodes_num_pp, self.dim), device=self.device) - 1, requires_grad=True
        )
        self.p_star = torch.nn.Parameter(
            2 * torch.rand(size=(self.nodes_num_pp, self.dim), device=self.device) - 1, requires_grad=True
        )
        self.a = torch.nn.Parameter(
            2 * torch.rand(size=(self.nodes_num_ap-self.nodes_num_pp, self.dim), device=self.device) - 1, requires_grad=True
        )

        
        
        '''
        self.p = torch.nn.Parameter(
            torch.as_tensor(p), requires_grad=True
        )
        
        self.p_star = torch.nn.Parameter(
            torch.as_tensor(p_star), requires_grad=True
        )
        self.a = torch.nn.Parameter(
            torch.as_tensor(a), requires_grad=True
        )
        '''
        self.epoch_num = epoch_num
        self.steps_per_epoch = spe
        self.batch_size = batch_size if batch_size else self.nodes_num_pp
        self.learning_rate = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss = []

        self.pdist = torch.nn.PairwiseDistance(p=2)

    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)

    def get_embs(self):

        return self.z

    def get_intensity_sum_ap(self, p_star_nodes=None, a_nodes = None):

        beta_p_star = self.beta_ap if p_star_nodes is None else torch.index_select(self.beta_ap, index=p_star_nodes, dim=0)
        beta_a = self.beta_ap if a_nodes is None else torch.index_select(self.beta_ap, index=a_nodes, dim=0)
        
        a = self.p if a_nodes is None else torch.index_select(self.a, index=a_nodes-self.nodes_num_pp, dim=0)
        p_star = self.p_star if p_star_nodes is None else torch.index_select(self.p_star, index=p_star_nodes, dim=0)        

        beta_mat = beta_p_star.unsqueeze(0) + beta_a.unsqueeze(1)
        dist_mat = torch.cdist(p_star, a, p=2)

        return torch.exp(beta_mat.T - dist_mat).sum()
    
    def get_intensity_sum_pp(self, p_star_nodes=None, p_nodes=None):
        
        gamma_p_star = self.gamma_pp if p_star_nodes is None else torch.index_select(self.gamma_pp, index=p_star_nodes, dim=0)
        gamma_p = self.gamma_pp if p_nodes is None else torch.index_select(self.gamma_pp, index=p_nodes+self.nodes_num_pp, dim=0)
        
        p = self.p if p_nodes is None else torch.index_select(self.p, index=p_nodes, dim=0)
        p_star = self.p_star if p_star_nodes is None else torch.index_select(self.p_star, index=p_star_nodes, dim=0)
        
        gamma_mat = gamma_p_star.unsqueeze(0) + gamma_p.unsqueeze(1)
        dist_mat = torch.cdist(p_star, p, p=2)

        return torch.triu(torch.exp(gamma_mat.T - dist_mat), diagonal=1).sum()

    def get_log_intensity_sum_ap(self, edges):

        beta_pair = torch.index_select(self.beta_ap, index=edges[0], dim=0) + \
                    torch.index_select(self.beta_ap, index=edges[1], dim=0)

        #Care for indexes - paper index must go from 0 to n and author indexes from n to n+m
        z_dist = self.pdist(
            torch.index_select(self.p_star, index=edges[0], dim=0),
            torch.index_select(self.a, index=edges[1]-self.nodes_num_pp, dim=0),
        )

        return (beta_pair - z_dist).sum()
    
    def get_log_intensity_sum_pp(self, edges):

        beta_pair = torch.index_select(self.gamma_pp, index=edges[0], dim=0) + \
                    torch.index_select(self.gamma_pp, index=edges[1]+self.nodes_num_pp, dim=0)

        z_dist = self.pdist(
            torch.index_select(self.p_star, index=edges[0], dim=0),
            torch.index_select(self.p, index=edges[1], dim=0),
        )

        return (beta_pair - z_dist).sum()


    def get_neg_likelihood_pp(self, edges, p_star_nodes=None, p_nodes=None):

        # Compute the link term
        link_term = self.get_log_intensity_sum_pp(edges=edges)

        # Compute the non-link term
        non_link = self.get_intensity_sum_pp(p_star_nodes=p_star_nodes, p_nodes = p_nodes)

        return -(link_term - non_link) 
    
    
    def get_neg_likelihood_ap(self, edges, p_star_nodes=None, a_nodes=None):

        # Compute the link term
        link_term = self.get_log_intensity_sum_ap(edges=edges)

        # Compute the non-link term
        non_link = self.get_intensity_sum_ap(p_star_nodes=p_star_nodes, a_nodes = a_nodes)

        return -(link_term - non_link)

    def learn(self):

        for epoch in range(self.epoch_num):

            self.__train_one_epoch(current_epoch=epoch)

        return self.loss

    def __train_one_epoch(self, current_epoch):

        init_time = time.time()

        total_batch_loss = 0
        self.loss.append([])
        for batch_num in range(self.steps_per_epoch):
            batch_loss = self.__train_one_batch()

            self.loss[-1].append(batch_loss)

            total_batch_loss += batch_loss

            # Set the gradients to 0
            self.optimizer.zero_grad()

            # Backward pass
            batch_loss.backward()

            # Perform a step
            self.optimizer.step()

        # Get the average epoch loss
        epoch_loss = total_batch_loss / float(self.steps_per_epoch)

        if not math.isfinite(epoch_loss):
            print(f"Epoch loss is {epoch_loss}, stopping training")
            sys.exit(1)

        if self.verbose and (current_epoch % 1 == 0 or current_epoch == self.epoch_num - 1):
            print(f"| Epoch = {current_epoch} | Loss/train: {epoch_loss} | Epoch Elapsed time: {time.time() - init_time}")
            self.training_loss.append(epoch_loss.cpu().detach().numpy())
            

    def __train_one_batch(self):
        self.train()
        
        #Sampling p_star nodes
        sampled_nodes = torch.multinomial(self.sampling_weights, self.batch_size, replacement=False)
        sampled_p_star_nodes, _ = torch.sort(sampled_nodes, dim=0)
        
        #Paper paper graph edge sampling
        #Edges 
        nodes_num_pp = self.nodes_num_pp
        edges_num_pp = self.edges_num_pp
        A = torch.sparse_coo_tensor(indices=self.edges_pp.to(torch.long), values=torch.ones(edges_num_pp, ),
        size=(nodes_num_pp, nodes_num_pp)).coalesce()
        # Select the edges among the batch nodes
        batch_idx_pp = torch.index_select(A, dim=0, index=sampled_p_star_nodes).coalesce().indices()
        # Get the batch edges
        batch_edges_pp = torch.vstack((sampled_p_star_nodes[batch_idx_pp[0]], batch_idx_pp[1]))
        sampled_p_nodes = torch.tensor(list(set(batch_edges_pp[1].tolist())))


        
        #Author paper graph edge sampling
        #Edges 
        nodes_num_ap = self.nodes_num_ap
        edges_num_ap = self.edges_num_ap
        A = torch.sparse_coo_tensor(indices=self.edges_ap.to(torch.long), values=torch.ones(edges_num_ap, ),
        size=(nodes_num_pp, nodes_num_ap)).coalesce()
        # Select the edges among the batch nodes
        batch_idx_ap = torch.index_select(A, dim=0, index=sampled_p_star_nodes).coalesce().indices()
        # Get the batch edges
        batch_edges_ap = torch.vstack((sampled_p_star_nodes[batch_idx_ap[0]], batch_idx_ap[1]))
        sampled_a_nodes = torch.tensor(list(set(batch_edges_ap[1].tolist())))
       
        average_batch_loss = (self.alpha*(self.forward_pp(edges=batch_edges_pp, p_star_nodes=sampled_p_star_nodes, p_nodes=sampled_p_nodes))/(len(sampled_p_nodes))+(1-self.alpha)*(self.forward_ap(edges=batch_edges_ap, p_star_nodes=sampled_p_star_nodes, a_nodes = sampled_a_nodes))/(len(sampled_a_nodes)))
        return average_batch_loss

    def forward_pp(self, edges, p_star_nodes, p_nodes):

        nll = self.get_neg_likelihood_pp(edges=edges, p_star_nodes=p_star_nodes, p_nodes = p_nodes)

        return nll
    
    def forward_ap(self, edges, p_star_nodes, a_nodes):

        nll = self.get_neg_likelihood_ap(edges=edges, p_star_nodes=p_star_nodes, a_nodes = a_nodes)

        return nll

    def get_params(self):

        return self.beta.detach().numpy(), self.z.detach().numpy()


    def training_error_list(self):
        return self.training_loss
        
    def get_training_errors(self):
        with open("/zhome/c2/3/167669/GRL/Training_loss/training_loss.txt", mode = "w", encoding = "utf-8") as f:
            for i, k in enumerate(self.training_loss):
	            f.write(str(i) + ' ' + str(k) + '\n')
	            
	            
    def save_model(self, path):
        torch.save(self.state_dict(), f'/zhome/c2/3/167669/GRL/Embeddings/{self.alpha}/{path}')

    def save_embs(self, path, format="word2vec"):
        if format == "torch":
            torch.save(model.state_dict(), '/zhome/c2/3/167669/GRL/Embeddings/{self.alpha}/ldm_p_alpha={self.alpha}.emb')
            
			
        if format == "word2vec":
           pass
        '''
        print("Writing p embeddings")
        with open(f'/zhome/c2/3/167669/GRL/Embeddings/{self.alpha}/ldm_p_alpha={self.alpha}.emb', 'w', encoding='utf-8') as f:
            for i in range(self.nodes_num_pp):
                f.write(str(i) + ' ' + str(self.p.cpu().detach().numpy()[i][0]) + ' ' + str(self.p.cpu().detach().numpy()[i][1]) + '\n')
        
        print("Writing pstar embeddings")
        with open(f'/zhome/c2/3/167669/GRL/Embeddings/{self.alpha}/ldm_p_star_alpha={self.alpha}.emb', 'w', encoding='utf-8') as f:
            for i in range(self.nodes_num_pp):
                f.write(str(i) + ' ' + str(self.p_star.cpu().detach().numpy()[i][0]) + ' ' + str(self.p_star.cpu().detach().numpy()[i][1]) + '\n')
        print("Writing a embeddings")        
        with open(f'/zhome/c2/3/167669/GRL/Embeddings/{self.alpha}/ldm_a_alpha={self.alpha}.emb', 'w', encoding='utf-8') as f:
            for i in range(self.nodes_num_ap-self.nodes_num_pp):
                f.write(str(i) + ' ' + str(self.a.cpu().detach().numpy()[i][0]) + ' ' + str(self.a.cpu().detach().numpy()[i][1]) + '\n')

        dim_beta=len(self.beta_ap)
        with open(f'/zhome/c2/3/167669/GRL/Embeddings/{self.alpha}/ldm_beta={self.alpha}.emb', 'w', encoding='utf-8') as f:
            for i in range(dim_beta):
                f.write(str(i) + ' ' + str(self.beta_ap.cpu().detach().numpy()[i][0]) + ' ' + str(self.beta_ap.cpu().detach().numpy()[i][1]) + '\n')
        
        dim_gamma=len(self.gamma_pp)
        with open(f'/zhome/c2/3/167669/GRL/Embeddings/{self.alpha}/ldm_beta={self.alpha}.emb', 'w', encoding='utf-8') as f:
            for i in range(dim_gamma):
                f.write(str(i) + ' ' + str(self.gamma_pp.cpu().detach().numpy()[i][0]) + ' ' + str(self.gamma_pp.cpu().detach().numpy()[i][1]) + '\n')
        '''


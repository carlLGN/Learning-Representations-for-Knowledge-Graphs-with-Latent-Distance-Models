import torch
from torch_sparse import spspmm
import math
import time
import sys
import random


class LDM(torch.nn.Module):
    def __init__(self,  edges, dim, lr=0.1, epoch_num=100, batch_size = 0, spe=1, device=torch.device("cpu"),
                 verbose=False, seed=0):
        super(LDM, self).__init__()

        self.__edges = edges.to(device)
        self.__nodes_num = torch.max(self.__edges) + 1
        self.__edges_num = self.__edges.shape[1]
        self.__dim = dim
        self.__sampling_weights = torch.ones(self.__nodes_num, dtype=torch.float, device=device)
        self.__device = device
        self.__seed = seed
        self.__verbose = verbose

        # Set the seed
        self.__set_seed(self.__seed)

        # Initialize the parameters
        self.beta_p = torch.nn.Parameter(
            2 * torch.rand(size=(self.__nodes_num,), device=self.__device) - 1, requires_grad=True
        )
        
        self.beta_p_star = torch.nn.Parameter(
            2 * torch.rand(size=(self.__nodes_num,), device=self.__device) - 1, requires_grad=True
        )
        
        self.p = torch.nn.Parameter(
            2 * torch.rand(size=(self.__nodes_num, self.__dim), device=self.__device) - 1, requires_grad=True
        )
        self.p_star = torch.nn.Parameter(
            2 * torch.rand(size=(self.__nodes_num, self.__dim), device=self.__device) - 1, requires_grad=True
        )
        self.__epoch_num = epoch_num
        self.__steps_per_epoch = spe
        self.__batch_size = batch_size if batch_size else self.__nodes_num
        self.__learning_rate = lr
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__learning_rate)
        self.__loss = []

        self.__pdist = torch.nn.PairwiseDistance(p=2)

    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)

    def get_embs(self):

        return self.__z

    def get_intensity_sum(self, nodes_p_star, nodes_p):

        beta_p_star = self.beta_p_star if nodes_p_star is None else torch.index_select(self.beta_p_star, index=nodes_p_star, dim=0)
        beta_p = self.beta_p if nodes_p is None else torch.index_select(self.beta_p, index=nodes_p, dim=0)
        
        p_star = self.p_star if nodes_p_star is None else torch.index_select(self.p_star, index=nodes_p_star, dim=0)
        p = self.p_star if nodes_p_star is None else torch.index_select(self.p, index=nodes_p, dim=0)
        
        beta_mat = beta_p_star.unsqueeze(0) + beta_p.unsqueeze(1)
        dist_mat = torch.cdist(p_star, p, p=2)

        return torch.triu(torch.exp(beta_mat.T - dist_mat), diagonal=1).sum()

    def get_log_intensity_sum(self, edges):

        beta_pair = torch.index_select(self.beta_p_star, index=edges[0], dim=0) + \
                    torch.index_select(self.beta_p, index=edges[1], dim=0)

        z_dist = self.__pdist(
            torch.index_select(self.p, index=edges[0], dim=0),
            torch.index_select(self.p_star, index=edges[1], dim=0),
        )

        return (beta_pair - z_dist).sum()

    def get_intensity_for(self, i, j):

        beta_sum = self.__beta[i] + self.__beta[j]
        z_dist = self.__pdist(self.__z[i, :], self.__z[j, :])
        return torch.exp(beta_sum - z_dist)


    def get_neg_likelihood(self, edges, nodes_p_star=None, nodes_p=None):

        # Compute the link term
        link_term = self.get_log_intensity_sum(edges=edges)

        # Compute the non-link term
        non_link = self.get_intensity_sum(nodes_p_star=nodes_p_star, nodes_p=nodes_p)

        return -(link_term - non_link)

    def learn(self):

        for epoch in range(self.__epoch_num):

            self.__train_one_epoch(current_epoch=epoch)

        return self.__loss

    def __train_one_epoch(self, current_epoch):

        init_time = time.time()

        total_batch_loss = 0
        self.__loss.append([])
        for batch_num in range(self.__steps_per_epoch):
            batch_loss = self.__train_one_batch()

            self.__loss[-1].append(batch_loss)

            total_batch_loss += batch_loss

            # Set the gradients to 0
            self.__optimizer.zero_grad()

            # Backward pass
            batch_loss.backward()

            # Perform a step
            self.__optimizer.step()

        # Get the average epoch loss
        epoch_loss = total_batch_loss / float(self.__steps_per_epoch)

        if not math.isfinite(epoch_loss):
            print(f"Epoch loss is {epoch_loss}, stopping training")
            sys.exit(1)

        if self.__verbose and (current_epoch % 10 == 0 or current_epoch == self.__epoch_num - 1):
            print(f"| Epoch = {current_epoch} | Loss/train: {epoch_loss} | Epoch Elapsed time: {time.time() - init_time}")

    def __train_one_batch(self):

        self.train()

        sampled_nodes = torch.multinomial(self.__sampling_weights, self.__batch_size, replacement=False)
        sampled_p_star_nodes, _ = torch.sort(sampled_nodes, dim=0)

        '''
        batch_edges, _ = spspmm(
            indexA=self.__edges.type(torch.long),
            valueA=torch.ones(size=(self.__edges_num, ), dtype=torch.float, device=self.__device),
            indexB=torch.vstack((sampled_nodes, sampled_nodes)).type(torch.long),
            valueB=torch.ones(size=(self.__batch_size,), dtype=torch.float, device=self.__device),
            m=self.__nodes_num, k=self.__nodes_num, n=self.__nodes_num, coalesced=True
        )
        '''
        #Author paper graph edge sampling
        #Edges 
        nodes_num = self.__nodes_num
        edges_num = self.__edges_num
        A = torch.sparse_coo_tensor(indices=self.__edges.to(torch.long), values=torch.ones(edges_num, ),
        size=(nodes_num, nodes_num)).coalesce()
        # Select the edges among the batch nodes
        batch_idx = torch.index_select(A, dim=0, index=sampled_nodes).coalesce().indices()
        # Get the batch edges
        batch_edges = torch.vstack((sampled_p_star_nodes[batch_idx[0]], batch_idx[1]))
        sampled_p_nodes = torch.tensor(list(set(batch_edges[1].tolist())))

        # Forward pass
        average_batch_loss = self.forward(edges=batch_edges, nodes_p_star=sampled_p_star_nodes, nodes_p = sampled_p_nodes)

        return average_batch_loss

    def forward(self, edges, nodes_p_star, nodes_p):

        nll = self.get_neg_likelihood(edges=edges, nodes_p_star = nodes_p_star, nodes_p=nodes_p)

        return nll

    def get_params(self):

        return self.__beta.detach().numpy(), self.__z.detach().numpy()


    def save_embs(self, path, format="word2vec"):

        assert format == "word2vec", "Only acceptable format is word2vec."

        with open(path, 'w') as f:
            f.write(f"{self.__nodes_num} {self.__dim}\n")
            for i in range(self.__nodes_num):
                f.write("{} {}\n".format(i, ' '.join(str(value) for value in self.__z[i, :].detach().cpu().numpy())))

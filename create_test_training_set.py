import random
from paper_size import read_emb3
from load_data import load_subgraph, load_data
import numpy as np
from operator import itemgetter
from tqdm import tqdm



def create_training_set():
    
    author2paper_edgelist = np.asarray(read_emb3('Data/author2paper_edgelist'))
    paper2paper_edgelist = np.asarray(read_emb3('Data/paper2paper_edgelist'))

    G = load_data(path="./Data/paper2paper_2000_gcc.gml")
    print('dataloaded')

    G1=load_data(path="./Data/author2paper_2000_gcc.gml")
    print('dataloaded')

    nodelist = list(G.nodes())
    for node in tqdm(nodelist):
        if not G1.has_node(node):
            G.remove_node(node)

    print('Removing Nodes That Do Not Exist In Both Graphs')
    nodelist = [node for node in G1.nodes if node[0]=='W']
    for node in tqdm(nodelist):
        if not G.has_node(node):
            G1.remove_node(node)

    dates = set([(lis[0], lis[2]) for lis in G.edges(data='date')])
    dates_sorted = sorted(dates, key=itemgetter(1))
    papers = np.asarray(dates_sorted)[:, 0][::-1]

    G_index = {a: i for i, a in enumerate(papers)}

    i = len(papers)
    for node in G.nodes:
        if node not in G_index:
            G_index[node] = i
            i += 1

    #doesnt seem right..
    authors = [node for node in G1.nodes if node[0] == 'A']
    i = len(G.nodes)
    for author in authors:
        G_index[author] = i
        i += 1

    indegreelistpaper=[]
    indpaper = list(G.in_degree())
    for i in tqdm(range(len(G.nodes()))):
        ind_i=indpaper[i][1]
        index_i=G_index[indpaper[i][0]]
        indegreelistpaper.append([index_i,ind_i])

    outdegreelistpaper=[]
    outdpaper = list(G.out_degree())
    for j in tqdm(range(len(G.nodes()))):
        outd_j=outdpaper[j][1]
        index_j=G_index[outdpaper[j][0]]
        outdegreelistpaper.append([index_j,outd_j])

    indegreelist_sorted_paper = sorted(indegreelistpaper, key=itemgetter(0))
    outdegreelist_sorted_paper = sorted(outdegreelistpaper, key=itemgetter(0))

    paper2paper_edgelist_deleted = paper2paper_edgelist.copy()
    edges_deleted_paper=[]
    random.seed(0)
    z=0
    n_to_delete = (0.01*len(paper2paper_edgelist[:,0]))
    while z<=n_to_delete:
        possible_remove = random.choice(list(enumerate(paper2paper_edgelist)))
        if possible_remove[0] not in edges_deleted_paper:
            array_remove=possible_remove[1]
            index_remove=possible_remove[0]
            citing=int(array_remove[0])
            cited=int(array_remove[1])
            if indegreelist_sorted_paper[cited][1]<=1 or outdegreelist_sorted_paper[citing][1]<=1:
                print('not it')
                continue
            else:
                outdegreelistpaper[citing][1]-=1
                indegreelistpaper[cited][1]-=1
                edges_deleted_paper.append(index_remove)
                paper2paper_edgelist_deleted=np.delete(paper2paper_edgelist_deleted, index_remove, 0)
                z+=1
                print(f"{z/n_to_delete}"+"%")

    indegreelistauthor=[]
    indauthor = list(G1.in_degree())
    for i in tqdm(range(len(G1.nodes()))):
        ind_i=indauthor[i][1]
        index_i=G_index[indauthor[i][0]]
        indegreelistauthor.append([index_i, ind_i])

    outdegreelistauthor=[]
    outdauthor=list(G1.out_degree())
    for j in tqdm(range(len(G1.nodes()))):
        outd_j=outdauthor[j][1]
        index_j=G_index[outdauthor[j][0]]
        outdegreelistauthor.append([index_j, outd_j])

    indegreelist_sorted_author = sorted(indegreelistauthor, key=itemgetter(0))
    outdegreelist_sorted_author = sorted(outdegreelistauthor, key=itemgetter(0))

    author2paper_edgelist_deleted = author2paper_edgelist.copy()
    edges_deleted_author=[]
    random.seed(0)
    z=0
    while z<=n_to_delete:
        possible_remove = random.choice(list(enumerate(author2paper_edgelist)))
        if possible_remove[0] not in edges_deleted_author:
            array_remove=possible_remove[1]
            index_remove=possible_remove[0]
            citing=int(array_remove[0])
            cited=int(array_remove[1])
            if indegreelist_sorted_author[cited][1]<=1 or outdegreelist_sorted_author[citing][1]<=1:
                print('not it')
                continue
            else:
                outdegreelistauthor[citing][1]-=1
                indegreelistauthor[cited][1]-=1
                edges_deleted_author.append(index_remove)
                author2paper_edgelist_deleted=np.delete(author2paper_edgelist_deleted, index_remove, 0)
                z+=1
                print(f"{z/n_to_delete}"+"%")

    return paper2paper_edgelist_deleted, paper2paper_edgelist, edges_deleted_paper, author2paper_edgelist_deleted, author2paper_edgelist, edges_deleted_author


if __name__ == '__main__':
    create_training_set()







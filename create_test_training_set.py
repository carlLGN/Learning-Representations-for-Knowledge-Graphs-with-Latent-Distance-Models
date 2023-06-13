import random
from paper_size import read_emb3
from load_data import load_subgraph, load_data
import numpy as np
from operator import itemgetter

# data
author2paper_edgelist = np.asarray(read_emb3('Data/author2paper_edgelist'))
paper2paper_edgelist = np.asarray(read_emb3('Data/paper2paper_edgelist'))

G = load_data()
print('dataloaded')

dates = set([(lis[0], lis[2]) for lis in G.edges(data='date')])
dates_sorted = sorted(dates, key=itemgetter(1))
papers = np.asarray(dates_sorted)[:, 0][::-1]

G_index = {a: i for i, a in enumerate(papers)}

i = len(papers)
for node in G.nodes:
    if node not in G_index:
        G_index[node] = i
        i += 1

indegreelist=[]
for i in range(len(G.nodes())):
    a = list(G.in_degree())
    a2=a[i][1]
    a1=G_index[a[i][0]]
    indegreelist.append([a1,a2])

outdegreelist=[]
for j in range(len(G.nodes())):
    a = list(G.out_degree())
    a2=a[i][1]
    a1=G_index[a[i][0]]
    outdegreelist.append([a1,a2])

indegreelist_sorted = sorted(indegreelist, key=itemgetter(0))
outdegreelist_sorted = sorted(outdegreelist, key=itemgetter(0))

paper2paper_edgelist_deleted = paper2paper_edgelist
edges_deleted=[]
random.seed(0)
z=0
while i<=(0.01*len(paper2paper_edgelist_deleted[:,0])):
    possible_remove = random.choice(list(enumerate(paper2paper_edgelist_deleted)))
    array_remove=possible_remove[1]
    index_remove=possible_remove[0]
    citing=int(array_remove[0])
    cited=int(array_remove[1])
    if indegreelist_sorted[citing][1]<=1 or indegreelist_sorted[cited][1]<=1 or outdegreelist_sorted[citing][1]<=1 or indegreelist_sorted[cited][1]<=1:
        continue
    else:
        outdegreelist[citing][1]-=1
        indegreelist[cited][1]-=1
        edges_deleted.append(index_remove)
        paper2paper_edgelist_deleted=np.delete(paper2paper_edgelist_deleted, index_remove, 0)
        z+=1

print(len(paper2paper_edgelist))
print(len(paper2paper_edgelist_deleted))







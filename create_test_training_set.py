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

degreelist=[]
for i in range(len(G.nodes())):
    a = list(G.degree())
    a2=a[i][1]
    a1=G_index[a[i][0]]
    degreelist.append([a1,a2])

degreelist_sorted = sorted(degreelist, key=itemgetter(0))

paper2paper_edgelist_deleted = paper2paper_edgelist
random.seed(0)
i=0
while i<=(0.01*len(paper2paper_edgelist_deleted[:,0])):
    possible_remove = random.choice(list(enumerate(paper2paper_edgelist_deleted)))
    array_remove=possible_remove[1]
    index_remove=possible_remove[0]
    citing=int(array_remove[0])
    cited=int(array_remove[1])
    if degreelist_sorted[citing][1]<=1 or degreelist_sorted[cited][1]<=1:
        continue
    else:
        paper2paper_edgelist_deleted=np.delete(paper2paper_edgelist_deleted, index_remove, 0)
        i+=1

print(len(paper2paper_edgelist))
print(len(paper2paper_edgelist_deleted))







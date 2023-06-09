import networkx as nx
from tqdm import tqdm
from operator import itemgetter
import numpy as np

def load_data(mode="paper2paper", path=None):
    if path:
        return nx.read_gml(path)
    else:
        return nx.read_gml(f"./Data/{mode}_2000_gcc.gml")

def load_subgraph(mode = "paper2paper"):
    return nx.read_gml(f"./Data/subgraph_{mode}.gml")
    

def create_subgraph(mode = "paper2paper", nodes=1000):
    graph = load_data(mode)
    print(graph)
    subgraph = graph.subgraph(list(graph.nodes)[0:nodes])
    print(subgraph)
    nx.write_gml(subgraph, f"./Data/subgraph_{mode}.gml")


#Converts nx graph to edgelist. Takes nx graph.
def nx_to_edgelist(): #G=None

    # if not G:
    #     G = load_subgraph()

    G=load_data(path="./Data/paper2paper.gml")

    print('graph loaded')
    G1=load_data(path="./Data/author2paper.gml")

    print('graph loaded')


    nodelist = list(G.nodes())
    for node in tqdm(nodelist):
        if not G1.has_node(node):
            G.remove_node(node)

    nodelist = [node for node in G1.nodes if node[0]=='W']
    for node in tqdm(nodelist):
        if not G.has_node(node):
            G1.remove_node(node)

    dates = [(lis[0], lis[2]) for lis in G.edges(data='date')]
    dates_sorted = sorted(dates,key=itemgetter(1))
    articles = np.asarray(dates_sorted)[:,0][::-1]

    mapping = {a: i for i,a in enumerate(articles)}

    i = len(articles)
    for node in G.nodes:
        if node not in mapping:
            mapping[node] = i
            i += 1
    

    authors = [node for node in G1.nodes if node[0] == 'A']
    i = len(G.nodes)
    for author in authors:
        mapping[author] = i
        i += 1

    #Update function to write reverse mapping to document?
    n_paper=len(G.nodes)
    with open(file='paper2paper_edgelist', mode = 'w', encoding='utf-8') as f:
        for i in G.edges():
                #We map all article values to a value between 1 and len articles         
            f.write(str(mapping[i[0]]) + ' ' + str(mapping[i[1]]) + ' 1.0\n')

    with open(file='author2paper_edgelist', mode = 'w', encoding='utf-8') as f:
        for i in G1.edges():
                #We map all article values to a value between 1 and len articles
                
            f.write(str(mapping[i[1]]) + ' ' + str(mapping[i[0]]+n_paper) + ' 1.0\n')

# def create_edgelists():
#     G1 = load_data(path="./Data/paper2paper.gml")
#     print("First Graph Loaded")
#     G2 = load_data(path="./Data/author2paper.gml")
#     print("Data Loaded")

#     # This piece of code handles papers that appear in the citation network,
#     # but not in the author-paper network
#     nodelist = list(G1.nodes())
#     for node in tqdm(nodelist):
#         if not G2.has_node(node):
#             G1.remove_node(node)

    

#     print("Saving first edgelist")
#     nx_to_edgelist("Data/paper2paper_edgelist", G1)
#     print("Saving second edgelist")
#     nx_to_edgelist("Data/author2paper_edgelist", G2)

if __name__ == '__main__':
    nx_to_edgelist()
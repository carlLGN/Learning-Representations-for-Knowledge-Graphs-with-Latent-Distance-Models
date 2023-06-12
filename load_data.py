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
def nx_to_edgelist():

    G=load_data(path="./Data/paper2paper.gml")

    print('First Graph Loaded')
    G1=load_data(path="./Data/author2paper.gml")

    print('Second Graph Loaded')


    nodelist = list(G.nodes())
    for node in tqdm(nodelist):
        if not G1.has_node(node):
            G.remove_node(node)


    print('Removing Nodes That Do Not Exist In Both Graphs')
    nodelist = [node for node in G1.nodes if node[0]=='W']
    for node in tqdm(nodelist):
        if not G.has_node(node):
            G1.remove_node(node)

    print('Sorting By Dates')
    dates = set([(lis[0], lis[2]) for lis in G.edges(data='date')])
    dates_sorted = sorted(dates,key=itemgetter(1))
    papers = np.asarray(dates_sorted)[:,0][::-1]


    print('Mapping')
    mapping = {a: i for i,a in enumerate(papers)}

    i = len(papers)
    for node in G.nodes:
        if node not in mapping:
            mapping[node] = i
            i += 1
    

    authors = [node for node in G1.nodes if node[0] == 'A']
    i = len(G.nodes)
    for author in authors:
        mapping[author] = i
        i += 1

    print('Writing Edgelists')
    with open(file='./Data/paper2paper_edgelist', mode = 'w', encoding='utf-8') as f:
        for k in G.edges():
                #We map all article values to a value between 1 and len articles         
            f.write(str(mapping[k[0]]) + ' ' + str(mapping[k[1]]) + ' 1.0\n')

    with open(file='./Data/author2paper_edgelist', mode = 'w', encoding='utf-8') as f:
        for k in G1.edges():
                #We map all article values to a value between 1 and len articles
                
            f.write(str(mapping[k[1]]) + ' ' + str(mapping[k[0]]) + ' 1.0\n')

if __name__ == '__main__':
    nx_to_edgelist()
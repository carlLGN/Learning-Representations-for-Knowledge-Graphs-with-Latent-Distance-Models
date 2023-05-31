import networkx as nx

def load_data(mode="paper2paper"):
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
def nx_to_edgelist(G=None):

    if not G:
        G = load_subgraph()

    with open("Data/paper2paper_edgelist", 'w', encoding='utf-8') as f:
        for i in G.edges():
            #As all vertices in the paper2paper graph are papers (and thus start with a W)
            #We remove 'W' by indexing in the string
            f.write(str(i[0])[1:] + ' ' + str(i[1])[1:] + ' 1.0\n')


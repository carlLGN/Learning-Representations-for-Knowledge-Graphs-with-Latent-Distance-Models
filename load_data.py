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

    
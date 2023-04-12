import networkx as nx

def load_data(mode="paper2paper"):
    return nx.read_gml(f"./Data/{mode}_2000_gcc.gml")
    
def load_subgraph(mode = "paper2paper"):
    return nx.read_gml(f"./Data/subgraph_{mode}.gml")
    

def create_subgraph(mode = "paper2paper", nodes=1000):
    graph = load_data(mode)
    node_subset = {i for i in range(nodes)}
    subgraph = graph.subgraph(node_subset)
    
    nx.write_gml(subgraph, f"./Data/subgraph_{mode}.gml")

    
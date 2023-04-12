import networkx as nx

def load_data(mode="paper2paper"):
    if mode=="paper2paper":
        return nx.read_gml("./paper2paper_2000_gcc.gml")
    
    if mode=="author2paper":
        return nx.read_gml("./author2paper_2000_gcc.gml")


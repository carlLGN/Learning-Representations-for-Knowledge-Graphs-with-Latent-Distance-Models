import networkx as nx

def load_part_data(mode="paper2paper"):
    if mode=="paper2paper":
        return nx.read_gml("./Data/paper2paper_2000_gcc.gml", label = 'id', limit = 1000)
    
    if mode=="author2paper":
        return nx.read_gml("./Data/author2paper_2000_gcc.gml", label = 'id', limit = 1000)


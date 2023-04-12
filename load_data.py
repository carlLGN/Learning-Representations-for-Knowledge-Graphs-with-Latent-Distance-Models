import networkx as nx

def load_data(mode="paper2paper"):
    if mode=="paper2paper":
        return nx.read_gml("./Data/paper2paper_2000_gcc.gml")
    
    if mode=="author2paper":
        return nx.read_gml("./Data/author2paper_2000_gcc.gml")


graph = load_data()


print(graph)

print('123')
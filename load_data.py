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

    articles = set(list(G.nodes))
    mapping = {a: i for i,a in enumerate(articles)}
    
    #Update function to write reverse mapping to document?
    
    with open("Data/paper2paper_edgelist", 'w', encoding='utf-8') as f:
        for i in G.edges():
            #We map all article values to a value between 1 and len articles
            
            f.write(str(mapping[i[0]]) + ' ' + str(mapping[i[1]]) + ' 1.0\n')

nx_to_edgelist()


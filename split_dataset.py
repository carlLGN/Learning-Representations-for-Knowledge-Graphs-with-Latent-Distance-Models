from visualization_utils import read_emb3
import random
import tqdm


'''Script splits dataset into train and test edgelists, given edgelists. Selects 1% of edges to be test-edges'''
if __name__ == '__main__':
    edgelist_pp = read_emb3('./Data/paper2paper_edgelist')
    p_star = [int(x[0]) for x in edgelist_pp]
    p = [int(x[1]) for x in edgelist_pp]
    num_nodes = max(max(p_star), max(p)) + 1
    num_edges = len(p_star)
    
    deg_p_star = {x:0 for x in range(num_nodes)}
    for i in p_star:
        deg_p_star[i] += 1
        
    deg_p = {x:0 for x in range(num_nodes)}
    for i in p:
        deg_p[i] += 1
    
    
    train = edgelist_pp
    test = []
    
    print("starting loop")
    n_to_remove = 0.01*num_edges
    while len(test) < n_to_remove:
        index = random.randint(0, len(train))
        random_p_star, random_p, value = train[index]
        if deg_p_star[random_p_star] >= 2 and deg_p[random_p] >= 2:
            test.append(train[index])
            train.pop(index)
            deg_p_star[random_p_star] -= 1
            deg_p[random_p] -= 1
            if len(test) % 1000 == 0:
                print(len(test))
        else:
            continue

    
    with open("train_edgelist_pp", mode = 'w', encoding='utf-8') as f:
        for edge in train:
            f.write(str(int(edge[0])) + ' ' + str(int(edge[1])) + ' 1.0\n')
            
    
    with open("test_edgelist_pp", mode = 'w', encoding='utf-8') as f:
        for edge in test:
            f.write(str(int(edge[0])) + ' ' + str(int(edge[1])) + ' 1.0\n')
            
    
    print('debug')





import numpy as np
import networkx as nx

from der_graph_clustering import *



def block_model_graph(Nnodes,Ncomps,p_in,p_out,permute_order = True):
    
    comp_len = Nnodes / Ncomps

    
    nodes = range(Nnodes)

    G = nx.Graph()    
    G.add_nodes_from(nodes)

    
    #to make sure that initialization of algorithms does not 
    #get the order as a hint
    if permute_order:    
        nodes = list(np.random.permutation(nodes))
    
    components = []
    
    c_idx = 0
    for i in range(Ncomps):
        
        if i == (Ncomps-1): 
            components.append(nodes[c_idx:])
        else:            
            components.append(nodes[c_idx:c_idx+comp_len])
        
        c_idx += comp_len
        
        
    #print components
    
    
    edges = []
    
    #now create edges
    #within the components
    for c in components:
        clen = len(c)
        full_edge_cnt = clen*(clen - 1) / 2
        rnd_cnt = 0
        rnd = np.random.random(full_edge_cnt)
        
        for i,n1 in enumerate(c[:-1]):
            for n2 in c[i+1:]:
                if rnd[rnd_cnt] <= p_in:
                    edges.append((n1,n2))
                rnd_cnt += 1
                
                     
    #between the components
    for i,c1 in enumerate(components[:-1]):
        for c2 in components[i+1:]:
                
            full_edge_cnt = len(c1)*len(c2)
            rnd_cnt = 0
            rnd = np.random.random(full_edge_cnt)
        
            for n1 in c1:
                for n2 in c2:
                    if rnd[rnd_cnt] <= p_out:
                        edges.append((n1,n2))
                    rnd_cnt += 1

     
    G.add_edges_from(edges)
     
    return G,components




def communities_to_labels(N, comms):
    
    Ncomms = len(comms)
    
    assert sum(map(len,comms)) == N , 'Communities do not cover the node set'
    
    labels = np.zeros(N, dtype = 'int')
    for i,comm in enumerate(comms):
        labels[comm] = i
    
    return labels



from sklearn.metrics import normalized_mutual_info_score


if __name__ == '__main__':
    
    N = 1000
    Ncomp = 10
    
    G, orig_components  = block_model_graph(N,Ncomp,.5,.1)
            
    TM = graph_transition_matrix(G,sparse = True)
    
    #use multiple restarts in general 
    communities, mix = der_graph_clustering(G,TM,
                                             NCOMPONENTS = Ncomp,    
                                             WALK_LEN = 3, 
                                             alg_threshold = .00001, 
                                             alg_iterbound = 50
                                            )
    
    labels = communities_to_labels(N, communities)
    labels_orig = communities_to_labels(N,orig_components)
    
    #note that sklearn normalization is different from the one used in the literature
    print 'NMI:{}'.format(normalized_mutual_info_score(labels,labels_orig))
















from collections import defaultdict
import networkx as nx
import jsonlines
from collections import Counter
import numpy as np
from numpy import linalg as LA
import operator

def create_tweet_graph_from_file(filename):
    edges_list = [] ## This list contains all the directed edges in the graph

    ## Reading the input json file
    with jsonlines.open(filename, 'r') as f:
        for jsn in f:
            rt_user_id = jsn["user"]["id"]
            source_user_id = jsn["retweeted_status"]["user"]["id"]
            if rt_user_id != source_user_id:
                edges_list.append((rt_user_id, source_user_id))
    
    ## Creating a weighted edge list from the edges_list
    weighted_edge_list = Counter(edges_list)
    tweet_graph = nx.DiGraph() ## Creating an empty directed graph
    
    # Adding edges to the directed graph from the weighted edges list
    for edge in weighted_edge_list.items():
        source = edge[0][0]
        destination = edge[0][1]
        weight = edge[1]
        tweet_graph.add_edge(source, destination, weight=weight)
    return tweet_graph

tweet_graph = create_tweet_graph_from_file('HITS.json')
tweet_graph.size()

# Given input graph, this method is the implementation of hits algorithms.
# It returns the hubs and authorities score

def hits(graph, iter_count = 20):
    nodes = graph.nodes()
    nodes_count = len(nodes)
    matrix = nx.to_numpy_matrix(graph, nodelist=nodes)
    
    hubs_score = np.ones(nodes_count)
    auth_score = np.ones(nodes_count)
    H = matrix * matrix.T
    A = matrix.T * matrix
   
    for i in range(iter_count):
       
        hubs_score = hubs_score * H 
        auth_score = auth_score * A 
        hubs_score = hubs_score / LA.norm(hubs_score)
        auth_score = auth_score / LA.norm(auth_score)
        
    hubs_score = np.array(hubs_score).reshape(-1,)
    auth_score = np.array(auth_score).reshape(-1,)
    
    hubs = dict(zip(nodes, hubs_score))
    authorities = dict(zip(nodes, auth_score))
    return hubs, authorities

# Given a graph, this method returns top k hubs
def get_top_k_hubs(graph, k = 10):
    hubs = hits(graph)[0]
    return sorted(hubs.items(), key = operator.itemgetter(1), reverse = True)[:k]

#Given a graph, this method returns top k authorities
def get_top_k_authorities(graph, k = 10):
    auth = hits(graph)[1]
    return sorted(auth.items(), key = operator.itemgetter(1), reverse = True)[:k]

top_10_tweet_hubs = get_top_k_hubs(tweet_graph)
print("Top 10 hubs ")
top_10_tweet_hubs

top_10_tweet_auth = get_top_k_authorities(tweet_graph)
print("Top 10 authorities")
top_10_tweet_auth


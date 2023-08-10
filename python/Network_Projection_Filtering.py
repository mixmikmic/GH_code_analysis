import networkx as nx
from networkx.algorithms import bipartite
import scipy as sp
import numpy as np
import math
import cmath
import csv
import itertools
import sys
import unicodedata
import string
from scipy.stats import hypergeom
from math import log
from scipy import special
from numpy import mean, sqrt, square
from collections import defaultdict, Counter
from itertools import permutations
from matplotlib.dates import date2num , DateFormatter
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from IPython.display import Image

import igraph

get_ipython().magic('matplotlib inline')

g=nx.Graph()
heroes_ids= set()
for line in open ('marvel_heroes.nodes'):
    id, name = line.strip().split(' ', 1)
    g.add_node(int(id), {'name': name.replace('"', ''), 'bpartite': 0})
    heroes_ids.add(int(id))

with open('marvel_hero_attr.csv','wb') as csvfile:
    writer=csv.writer(csvfile)
    for (a,k) in g.nodes(data=True):
        writer.writerow((str(a), unicode(k['name'])))

issues_ids= set()
issue_name_to_id={}
for line in open ('marvel_issues.nodes'):
    id, name = line.strip().split(' ', 1)
    g.add_node(int(id), {'name': name.replace('"', ''), 'bpartite': 1})
    issues_ids.add(int(id))
    issue_name_to_id[name.replace('"', '')]=int(id)

for line in open ('marvel.edges'):
    ids = map (int, line.strip().split())
    from_id = ids[0]
    for to_id in ids[1:]:
        g.add_edge(from_id, to_id)    

Projection_structure=defaultdict(set)
issues=()

for (hero, issue) in g.edges():
    Projection_structure[hero].add(issue)

def transpose(out_edges):
    in_edges = {}
    for src, es in out_edges.items():
        for dst in es:
            in_edges.setdefault(dst, set()).add(src)

    return in_edges

in_edges=dict(Projection_structure)
out_edges=dict(transpose(Projection_structure))

def simple_projection(out_edges):
    res = {}
    
    for s in out_edges.values():
        # s will be a set of neighbors of a given node
        for src in s:
            es = res.setdefault(src, {})
            for dst in s:
                if src == dst:
                    continue
                
                weight = es.get(dst, 0)
                es[dst] = weight + 1
    return res

def jaccard_similarity(out_edges, in_edges):
    res = simple_projection(out_edges)
    
    for src, es in res.items():
        for dst, w in es.items():
            d_src = float(len(in_edges[src]))
            d_dst = float(len(in_edges[dst]))
            es[dst] = w / (d_src + d_dst - w)
    
    return res

def cosine_similarity(out_edges, in_edges):
    res = simple_projection(out_edges)
    
    for src, es in res.items():
        for dst, w in es.items():
            d_src = len(in_edges[src])
            d_dst = len(in_edges[dst])
            es[dst] = w / (d_src * d_dst) ** .5
    
    return res

def pearson_coefficient(out_edges, in_edges):
    res = simple_projection(out_edges)
    n = float(len(out_edges))
    
    for src, es in res.items():
        for dst, w in es.items():
            p_src = len(in_edges[src]) / n
            p_dst = len(in_edges[dst]) / n
            var = p_src * p_dst * (1. - p_src) * (1. - p_dst)
            es[dst] = (w / n - p_src * p_dst) / var ** .5
    
    return res

def newman_projection(out_edges):
    res = {}
    
    for s in out_edges.values():
        for src in s:
            es = res.setdefault(src, {})
            d = len(s)
            
            for dst in s:
                if src == dst:
                    continue
                
                # Basically the same as a simple projection, except that
                # if a node is connected to multiple other nodes, the additional
                # weight is penalized.
                es[dst] = es.get(dst, 0) + 1. / (d - 1.)
    return res

Image(filename='keplet.png') 

def tbj_similarity(out_edges, in_edges):
    res = simple_projection(out_edges)
    n = len(out_edges)
    
    for src, es in res.items():
        for dst, w in es.items():
            d_src = len(in_edges[src])
            d_dst = len(in_edges[dst])
            
            # Basically negative of the log-survival function, except that
            # to avoid problems with digital precision, a maximal value is set.
            es[dst] = -log(max(1e-300, hypergeom.sf(w, n, d_src, d_dst)))
    
    return res

simple=simple_projection(out_edges)
newman=newman_projection(out_edges)
jaccard=jaccard_similarity(out_edges, in_edges)
cosine=cosine_similarity(out_edges, in_edges)
pearson=pearson_coefficient(out_edges, in_edges)
tbj=tbj_similarity(out_edges, in_edges)

def weight_distr(projection):
    weights=[]
    for key,values in projection.iteritems():
        for k,i in values.iteritems():
            weights.append(i)
    
    x=max(weights)
    weights_norm=[float(i)/x for i in weights]
    
    
    plt.hist((weights_norm),bins=50)
    plt.title("Weights distribution")
    plt.ylabel("number of nodes")
    plt.xlabel("weigths")
    plt.yscale('log', nonposy='clip')
    plt.show()
    
    
    print 'Normalized weights'
    print "mean: "+ str(np.mean(weights_norm))
    print "std: "+ str(np.std(weights_norm))
    print "min: "+ str(min(weights_norm))
    print "max: "+ str(max(weights_norm))
    
    print 'Original weights'
    print "mean: "+ str(np.mean(weights))
    print "std: "+ str(np.std(weights))
    print "min: "+ str(min(weights))
    print "max: "+ str(max(weights))
    

weight_distr(simple)

weight_distr(jaccard)

weight_distr(pearson)

weight_distr(cosine)

weight_distr(newman)

weight_distr(tbj)

Image(filename='clustering_original.png') 

with open('simple.csv','wb') as csvfile:
    writer=csv.writer(csvfile)
    for ke,ve in simple.iteritems():
        for k,v in simple[ke].iteritems():
            writer.writerow((int(ke), int(k), int(v)))

with open('newman.csv','wb') as csvfile:
    writer=csv.writer(csvfile)
    for ke,ve in newman.iteritems():
        for k,v in newman[ke].iteritems():
            writer.writerow((ke, k, v))

with open('pearson.csv','wb') as csvfile:
    writer=csv.writer(csvfile)
    for ke,ve in pearson.iteritems():
        for k,v in pearson[ke].iteritems():
            writer.writerow((ke, k, v))

with open('cosine.csv','wb') as csvfile:
    writer=csv.writer(csvfile)
    for ke,ve in cosine.iteritems():
        for k,v in cosine[ke].iteritems():
            writer.writerow((ke, k, v))

with open('tbj.csv','wb') as csvfile:
    writer=csv.writer(csvfile)
    for ke,ve in tbj.iteritems():
        for k,v in tbj[ke].iteritems():
            writer.writerow((ke, k, v))

with open('jaccard.csv','wb') as csvfile:
    writer=csv.writer(csvfile)
    for ke,ve in jaccard.iteritems():
        for k,v in jaccard[ke].iteritems():
            writer.writerow((ke, k, v))

def extract_backbone(network, alpha, directed):
    nodes = set()
    edges = set()
    for n in network:
        k_n = len(network[n])
        if k_n > 1:
            sum_w = sum(network[n][nj] for nj in network[n])
            for nj in network[n]:
                pij = 1.0 * network[n][nj] / sum_w
                if (1 - pij) ** (k_n - 1) < alpha:
                    nodes.add(n)
                    nodes.add(nj)
                    if directed:
                        edges.add((n, nj, network[n][nj]))
                    else:
                        if n < nj:
                            edges.add((n, nj, network[n][nj]))
                        else:
                            edges.add((nj, n, network[nj][n]))
    return edges

def backbone_test(projection, infile, outfile):

    f = open(infile, "r")
    network = defaultdict(lambda : defaultdict(float))
    for line in f:
        fields = line.strip().split(',')
        network[fields[0]][fields[1]] = float(fields[2])
        network[fields[1]][fields[0]] = float(fields[2])
    f.close()
    
    directed = ("n")
    edgesc=[]
    
    for ke,ve in simple.iteritems():
            for k,v in simple[ke].iteritems():
                edgesc.extend((ke,k))
    alpha=float(0.01)/(2*len(edgesc))
    #float(0.01/2*len(edgesc))
    
    edges=extract_backbone(network, alpha, directed)
    
    with open(outfile, 'wb') as csvfile:
        writer=csv.writer(csvfile)
        for (fr,to,weight) in edges:
            writer.writerow((fr,to,weight))

    
    G = nx.read_weighted_edgelist(outfile, delimiter=',')
    print "Number of nodes:"+str(G.number_of_nodes())
    print "Number of edges:"+str(G.number_of_edges())
    print 'Network Desity: '+str(nx.density(G))
    giant = max(nx.connected_component_subgraphs(G), key=len)
    print 'Percentage of the number of nodes belonging to the giant component: '+str(float(giant.number_of_nodes())/float(G.number_of_nodes()))

    d = nx.degree(G)
    plt.hist(d.values(),bins=50)
    plt.title("Degree Distribution Histogram")
    plt.ylabel("number of nodes")
    plt.xlabel("degree")
    plt.yscale('log', nonposy='clip')
    plt.show()

    weights=[]
    for (_,_,k) in edges:
        weights.append(k)
    print "min:"+str(min(weights))
    print "max:"+str(max(weights))
    print "SD:"+str(np.std(weights))
    

    #plt.hist((weights),bins=1)
    #plt.title("Weights distribution")
    #plt.ylabel("number of nodes")
    #plt.xlabel("weigths")
    #plt.yscale('log', nonposy='clip')
    #plt.show() 

Image(filename='original.png') 

backbone_test(simple, 'simple.csv', 'simple_backbone.csv')

backbone_test(jaccard, 'jaccard.csv', 'jaccard_backbone.csv')

backbone_test(cosine, 'cosine.csv', 'cosine_backbone.csv')

backbone_test(pearson, 'pearson.csv', 'pearson_backbone.csv')

backbone_test(newman, 'newman.csv', 'newman_backbone.csv')

backbone_test(tbj, 'tbj.csv', 'tbj_backbone.csv')

Image(filename='table.png') 


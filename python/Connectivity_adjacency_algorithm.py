from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
from plotly import tools
import plotly

import os

import csv,gc  # garbage memory collection :)

import numpy as np
from numpy import linalg as LA

import csv
import re
import matplotlib
import time
import seaborn as sns

from collections import OrderedDict

import networkx as nx
import math

plotly.offline.init_notebook_mode()

def plot_connectivity(dictionary):
    ## plot points in XY-plane
    ## clusters of points are determined beforehand, each group given a different color
    ## also plots centroid of each cluster ((x,y) of centroid is average of cluster points)
    ## lines indicate nearest centroid by Euclidean distance, prints distance
    current_palette = sns.color_palette("husl", len(dictionary.keys()))
    Xe = []
    Ye = []
    data = []
    avg_dict = OrderedDict()
    i = 0
    for key, region in dictionary.iteritems():
        X = []
        Y = []
#         Z = []
        tmp_x = []
        tmp_y = []
        
        region_col = current_palette[i]
        region_col_lit = 'rgb' + str(region_col)
        i += 1
        for coord in region:    
            X.append(coord[0])
            Y.append(coord[1])
            tmp_x.append(coord[0])
            tmp_y.append(coord[1])
        avg_dict[key] = [[np.mean(tmp_x), np.mean(tmp_y)]]
            
        trace_scatter = Scatter(
                x = X, 
                y = Y,
                name=key,
                mode='markers',
                marker=dict(
                    size=10,
                    color=region_col_lit, #'purple',                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.5
                )
        )
        avg_scatter = Scatter(
                x = [avg_dict[key][0][0]],
                y = [avg_dict[key][0][1]],
                mode='markers',
                name=key+'_avg',
                marker=dict(
                    size=10,
                    color=region_col_lit,
                    colorscale='Viridis',
                    line=dict(
                        width = 2,
                        color = 'rgb(0, 0, 0)'
                    )
                )
        )
        data.append(trace_scatter)
        data.append(avg_scatter)
        
    connectivity=np.zeros((4,4))
    connectivity2=np.zeros((4,4))
    locations = avg_dict.keys()
    for i, key in enumerate(avg_dict):
        tmp = []
        for j in range(len(locations)):
            if j == i:
                connectivity[i,j] = 1
                connectivity2[i,j] = 1
                continue
            p1 = np.asarray(avg_dict[key][0])
            p2 = np.asarray(avg_dict[locations[j]][0])
            dist = LA.norm(p1 - p2)
            tmp.append(dist)
            connectivity[i,j] = math.exp(-dist)
            connectivity2[i,j] = math.exp(-(dist)**2)
            print "Distance between region " + key + " and region " + locations[j] + " is: " + str(dist)
        newmin = tmp.index(min(tmp))
        if newmin >= i:
            newmin += 1
        print "region " + key + " is closest to region " + locations[newmin] + "\n"
        tmp2 = avg_dict.keys()[newmin]
        Xe+=[avg_dict[key][0][0],avg_dict[tmp2][0][0],None]
        Ye+=[avg_dict[key][0][1],avg_dict[tmp2][0][1],None]
#         Ze+=[dictionary[key][0][2],dictionary[tmp2][0][2],None]
    
    trace_edge = Scatter(x=Xe,
               y=Ye,
               mode='lines',
               line=Line(color='rgb(0,0,0)', width=3),
               hoverinfo='none'
    )

    data.append(trace_edge)
    
    layout = Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(255,255,255)'
    )
        
    fig = Figure(data=data, layout=layout)
    iplot(fig, validate=False)
    
    return connectivity, connectivity2

# Fix random seed
np.random.seed(123456789)

np.random.seed(123456789)
a = np.random.randn(4, 2)
i=0
while i != 4:
    tmp = np.random.randn(2,)
    if tmp[0] < 0 or tmp[1] < 0:
        continue
    if np.linalg.norm(tmp) < 1:
        a[i,] = tmp
        i += 1
print a

B = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        B[i,j] = np.dot(a[i,:], a[j,:])

print B

# create 2D plot showing algorithm
# 4 'regions' indicated by 4 x 100 points: a,b,c,d
# obtain the average coordinates for each region to obtain the representative point for each region
# and connect the 4 points such that if any of the points were chosen as starting point, it would only connect to nearest 'region'
# loop through all points and set them as starting point, connectivity map will self-terminate when the last two points point to each other as mutual 'nearest regions'

# test with spherical gaussian mixture

# np.random.seed(123456789)
# a_norm = 0.5 * np.random.randn(100, 2) + np.array([-0.1,0])
# b_norm = 0.5 * np.random.randn(100, 2) + np.array([0,2.1])
# c_norm = 0.5 * np.random.randn(100, 2) + np.array([2,2.1])
# d_norm = 0.5 * np.random.randn(100, 2) + np.array([2.1,0])

# norm_dict = OrderedDict([('a',a_norm),('b',b_norm),('c',c_norm),('d',d_norm)])


np.random.seed(123456789)
a_norm = np.random.randn(100, 2) + a[0,:]
b_norm = np.random.randn(100, 2) + a[1,:]
c_norm = np.random.randn(100, 2) + a[2,:]
d_norm = np.random.randn(100, 2) + a[3,:]

norm_dict = OrderedDict([('a',a_norm),('b',b_norm),('c',c_norm),('d',d_norm)])

# regions a to b to c to d
original, original2 = plot_connectivity(norm_dict)

# connect the 400 points using epsilon ball 
# want fully connected graph to create one connected component
radius = 1
allpt = np.concatenate([a_norm,b_norm,c_norm,d_norm])

G=nx.Graph()

# generate networkx graph object to obtain adjacency matrix, edges have weight of 1, 
# with epsilon ball radius according to above
for i in range(len(allpt)):
    G.add_node(str(i),pos=allpt[i],region=i/100)
    for j in range(i+1,len(allpt)):
        dist = LA.norm(allpt[i] - allpt[j])
        if dist < radius:
            G.add_edge(str(i),str(j),distance=dist)

print len(G.nodes())
print len(G.edges())
# print G.nodes()
# nx.write_graphml(G, "connectivity.graphml")

def plot_graphml(G):
    ## plots graphML object
    ## plots all edges and points
    ## edge graph
    current_palette = sns.color_palette("husl", len(G.nodes())/100)
    Xe = []   ## list of x-coordinates of edges
    Ye = []   ## list of y-coordinates of edges
    data = []
    i = 0
    
    X = []
    Y = []
    regiondict = {}
    for r, node in enumerate(G.nodes()):
        tmp = G.node[node]
        pos = tmp['pos']
        region = tmp['region']
        if str(region) not in regiondict:
            regiondict[str(region)] = [pos]
        else:
            tmp = regiondict[str(region)]
            tmp.append(pos)
            regiondict[str(region)] = tmp

    for region, reg in enumerate(regiondict):
        for pos in regiondict[reg]:
            X.append(pos[0])
            Y.append(pos[1])
                
        region_col = current_palette[region]
        region_col_lit = 'rgb' + str(region_col)
        
        trace_scatter = Scatter(
                x = X, 
                y = Y,
                name=region,
                mode='markers',
                marker=dict(
                    size=10,
                    color=region_col_lit, #'purple',                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.5
                )
        )
        data.append(trace_scatter)
        X = []
        Y = []
        
    for r, edge in enumerate(G.edges()):
        firstpt = G.node[edge[0]]
        secondpt = G.node[edge[1]]
        dist = LA.norm(firstpt['pos'] - secondpt['pos'])

        Xe+=[firstpt['pos'][0],secondpt['pos'][0],None]
        Ye+=[firstpt['pos'][1],secondpt['pos'][1],None]
#         Ze+=[dictionary[key][0][2],dictionary[tmp2][0][2],None]
    
    trace_edge = Scatter(x=Xe,
               y=Ye,
               mode='lines',
               line=Line(color='rgb(0,0,0)', width=2),
               hoverinfo='none'
    )

    data.append(trace_edge)
    
    layout = Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(255,255,255)'
    )
        
    fig = Figure(data=data, layout=layout)
    iplot(fig, validate=False)

# plot the points and edges of the networkx graph object with epsilon ball of radius = 0.9
plot_graphml(G)

## use networkx to generate normalized laplacian matrix from graphML
## find all eigenvalues and eigenvectors of laplacian

import scipy.sparse as sparse
lapl = nx.normalized_laplacian_matrix(G)
evals, evecs = sparse.linalg.eigs(lapl, k=398)

print evals.shape, evecs.shape

print evals.shape

import sys
def find3Smallest(arr):
    ## There should be atleast three elements
    ## Finds the three smallest values of array and their respective indices of array
    arr_size = len(arr)
    if arr_size < 3:
        print "Invalid Input"
        return
 
    first = second = third = sys.maxint
    i1 = i2 = i3 = 0
    for i in range(0, arr_size):
 
        # If current element is smaller than first then
        # update first, second, and third
        if arr[i] < first:
            third = second
            i3 = i2
            second = first
            i2 = i1
            first = arr[i]
            i1 = i
 
        # If arr[i] is in between first and second then
        # update second and third
        elif (arr[i] < second and arr[i] != first):
            third = second
            i3 = i2
            second = arr[i]
            i2 = i
        # If arr[i] is in between second and third then
        # update third
        elif(arr[i] < third and arr[i] != second):
            third = arr[i]
            i3 = i
    
    return i1, i2, i3

# find smallest eigenvalues and eigenvectors
first, second, third = find3Smallest(evals)

evecs3 = evecs[:,(first, second, third)]
print evecs3

nodelist = G.nodes()

# get list of nodes from networkx
# sort nodes by region, group them, add to dictionary
# create dictionary of regions and list of rows of embedded eigenvectors
a_list = []
b_list = []
c_list = []
d_list = []
for ind, i in enumerate(nodelist):
    region = int(float(i))/100
    if region == 0:
        a_list.append(ind)
    elif region == 1:
        b_list.append(ind)
    elif region == 2:
        c_list.append(ind)
    elif region == 3:
        d_list.append(ind)
        
a_region = evecs3[a_list].real  # shape is of n x 2
b_region = evecs3[b_list].real
c_region = evecs3[c_list].real
d_region = evecs3[d_list].real
se_regions_manual = OrderedDict([('a', a_region),('b', b_region),('c', c_region),('d', d_region)])

manual, manual2 = plot_connectivity(se_regions_manual)





# comparison of spectral embedding with Sklearn's implementation
# our manual implementation follows the uploaded PDF documentation of spectral embedding using the first two non-zero eigenvalues and 
# respective eigenvectors
# our results differs with what Sklean outputs

from sklearn.manifold import spectral_embedding as se

# generate adjacency matrix from networkx
# scipy sparse matrix
A2 = nx.adjacency_matrix(G)

# use sklearn's implementation of spectral_embedding to calculate
# laplacian and obtain eigenvectors and eigenvalues from it
a2out = se(A2,n_components=2,drop_first=True)
print a2out.shape

# get list of nodes from networkx
# sort nodes by region, group them, add to dictionary
# create dictionary of regions and list of rows of embedded eigenvectors
a_list = []
b_list = []
c_list = []
d_list = []
for ind, i in enumerate(nodelist):
    region = int(float(i))/100
    if region == 0:
        a_list.append(ind)
    elif region == 1:
        b_list.append(ind)
    elif region == 2:
        c_list.append(ind)
    elif region == 3:
        d_list.append(ind)
        
a_region = a2out[a_list]  
b_region = a2out[b_list]  
c_region = a2out[c_list]  
d_region = a2out[d_list]  
se_regions = OrderedDict([('a', a_region),('b', b_region),('c', c_region),('d', d_region)])

# plot sklearn's spectral embedded output
connect, connect2 = plot_connectivity(se_regions)

print B

print connect

print manual

order_ind = a_list + b_list + c_list + d_list
adjacency_order = A2[np.ix_(order_ind, order_ind)]

# print [nodelist[i] for i in order_ind]

b_hat = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        row = i * 100
        col = j * 100
        reg = np.sum(adjacency_order[row:row+100, col:col+100])/100.**2
        b_hat[i,j] = reg

print b_hat

regdict = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
for i in range(4):
    R = np.argsort(b_hat[i,:])
#     print regdict[i] + " is closest to region " + str(regdict[R[2]])

B_norm = B/np.max(B)
b_hat_norm = b_hat/np.max(b_hat)

trace = Heatmap(z=B_norm)
data = [trace]
layout = Layout(title='B (normalized to 1)')
fig = Figure(data=data, layout=layout)
iplot(fig, validate=False)

trace = Heatmap(z=b_hat_norm)
data=[trace]
layout = Layout(title='B_hat (normalized to 1)')
fig = Figure(data=data, layout=layout)
iplot(fig, validate=False)

trace = Heatmap(z=manual)
data = [trace]
layout = Layout(title='Manual Spectral Embedding B_hat')
fig = Figure(data=data, layout=layout)
iplot(fig, validate=False)

trace = Heatmap(z=connect)
data = [trace]
layout = Layout(title='Sklearn Spectral Embedding B_hat')
fig = Figure(data=data, layout=layout)
iplot(fig, validate=False)

print B_norm
print ""
print b_hat_norm
print ""
print manual
print ""
print connect

B_Bhat = ((B_norm - b_hat_norm) ** 2).mean(axis=None)
B_manualBhat = ((B_norm - manual) ** 2).mean(axis=None)
B_sklearnBhat = ((B_norm - connect) ** 2).mean(axis=None)

print B_Bhat
print B_manualBhat
print B_sklearnBhat




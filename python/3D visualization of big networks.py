import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import mmap
import time

import plotly.plotly as py
from scipy import sparse as sp
from plotly.graph_objs import *

import igraph as ig

import json
import urllib2

import plotly 
plotly.tools.set_credentials_file(username='kolumbianisch', api_key='Sy6MJv8T3iEjvW2QmWrO')

data = []
req = urllib2.Request("https://raw.githubusercontent.com/plotly/datasets/master/miserables.json")
opener = urllib2.build_opener()
f = opener.open(req)
data = json.loads(f.read())

N=len(data['nodes'])

L=len(data['links'])
Edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]
print Edges
G=ig.Graph(Edges, directed=False)

labels=[]
group=[]
for node in data['nodes']:
    labels.append(node['name'])
    group.append(node['group'])
    
layt=G.layout('kk', dim=3)

Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
Yn=[layt[k][1] for k in range(N)]# y-coordinates
Zn=[layt[k][2] for k in range(N)]# z-coordinates
Xe=[]
Ye=[]
Ze=[]
for e in Edges:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    Ze+=[layt[e[0]][2],layt[e[1]][2], None]
    
print group
    
trace1=Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=Line(color='rgb(125,125,125)', width=1),
               hoverinfo='none'
               )
trace2=Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=Marker(symbol='dot',
                             size=6,
                             color=group,
                             colorscale='Viridis',
                             line=Line(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )

axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout = Layout(
         title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
         width=1000,
         height=1000,
         showlegend=False,
         scene=Scene(
         xaxis=XAxis(axis),
         yaxis=YAxis(axis),
         zaxis=ZAxis(axis),
        ),
     margin=Margin(
        t=100
    ),
    hovermode='closest',
    annotations=Annotations([
           Annotation(
           showarrow=False,
            text="Data source: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>[1]</a>",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=Font(
            size=14
            )
            )
        ]),    )

data=Data([trace1, trace2])
fig=Figure(data=data, layout=layout)

py.iplot(fig, filename='Les-Miserables')

def counter(filename):
    """
    Determines the number of nodes in the filename added
    """
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines

def loader(filename = "FB/facebook-wosn-links/out.facebook-wosn-links.txt", size = 0, timed_ = False, from_ = 1165708800, to_ = time.time()):
    """
    Inputs:
        Filename: Contains columns with first and second being source and target
            of the edge. It can contain weights and time of edge creation.
        Size: Number of lines to be read, number of edges to be taken into account.
        Timed: Are we taking into account a timed network?
        From: Epoch time of first node.
        To: Epoch time of last node.

    Outputs:
        An RXN array containing the edges of the given file. Rows are edges, co-
            lumns are source, target, weight, time respectively.

    * We are missing to do anything with the weight
    """
    # This part makes sure the entrance lines are cleared off.
    with open(filename) as myfile:
        head = [next(myfile) for x in xrange(5)]

    contador = 0
    for i in head:
        linea_head = i.strip().split(' ')
        if linea_head[0] == '%':
            contador += 1
        else:
            break

    this = counter(filename)

    # Number of Edges
    if size == False or size > this: # it makes sure that it either does the whole network or the size that you are looking for
        lines = this - contador #you can calculate the number of lines to ignore
    else:
        lines = size

    with open(filename, "r") as f:
        for i in xrange(contador):
            secon = next(f).strip().split(' ') # you are loosing your first line
        if timed_ == True:
            print True
            # clear_mat = [i.strip().split(' ') for i in f if int(i.strip().split(' ')[3])>from_ and int(i.strip().split(' ')[3]) < to_]
        else:
            clear_mat = [tuple(np.array(next(f).strip().split(' '),int)) for i in xrange(lines-contador+1)]
    matrix_array = np.array(clear_mat)
    
    with open('subelj_euroroad/ent.subelj_euroroad_euroroad.city.name.txt') as f:
        lines = f.read().splitlines()
        
    conjunto = sorted(set(matrix_array[:,0]).union(matrix_array[:,1]))
    names = [lines[i] for i in conjunto]
    
    return clear_mat, names

Edges, labels_ = loader(filename = "subelj_euroroad/out.subelj_euroroad_euroroad.txt", size = 1420)
print "edges ", np.shape(Edges)
num_nodes = np.shape(labels_)[0]
print "nodes ", num_nodes
print "loading done..."
G = ig.Graph(Edges, directed=False)
print "graph created..."
between = G.betweenness()
print "betweenness calculated..."
layt = G.layout('kk', dim=3)

Xn=[layt[k][0] for k in range(num_nodes)] # x-coordinates of nodes
Yn=[layt[k][1] for k in range(num_nodes)] # y-coordinates
Zn=[layt[k][2] for k in range(num_nodes)] # z-coordinates
Xe=[]
Ye=[]
Ze=[]

for e in Edges:
    Xe +=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye +=[layt[e[0]][1],layt[e[1]][1], None]
    Ze +=[layt[e[0]][2],layt[e[1]][2], None]

print "before trace"

trace1 = Scatter3d(x = Xe,
               y = Ye,
               z = Ze,
               mode = 'lines',
               line = Line(color='rgb(125,125,125)', width=1),
               hoverinfo='none'
               )
trace2 = Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode = 'markers',
               name='Cities',
               marker = Marker(symbol='dot',
                             size=6,
                             color=between,
                             colorscale='Viridis',
                             line=Line(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels_,
               hoverinfo='text'
               )
print "done with trace"

axis=dict(showbackground=False,
          showline=False,
          zeroline=True,
          showgrid=False,
          showticklabels=False,
          title=''
          )
print "here"
layout = Layout(
         title="Network of cities in Europe (3D visualization)",
         width=1000,
         height=1000,
         showlegend=False,
         scene=Scene(
         xaxis=XAxis(axis),
         yaxis=YAxis(axis),
         zaxis=ZAxis(axis),
        ),
     margin=Margin(
        t=100
    ),
    hovermode='closest',
    annotations=Annotations([
           Annotation(
           showarrow=False,
            text="Data source: <a href='http://konect.uni-koblenz.de/networks/subelj_euroroad'>[1]</a>",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=Font(
            size=14
            )
            )
        ]),    )

data = Data([trace1, trace2])
fig = Figure(data=data, layout=layout)

py.iplot(fig, filename='Euroroad_2000')






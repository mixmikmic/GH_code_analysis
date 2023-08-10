import networkx as nx

G=nx.Graph()#  G is an empty Graph

Nodes=range(9)
G.add_nodes_from(Nodes)
Edges=[(0,1), (0,2), (1,3), (1,4), (1,7), (2,5), (2,8), (3, 4), (3,5),(4,6), (4,7), (4,8), (5,7)]
G.add_edges_from(Edges)
G.add_edge(6,8)

pos1=nx.spring_layout(G)
print pos1

pos2=nx.spectral_layout(G)
print pos2

pos3=nx.spring_layout(G)
print pos3

import plotly.plotly as py
from plotly.graph_objs import *

def scatter_nodes(pos, labels=None, color=None, size=20, opacity=1):
    # pos is the dict of node positions
    # labels is a list  of labels of len(pos), to be displayed when hovering the mouse over the nodes
    # color is the color for nodes. When it is set as None the Plotly default color is used
    # size is the size of the dots representing the nodes
    #opacity is a value between [0,1] defining the node color opacity
    L=len(pos)
    trace = Scatter(x=[], y=[],  mode='markers', marker=Marker(size=[]))
    for k in range(L):
        trace['x'].append(pos[k][0])
        trace['y'].append(pos[k][1])
    attrib=dict(name='', text=labels , hoverinfo='text', opacity=opacity) # a dict of Plotly node attributes
    trace=dict(trace, **attrib)# concatenate the dict trace and attrib
    trace['marker']['size']=size
    if color is not None:
        trace['marker']['color']=color
    return trace       

def scatter_edges(G, pos, line_color=None, line_width=1):
    trace = Scatter(x=[], y=[], mode='lines')
    for edge in G.edges():
        trace['x'] += [pos[edge[0]][0],pos[edge[1]][0], None]
        trace['y'] += [pos[edge[0]][1],pos[edge[1]][1], None]  
        trace['hoverinfo']='none'
        trace['line']['width']=line_width
        if line_color is not None: # when it is None a default Plotly color is used
            trace['line']['color']=line_color
    return trace            

pos=nx.fruchterman_reingold_layout(G)   

labels=[str(k)  for k in range(len(pos))] # labels are  set as the nodes indices in the list of nodes
trace1=scatter_edges(G, pos)
trace2=scatter_nodes(pos, labels=labels)

width=500
height=500
axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )
layout=Layout(title= 'Fruchterman Reingold  layout',  
    font= Font(),
    showlegend=False,
    autosize=False,
    width=width,
    height=height,
    xaxis=XAxis(axis),
    yaxis=YAxis(axis),
    margin=Margin(
        l=40,
        r=40,
        b=85,
        t=100,
        pad=0,
       
    ),
    hovermode='closest',
    plot_bgcolor='#EFECEA', #set background color            
    )

data=Data([trace1, trace2])
fig = Figure(data=data, layout=layout)

def make_annotations(pos, text, font_size=14, font_color='rgb(25,25,25)'):
    L=len(pos)
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = Annotations()
    for k in range(L):
        annotations.append(
            Annotation(
                text=text[k], 
                x=pos[k][0], y=pos[k][1],
                xref='x1', yref='y1',
                font=dict(color= font_color, size=font_size),
                showarrow=False)
        )
    return annotations  

fig['layout'].update(annotations=make_annotations(pos, [str(k) for k in range(len(pos))]))  
py.sign_in('empet', '')
py.iplot(fig, filename='Network-12')

import numpy as np

Ad=np.array([[0,1,1,1,0,0,0,1], # Adjacency matrix
             [1,0,1,0,1,1,1,0],
             [1,1,0,0,0,0,1,1],
             [1,0,0,0,1,1,1,1],
             [0,1,0,1,0,1,1,0],
             [0,1,0,1,1,0,1,0],
             [0,1,1,1,1,1,0,1],
             [1,0,1,1,0,0,1,0]], dtype=float)
Gr=nx.from_numpy_matrix(Ad)

position=nx.spring_layout(Gr)

labels=['Fred', 'Alice', 'Bob', 'Anna', 'Frida', 'Andrew', 'Jack', 'Maria']

traceE=scatter_edges(Gr, position)
traceN=scatter_nodes(position, labels=labels)

layout.update(title='Friendship Network')
data1=Data([traceE, traceN])
fig = Figure(data=data1, layout=layout)
fig['layout'].update(annotations=make_annotations(position, [str(k) for k in range(len(position))]))  
py.iplot(fig, filename='Network-2N')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


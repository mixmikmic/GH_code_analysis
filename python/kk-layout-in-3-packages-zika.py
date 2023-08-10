import igraph as ig

G=ig.Graph.Read_GML('zika.gml')

N=len(G.vs)
V=range(N)

N

E=[e.tuple for e in G.es]# list of edges

labels=[v['name'] for v in G.vs]

layt=G.layout('kk') #kamada-kawai

from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
init_notebook_mode(connected=True)

def get_plotly_data(E, coords):
    # E is the list of tuples representing the graph edges
    # coords is the list of node coordinates 
    N=len(coords)
    Xnodes=[coords[k][0] for k in range(N)]# x-coordinates of nodes
    Ynodes=[coords[k][1] for k in range(N)]# y-coordnates of nodes

    Xedges=[]
    Yedges=[]
    for e in E:
        Xedges.extend([coords[e[0]][0], coords[e[1]][0], None])# x coordinates of the nodes defining the edge e
        Yedges.extend([coords[e[0]][1], coords[e[1]][1], None])# y - " - 
        
    return Xnodes, Ynodes, Xedges, Yedges 

def get_node_trace(x, y, labels, marker_size=5, marker_color='#6959CD', line_color='rgb(50,50,50)', line_width=0.5):
    return dict(type='scatter',
                x=x,
                y=y,
                mode='markers',
                marker=dict(symbol='dot',
                            size=marker_size, 
                            color=marker_color,
                            line=dict(color=line_color, width=line_width)
                             ),
            text=labels,
            hoverinfo='text'
               )
def get_edge_trace(x, y, linecolor='rgb(210,210,210)', linewidth=1):
    return dict(type='scatter',
                x=x,
                y=y,
                mode='lines',
                line=dict(color=linecolor, width=linewidth),
                hoverinfo='none'
               )
    

Xn, Yn, Xe, Ye=get_plotly_data(E, layt)
trace1=get_edge_trace(Xe, Ye)
trace2=get_node_trace(Xn, Yn, labels)

axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )

title1= "Unrooted tree"
width=800
height=800
layout=dict(title=title1,
            font= dict(size=12),
            showlegend=False,
            autosize=False,
            width=width,
            height=height,
            xaxis=dict(axis),
            yaxis=dict(axis),          
            hovermode='closest',
            annotations=([dict(showarrow=False, 
                               text='igraph Kamada-Kawai layout',  
                               xref='paper',     
                               yref='paper',     
                               x=0,  
                               y=-0.1,  
                               xanchor='left',   
                               yanchor='bottom',  
                               font=dict(size=14)     
                              )
                        ]),           
            )

fig1=dict(data=[trace1, trace2], layout=layout)
iplot(fig1)

len(G.clusters( mode='weak'))

import pygraphviz

import pygraphviz as pgv

g=pgv.AGraph(strict=True, directed=False)
g.add_nodes_from(V)
g.add_edges_from(E)
g.layout(prog='neato') 

def get_node_coordinates(g):
    #convert the string containing the node coordinates to a list of two floats
    #returns the list of lists of node coordinates
    if  not isinstance(g, pgv.AGraph):
        raise ValueError('The graph g must be a pygraphviz AGraph')
    N=len(g.nodes())    
    coords=[]
    for k in range(N):
        s=g.get_node(k).attr['pos']
        t=s.split(",")
        coords.append(map(float, t))
    return coords  

pgvlayt=get_node_coordinates(g)

Xnode, Ynode, Xedge, Yedge=get_plotly_data(E, pgvlayt)
trace3=get_edge_trace(Xedge, Yedge)
trace4=get_node_trace(Xnode, Ynode, labels)

annot2="Graphviz neato layout"

fig2=dict(data=[trace3, trace4], layout=layout)
fig2['layout']['annotations'][0]['text']=annot2
iplot(fig2)

import networkx as nx

H=nx.Graph()
H.add_nodes_from(V)
H.add_edges_from(E)

pos=nx.kamada_kawai_layout(H) 

nodes=pos.keys()

Xv=[pos[node][0] for node in nodes]
Yv=[pos[node][1] for node in nodes]
Xed=[]
Yed=[]
for edge in H.edges():
    Xed.extend([pos[edge[0]][0],pos[edge[1]][0], None])
    Yed.extend([pos[edge[0]][1],pos[edge[1]][1], None]) 

trace5=get_edge_trace(Xed, Yed)
trace6=get_node_trace(Xv, Yv, labels)

annot3="Networkx Kamada-Kawai layout"

fig3=dict(data=[trace5, trace6], layout=layout)
fig3['layout'].update(width=800, height=800)
fig3['layout']['annotations'][0]['text']=annot3
iplot(fig3)

print nx.number_connected_components(H)

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()




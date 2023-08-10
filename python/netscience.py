import igraph as ig
print ig.__version__

G=ig.Graph.Read_GML('netscience.gml')

labels=list(G.vs['label'])

N=len(labels)
N

labels[:10]

E=[e.tuple for e in G.es]# list of edges
len(E)

layt=G.layout('kk') #kamada-kawai layout

type(layt)

layt[:4]

import plotly.plotly as py
from plotly.graph_objs import *

Xn=[layt[k][0] for k in range(N)]
Yn=[layt[k][1] for k in range(N)]
Xe=[]
Ye=[]
for e in E:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]
    Ye+=[layt[e[0]][1],layt[e[1]][1], None] 

trace1=Scatter(x=Xe,
               y=Ye,
               mode='lines',
               line=Line(color='rgb(210,210,210)', width=1),
               hoverinfo='none'
               )
trace2=Scatter(x=Xn,
               y=Yn,
               mode='markers',
               name='ntw',
               marker=Marker(symbol='dot',
                             size=5, 
                             color='#6959CD',
                             line=Line(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )

axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )

width=800
height=800
layout=Layout(title= "Coauthorship network of scientists working on network theory and experiment"+              "<br> Data source: <a href='https://networkdata.ics.uci.edu/data.php?id=11'> [1]</a>",  
    font= Font(size=12),
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
    ),
    hovermode='closest',
    annotations=Annotations([
           Annotation(
           showarrow=False, 
            text='This igraph.Graph has the Kamada-Kawai layout',  
            xref='paper',     
            yref='paper',     
            x=0,  
            y=-0.1,  
            xanchor='left',   
            yanchor='bottom',  
            font=Font(
            size=14 
            )     
            )
        ]),           
    )

data=Data([trace1, trace2])
py.sign_in('empet', 'my_api_key')
fig=Figure(data=data, layout=layout)
py.plot(fig, filename='Coautorship-network-igraph') 

import networkx as nx

nx.__version__

G=nx.read_gml('netscience.gml')

V=range(N)# list of vertices

g=nx.Graph()
g.add_nodes_from(V)
g.add_edges_from(E)# E is the list of edges

pos=nx.fruchterman_reingold_layout(g) 

type(pos)

pos[0]

Xv=[pos[k][0] for k in range(N)]
Yv=[pos[k][1] for k in range(N)]
Xed=[]
Yed=[]
for edge in E:
    Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]
    Yed+=[pos[edge[0]][1],pos[edge[1]][1], None] 

trace3=Scatter(x=Xed,
               y=Yed,
               mode='lines',
               line=Line(color='rgb(210,210,210)', width=1),
               hoverinfo='none'
               )
trace4=Scatter(x=Xv,
               y=Yv,
               mode='markers',
               name='net',
               marker=Marker(symbol='dot',
                             size=5, 
                             color='#6959CD',
                             line=Line(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )

annot="This networkx.Graph has the Fruchterman-Reingold layout<br>Code:"+"<a href='http://nbviewer.ipython.org/gist/empet/07ea33b2e4e0b84193bd'> [2]</a>"

data1=Data([trace3, trace4])
fig1=Figure(data=data1, layout=layout)
fig1['layout']['annotations'][0]['text']=annot
py.plot(fig1, filename='Coautorship-network-nx')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


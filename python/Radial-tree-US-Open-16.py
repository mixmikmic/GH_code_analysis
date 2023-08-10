import networkx as nx
import pygraphviz as pgv
import pandas as pd
import numpy as np

df = pd.read_excel("Data/US-Open-2016.xls")
df.ix[:6, :]#print tree levels 0, 1, 2 

N=len(df)
N

labels=list(df['name'])

G=nx.balanced_tree(2, 6)
V=G.nodes()
E=G.edges()

H=pgv.AGraph(strict=True, directed=False)
H.add_nodes_from(V)
H.add_edges_from(E)
H.layout(prog='twopi') 

def position(g):
    if  not isinstance(g, pgv.AGraph):
        raise ValueError('The graph g must be a pygraphviz AGraph')
    N=len(g.nodes())    
    pos=[]
    for k in range(N):
        s=g.get_node(k).attr['pos']
        t=s.split(",")
        pos.append(map(float, t))
    return pos    

import plotly.plotly as py
from plotly.graph_objs import *

def plotly_graph(E, pos):
    # E is the list of tuples representing the graph edges
    # pos is the list of node coordinates 
    N=len(pos)
    Xn=[pos[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[pos[k][1] for k in range(N)]# y-coordnates of nodes

    Xe=[]
    Ye=[]
    for e in E:
        Xe+=[pos[e[0]][0],pos[e[1]][0], None]# x coordinates of the nodes defining the edge e
        Ye+=[pos[e[0]][1],pos[e[1]][1], None]# y - " - 
        
    return Xn, Yn, Xe, Ye    

pos=position(H)
#Rotate position with pi/2 anti-clockwise
pos=np.array(pos)
pos[:,[0, 1]] = pos[:,[1, 0]]
pos[:,0]=-pos[:,0]

Xn, Yn, Xe, Ye=plotly_graph(E, pos)

edges=Scatter(x=Xe,
               y=Ye, 
               mode='lines',
               line=Line(color='rgb(160,160,160)', width=0.75),
               hoverinfo='none'
               )
nodes=Scatter(x=Xn, 
               y=Yn,
               mode='markers',
               name='',
               marker=Marker(symbol='dot',
                             size=8, 
                             color='#85b6b6', 
                             line=Line(color='rgb(100,100,100)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )

Kerber_path=[0, 2, 6, 14, 30, 62, 126]
Pliskova_path=[1, 4, 10, 21, 43, 87]
colorKP=['#CC0000']*len(Kerber_path)+['rgb(65, 64, 123)']*len(Pliskova_path)# set color for both paths

XK=[pos[k][0] for k in Kerber_path]
YK=[pos[k][1] for k in Kerber_path]
XP=[pos[k][0] for k in Pliskova_path]
YP=[pos[k][1] for k in Pliskova_path]

finalists=Scatter(x=XK+XP, 
                  y=YK+YP, 
                  mode='markers', 
                  marker=dict(symbol='dot',
                              size=10, 
                              color=colorKP, 
                              line=Line(color='rgb(100,100,100)', width=0.5),                             
                             ),
                  text=['Kerber']*len(Kerber_path)+ ['Pliskova']*len(Pliskova_path), 
                  hoverinfo='text'  
                 )

def set_annotation(x, y, anno_text,  textangle, fontsize=11, color='rgb(10,10,10)'): 
  
    return Annotation(x= x,  
                      y= y,       
                      text= anno_text,      
                      textangle=textangle,#angle with horizontal line through (x,y), in degrees;
                                          #+ =clockwise, -=anti-clockwise
                      font= Font(size=fontsize, color=color),  
                      showarrow=False     
                     ) 

axis=dict(showbackground=False,
          showline=False,  
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )

layout = Layout(
         title="U.S. Open 2016<br>Radial binary tree associated to women's singles  players", 
         font=dict(family='Balto'),
         width=650,
         height=650,
         showlegend=False,
         xaxis=XAxis(axis),
         yaxis=YAxis(axis), 
         margin=Margin(t=100),
         hovermode='closest',
                )

center=np.array([(np.min(pos[63:, 0])+np.max(pos[63:, 0]))/2, (np.min(pos[63:, 1])+np.max(pos[63:, 1]))/2])
radius=np.linalg.norm(pos[63,:]-center)

angles=[]
for k in range(63, 127):
    v=pos[k,:]-center
    angles.append(-(180*np.arctan(v[1]/v[0])/np.pi))

pos_text=center+1.2*(pos[63:, :]-center)# text position

#define annotations for non-finalist players
for k in range(63, 87):
    layout['annotations']+=[set_annotation(pos_text[k-63][0], pos_text[k-63][1], labels[k],  angles[k-63])]
for k in range(88, 126):
    layout['annotations']+=[set_annotation(pos_text[k-63][0], pos_text[k-63][1], labels[k],  angles[k-63])]    

#insert colored annotations for the finalists, Pliskova and Kerber
layout['annotations']+=[set_annotation(pos_text[87-63][0], pos_text[87-63][1], 
                                       '<b>Pliskova</b>', 
                                       angles[87-63], 
                                       color='rgb(65, 64, 123)'),
                         set_annotation(pos_text[126-63][0], pos_text[126-63][1], 
                                        '<b>Kerber</b>',  
                                        angles[126-63], 
                                        color='#CC0000')]
layout['annotations']+= [set_annotation(center[0]-0.15, center[1]+45, 
                                        '<b>Winner<br>A. Kerber</b>', 
                                        0, fontsize=12, color='#CC0000')]

data_anno_text="Data source: "+               "<a href='http://www.wtatennis.com/SEWTATour-Archive/posting/2016/905/MDS.pdf'> [1] </a>,"+               " Excel file: "+                "<a href='https://github.com/empet/Plotly-plots/blob/master/Data/US-Open-2016.xls'> [2] </a>"
layout['annotations'].append(Annotation(
            showarrow=False, 
            text=data_anno_text,  
            xref='paper',     
            yref='paper',     
            x=0,  
            y=-0.1,  
            xanchor='left',   
            yanchor='bottom',  
            font=Font(
            size=12 )
            ))

data=Data([edges, nodes, finalists])
fig=Figure(data=data, layout=layout)

py.sign_in('empet', 'my_api_key')
py.iplot(fig, filename='US-Open-16')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


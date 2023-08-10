import igraph as ig

import json

data = []
with open('miserables.json') as f: 
    for line in f:
        data.append(json.loads(line))        

data=data[0]
data

print data.keys()

N=len(data['nodes'])
N

L=len(data['links'])
Edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]

G=ig.Graph(Edges, directed=False)

data['nodes'][0]

labels=[]
group=[]
for node in data['nodes']:
    labels.append(node['name'])
    group.append(node['group'])

layt=G.layout('kk', dim=3) 

layt[5]

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

import plotly.plotly as py
from plotly.graph_objs import *

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
py.sign_in('empet', 'jkxft90od0')
fig=Figure(data=data, layout=layout)

py.plot(fig, filename='Les-Miserables')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


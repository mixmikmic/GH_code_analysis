from igraph import *

G=read('facebook_combined.txt', directed=False)

E=[e.tuple for e in G.es]# list of edges
print 'number of edges: ', len(E)

N=len(G.vs)# number of vertices
print 'number of vertices:', N

layt=G.layout('fr')  #set graph layout; 'fr'=Fruchterman-Reingold force-directed algorithm

Xn=[layt[k][0] for k in range(N)]
Yn=[layt[k][1] for k in range(N)]
Xe=[]
Ye=[]
for edge in E:
    Xe+=[layt[edge[0]][0],layt[edge[1]][0], None]
    Ye+=[layt[edge[0]][1],layt[edge[1]][1], None]  

custom_cs=[[0.0, '#FFFFE0'],
 [0.06666666666666667, '#FFEEC1'],
 [0.13333333333333333, '#FFDEA7'],
 [0.2, '#FFCB91'],
 [0.26666666666666666, '#FFB880'],
 [0.3333333333333333, '#FFA474'],
 [0.4, '#FE906A'],
 [0.4666666666666667, '#F87D64'],
 [0.5333333333333333, '#F06A5E'],
 [0.6, '#E75758'],
 [0.6666666666666666, '#DB4551'],
 [0.7333333333333333, '#CF3447'],
 [0.8, '#C0223B'],
 [0.8666666666666667, '#B0122C'],
 [0.9333333333333333, '#9E051B'],
 [1.0, '#8B0000']]

labels=range(N)#  labels are displayed when hovering the mouse over vertices(nodes)

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls 

trace1=Scatter(x=Xe,
               y=Ye,
               mode='lines',
               line=Line(color='rgb(210,210,210)', width=0.5),
               hoverinfo='none'
               )
trace2=Scatter(x=Xn,
               y=Yn,
               mode='markers',
               name='Fb',
               marker=Marker(symbol='dot',
                             size=5, 
                             color='#C0223B',
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

title1= "Facebook network<br> Data:"+       "<a href='https://snap.stanford.edu/data/egonets-Facebook.html'> [1]</a>"

width=700
height=700
#Plotly plot layout!  Don't confuse with Graph.layout() in igraph 
layout=Layout(title=title1,  
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
    )

data=Data([trace1, trace2])
py.sign_in('empet', 'my_api_key')
fig=Figure(data=data, layout=layout)

py.plot(fig, filename='Facebook-igraph', world_readable=False)

community = G.community_multilevel()

for c in community:
    print len(c)

community_color={vertex_index: len(component)  for component in community for vertex_index in component}

trace3=Scatter(x=Xn,
               y=Yn,
               mode='markers',
               name='Fb-comm',
               marker=Marker(symbol='dot',
                             size=5, 
                             color=community_color.values(),
                             colorscale=custom_cs,
                             line=Line(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )

title2= "Communities in a Facebook network, defined as an igraph Graph<br> Data:"+       "<a href='https://snap.stanford.edu/data/egonets-Facebook.html'> [1]</a>"

data1=Data([trace1, trace3])
fig=Figure(data=data1, layout=layout)
fig['layout'].update(title=title2)
py.plot(fig, filename='Facebook-Comm-igraph', world_readable=False)

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


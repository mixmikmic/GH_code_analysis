import numpy as np
import json

url="https://raw.githubusercontent.com/evelinag/StarWars-social-network/master/"+"networks/starwars-episode-7-interactions-allCharacters.json"

import urllib2

data = []
req = urllib2.Request(url)
opener = urllib2.build_opener()
f = opener.open(req)
data = json.loads(f.read())

print data.keys()

L=len(data['nodes'])# number of nodes (characters)
data['nodes'][0]

labels=[item['name'].lower().title() for item in data['nodes']]
values=[item['value'] for item in data['nodes']]
hover_text=[labels[k]+', '+str(values[k]) +' scenes' for k in range(L)]

print data['links'][0]

Edges=[(item['source'], item['target'])  for item in data['links']]

interact_strength=[item['value'] for item in data['links']]
Keys=sorted(set(interact_strength))

widths=[0.5+k*0.25 for k in range(5)]+[2+k*0.25 for k in range(4)]+[3, 3.25, 3.75, 4.25, 5, 5.25, 7]
D=dict(zip(Keys, widths))
Widths=[D[val] for val in interact_strength]

def get_b1(b0, b2):
    b1= 0.5*(np.array(b0)+np.array(b2))+       0.5*np.array([0,1.0])*np.sqrt(3)*np.linalg.norm(np.array(b2)-np.array(b0))
    return b1.tolist()    

def dim_plus_1(b, w):#lift the points b0, b1, b2 to 3D points a0, a1, a2 (see Gallier book)
    #b is a list of 3 lists of 2D points 
    #w is a list of numbers (weights) of len equal to the len of b
    if len(b)!=len(w):
        raise ValueError('the number of weights is not equal to the nr of points')
    else:
        a=np.array([ point+[w[i]] for (i, point) in enumerate(b)])
        a[1, :2]*=w[1]
        return a

def Bezier_curve(bz, nr=75): #the control point coordinates are passed in a list bz=[bz0, bz1, bz2] 
                      
    t=np.linspace(0,1, nr)
    #for each parameter t[i] evaluate a point on the Bezier curve with the de Casteljau algorithm
    N=len(bz) 
    points=[]# the list of points to be computed on the Bezier curve
    for i in range(nr):
        aa=np.copy(bz) 
        for r in range(1,N):
            aa[:N-r,:]=(1-t[i])*aa[:N-r,:]+t[i]*aa[1:N-r+1,:]# convex combination of points
        points.append(aa[0,:])                                  
    return np.array(points) 

def Rational_Bezier_curve(a):
    C= Bezier_curve(a) 
    return [p[:2]/p[2] for p in C]

import plotly.plotly as py
from plotly.graph_objs import *

pl_density=[[0.0, 'rgb(230,240,240)'],
            [0.1, 'rgb(187,220,228)'],
            [0.2, 'rgb(149,197,226)'],
            [0.3, 'rgb(123,173,227)'],
            [0.4, 'rgb(115,144,227)'],
            [0.5, 'rgb(119,113,213)'],
            [0.6, 'rgb(120,84,186)'],
            [0.7, 'rgb(115,57,151)'],
            [0.8, 'rgb(103,35,112)'],
            [0.9, 'rgb(82,20,69)'],
            [1.0, 'rgb(54,14,36)']]

X=range(L)
Y=[0]*L
color='#6b8aca'             

trace=dict(type='scatter',
           x=X,
           y=Y,
           mode='markers',
           marker=dict(
                       size=12, 
                       color=values, 
                       colorscale=pl_density,
                       showscale=False,
                       line=dict(color='rgb(50,50,50)', width=0.75)
                            ),
          
           text=hover_text,
           hoverinfo='text'
               )

data=[]
interact=[]#list of strings to be displayed when hovering the mouse over the middle of the circle arcs
xx=[]
yy=[]

for i, (j, k) in enumerate(Edges):
    if j<k:
        interact.append('interactions('+labels[j]+', '+labels[k]+')='+'{:d}'.format(interact_strength[i]))
    else:
        interact.append('interactions('+labels[k]+', '+labels[j]+')='+'{:d}'.format(interact_strength[i]))
    b0=[X[j], 0.0]
    b2=[X[k], 0.0]
    b1=get_b1(b0, b2)
    a=dim_plus_1([b0, b1, b2], [1, 0.5, 1])
    pts= Rational_Bezier_curve(a)
    xx.append(pts[38][0])#abscissa of the middle point on the computed arc
    yy.append(pts[38][1])#ordinate of the same point
    x,y=zip(*pts)
    
    data.append(dict(type='scatter',
                     x=x, 
                     y=y, 
                     name='',
                     mode='lines', 
                     line=dict(width=Widths[i], color=color, shape='spline'),
                     hoverinfo='none'
                    )
                )

data.append(dict(type='scatter',
                 x=xx,
                 y=yy,
                 name='',
                 mode='markers',
                 marker=dict(size=0.5, color=color),
                 text=interact,
                 hoverinfo='text'))
data.append(trace)

title="Arc Diagram of Star Wars Characters that Interacted in The Force Awakens"

anno_text="Data source: "+          "<a href='https://github.com/evelinag/StarWars-social-network/tree/master/networks'> [1]</a>"

layout = dict(
         title=title, 
         font=dict(size=10), 
         width=1100,
         height=475,
         showlegend=False,
         xaxis=dict(anchor='y',
                    showline=False,  
                    zeroline=False,
                    showgrid=False,
                    tickvals=range(27), 
                    ticktext=labels,
                    tickangle=50,
                    ),
         yaxis=dict(anchor='x',
                    showline=False,  
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    ), 
         hovermode='closest',
         margin=dict(t=80, b=110, l=10, r=10),
         annotations=Annotations([Annotation(showarrow=False, 
                                             text=anno_text,
                                             xref='paper',     
                                             yref='paper',     
                                             x=0.05,  
                                             y=-0.3,  
                                             xanchor='left',   
                                             yanchor='bottom',  
                                             font=Font(size=11 )     
                                            )
                                  ])
                 
           
    )
fig=dict(data=data, layout=layout)
py.sign_in('empet', 'my_api_key') 
py.plot(fig, filename='arc-diagram-FA')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


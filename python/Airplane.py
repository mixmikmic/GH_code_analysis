from plyfile import PlyData, PlyElement
import numpy as np

import urllib2

req = urllib2.Request('http://people.sc.fsu.edu/~jburkardt/data/ply/airplane.ply')
opener = urllib2.build_opener()
f = opener.open(req)

plydata = PlyData.read(f)

for element in plydata.elements:
    print element

nr_points=plydata.elements[0].count
nr_faces=plydata.elements[1].count
print nr_points, nr_faces

points=[plydata['vertex'][k] for k in range(nr_points)]
points[:5]

points=np.array(map(list, points))
points[:5]

x,y,z=zip(*points)#

plydata['face'][0]

faces=[plydata['face'][k][0] for k in range(nr_faces)]
faces[:5]

triangles=map(lambda index: points[index], faces)

triangles[0]

Xe=[]
Ye=[]
Ze=[]
for T in triangles:
        Xe+=[T[k%3][0] for k in range(4)]+[ None]# x-coordinates of  edge ends for T
        Ye+=[T[k%3][1] for k in range(4)]+[ None]
        Ze+=[T[k%3][2] for k in range(4)]+[ None]

import plotly.plotly as py
from plotly.graph_objs import *

trace=Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=Line(color= 'rgb(130,130,130)', width=1.5)
               )

axis=dict(showbackground=False,
          showline=False,  
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )

layout = Layout(title="Airplane", 
                width=1000,
                height=1000,
                showlegend=False,
                scene=Scene(xaxis=XAxis(axis),
                            yaxis=YAxis(axis), 
                            zaxis=ZAxis(axis), 
                            aspectmode='manual',
                            aspectratio=dict(x=1, y=1, z=0.4)
                           ),
                margin=Margin(t=100),
                hovermode='closest',
                )

data=Data([trace])
py.sign_in('empet', 'my_api_key')
fig=Figure(data=data, layout=layout)
py.iplot(fig, filename='Ply-airplane')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
get_ipython().magic('matplotlib inline')

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(x, y, z, triangles=faces, cmap=plt.cm.viridis,  linewidth=0.2)

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


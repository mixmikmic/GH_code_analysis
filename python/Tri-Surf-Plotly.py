import numpy as np
import matplotlib.cm as cm
from scipy.spatial import Delaunay

u=np.linspace(0,2*np.pi, 24)
v=np.linspace(-1,1, 8)
u,v=np.meshgrid(u,v)
u=u.flatten()
v=v.flatten()

#evaluate the parameterization at the flattened u and v
tp=1+0.5*v*np.cos(u/2.)
x=tp*np.cos(u)
y=tp*np.sin(u)
z=0.5*v*np.sin(u/2.)

#define 2D points, as input data for the Delaunay triangulation of U
points2D=np.vstack([u,v]).T
tri = Delaunay(points2D)#triangulate the rectangle U

def map_z2color(zval, colormap, vmin, vmax):
    #map the normalized value val to a corresponding color in the mpl colormap
    
    if vmin>=vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t=(zval-vmin)/float((vmax-vmin))#normalize val
    C=map(np.uint8, np.array(colormap(t)[:3])*255)
    #convert to a Plotly color code:
    return 'rgb'+str((C[0], C[1], C[2]))
       

def mpl_to_plotly(cmap, pl_entries):
    h=1.0/(pl_entries-1)
    pl_colorscale=[]
    for k in range(pl_entries):
        C=map(np.uint8, np.array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale

import plotly.plotly as py
from plotly.graph_objs import *
import plotly
plotly.offline.init_notebook_mode() 

def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, showcolorbar=False, plot_edges=None):
    #x, y, z are lists of coordinates of the triangle vertices 
    #simplices are the simplices that define the triangulation;
    #simplices  is a numpy array of shape (no_triangles, 3)
    #insert here the  type check for input data
    
    points3D=np.vstack((x,y,z)).T
    tri_vertices= points3D[simplices]# vertices of the surface triangles  
    zmean=tri_vertices[:, :, 2].mean(-1)# mean values of z-coordinates of the
                                        #triangle vertices
      
    min_zmean, max_zmean=np.min(zmean), np.max(zmean)
    
    facecolor=[map_z2color(zz,  colormap, min_zmean, max_zmean) for zz in zmean] 
    I,J,K=zip(*simplices)
    
    triangles=Mesh3d(x=x,
                     y=y,
                     z=z,
                     facecolor=facecolor, 
                     i=I,
                     j=J,
                     k=K,
                     name=''
                    )
    
    if showcolorbar==True:
        pl_colorsc=mpl_to_plotly(colormap,11)
        # define a fake Scatter3d trace in order to enable displaying the colorbar for the trisurf
        
        colorbar=Scatter3d(x=x[:2],
                           y=y[:2],
                           z=z[:2],
                           mode='markers',
                           marker=dict(size=0.1, color=[min_zmean, max_zmean], 
                                      colorscale=pl_colorsc, showscale=True),
                             hoverinfo='None'
                          )
    
    
    if plot_edges is None: # the triangle sides are not plotted 
        if  showcolorbar is True:
            return Data([colorbar, triangles])
        else: 
            return  Data([triangles])
    else:#plot edges
        #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        #None separates data corresponding to two consecutive triangles
        Xe=[]
        Ye=[]
        Ze=[]
        for T in tri_vertices:
            Xe+=[T[k%3][0] for k in range(4)]+[ None]
            Ye+=[T[k%3][1] for k in range(4)]+[ None]
            Ze+=[T[k%3][2] for k in range(4)]+[ None]
       
        #define the lines to be plotted
        lines=Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        line=Line(color= 'rgb(50,50,50)', width=1.5)
               )
        if  showcolorbar is True:
            return Data([colorbar, triangles, lines])
        else: 
            
            return Data([triangles, lines])

data1=plotly_trisurf(x,y,z, tri.simplices, colormap=cm.RdBu, showcolorbar=True, plot_edges=True)

axis = dict(
showbackground=True, 
backgroundcolor="rgb(230, 230,230)",
gridcolor="rgb(255, 255, 255)",      
zerolinecolor="rgb(255, 255, 255)",  
    )

layout = Layout(
         title='Moebius band triangulation',
         width=800,
         height=800,
         showlegend=False,
         scene=Scene(xaxis=XAxis(axis),
                     yaxis=YAxis(axis), 
                     zaxis=ZAxis(axis), 
                     aspectratio=dict(x=1,
                                      y=1,
                                      z=0.5
                                     ),
                    )
        )

fig1 = Figure(data=data1, layout=layout)


plotly.offline.iplot(fig1)

n=12# number of radii
h=1.0/(n-1)
r = np.linspace(h, 1.0, n)
theta= np.linspace(0, 2*np.pi, 36)

r,theta=np.meshgrid(r,theta)
r=r.flatten()
theta=theta.flatten()

#Convert polar coordinates to cartesian coordinates (x,y)
x=r*np.cos(theta)
y=r*np.sin(theta)
x=np.append(x, 0)#  a trick to include the center of the disk in the set of points. It was avoided
                 # initially when we defined r=np.linspace(h, 1.0, n)
y=np.append(y,0)
z = np.sin(-x*y) 

points2D=np.vstack([x,y]).T
tri=Delaunay(points2D)

data2=plotly_trisurf(x,y,z, tri.simplices, colormap=cm.cubehelix, showcolorbar=True, plot_edges=None)
fig2 = Figure(data=data2, layout=layout)
fig2['layout'].update(dict(title='Triangulated surface',
                          scene=dict(camera=dict(eye=dict(x=1.75, 
                                                          y=-0.7, 
                                                          z= 0.75)
                                                )
                                    )))

plotly.offline.iplot(fig2)

from plyfile import PlyData, PlyElement

import urllib2
req = urllib2.Request('http://people.sc.fsu.edu/~jburkardt/data/ply/chopper.ply') 
opener = urllib2.build_opener()
f = opener.open(req)
plydata = PlyData.read(f)

for element in plydata.elements:
    print element

nr_points=plydata.elements[0].count
nr_faces=plydata.elements[1].count

points=np.array([plydata['vertex'][k] for k in range(nr_points)])
x,y,z=zip(*points)

faces=[plydata['face'][k][0] for k in range(nr_faces)]
faces[0]

data3=plotly_trisurf(x,y,z, faces, colormap=cm.RdBu, plot_edges=None)

title="Trisurf from a PLY file<br>"+                "Data Source:<a href='http://people.sc.fsu.edu/~jburkardt/data/ply/airplane.ply'> [1]</a>"

noaxis=dict(showbackground=False,
            showline=False,  
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='' 
          )

fig3 = Figure(data=data3, layout=layout)
fig3['layout'].update(dict(title=title,
                           width=1000,
                           height=1000,
                           scene=dict(xaxis=noaxis,
                                      yaxis=noaxis, 
                                      zaxis=noaxis, 
                                      aspectratio=dict(x=1, y=1, z=0.4),
                                      camera=dict(eye=dict(x=1.25, y=1.25, z= 1.25)     
                                     )
                           )
                     ))
                      

plotly.offline.iplot(fig3)

from skimage import measure
import cmocean
X,Y,Z = np.mgrid[-2:2:40j, -2:2:40j, -2:2:40j]
F = X**4 + Y**4 + Z**4 - (X**2+Y**2+Z**2)**2 + 3*(X**2+Y**2+Z**2) - 3  
vertices, simplices = measure.marching_cubes(F, 0)
x,y,z = zip(*vertices) 

data4=plotly_trisurf(x,y,z, simplices, colormap=cmocean.cm.waveheight_r, showcolorbar=True)
fig4 = Figure(data=data4, layout=layout)
fig4['layout'].update(dict(title='Isosurface',
                          scene=dict(camera=dict(eye=dict(x=1, 
                                                          y=1, 
                                                          z=1)
                                                ),
                                      aspectratio=dict(x=1, y=1, z=1)
                                    )))
plotly.offline.iplot(fig4)

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


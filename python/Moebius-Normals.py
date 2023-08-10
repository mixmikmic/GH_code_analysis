import numpy as np
import plotly.plotly as py

def arrow3d(headsize, theta):
    r=headsize*np.tan(theta)
    u=np.linspace(0,2*np.pi, 60)
    v=np.linspace(0, 1, 15)
    U,V=np.meshgrid(u,v)
    #parameterization of the standard cone 
    x=r*V*np.cos(U)
    y=r*V*np.sin(U)
    z=headsize*(1-V)
    cone=np.stack((x,y,z)) #shape(3, m, n)
    w=np.linspace(0, r, 10)
    u, w=np.meshgrid(u,w)
    #parameterization of the base disk
    xx=w*np.cos(u)
    yy=w*np.sin(u)
    zz=np.zeros(w.shape)
    disk=np.stack((xx,yy,zz))
    return cone, disk

def place_arrow3d(start, end, headsize, theta):
    #Move the standard arrow to a position in the 3d space, 
    #that is  computed from the inputted data
    
    #start = array of shape (3,) = the starting point of the arrow's support line
    #end = array of shape(3, ) = the end point of the segment of line
    #headsize
    #theta=the angle between the symmetry axis and a generatrice
    
    epsilon=1.0e-04 # any coordinate less than epsilon is considered 0
    
    cone, disk=arrow3d(headsize, theta)#get the standard cone
    arr_dir=end-start# the arrow direction
    if np.linalg.norm(arr_dir) > epsilon:
        #define a right orthonormal basis (u1, u2, u3), with u3 the unit vector of the arrow_dir
        u3=arr_dir/np.linalg.norm(arr_dir)
        origin=end-headsize * u3 #the point where the arrow starts on the supp line
        a, b, c = u3
        if abs(a) > epsilon or abs(b) > epsilon:
            v1=np.array([-b, a, 0])# v1 orthogonal to u3
            u1=v1/np.linalg.norm(v1)
        else: 
            u1=np.array([1., 0,  0])
        u2=np.cross(u3, u1)# this def ensures that the orthonormal basis is a right one
        T=np.vstack((u1, u2, u3)).T   #Transformation T, T(e_i)=u_i, to be applied to the standard cone 
        cone=np.einsum('ji, imn -> jmn', T, cone)#Transform the standard cone
        disk=np.einsum('ji, imn -> jmn', T, disk)#Transform the cone base
        cone=np.apply_along_axis(lambda a, v: a+v, 0, cone, origin)#translate the cone; 
                                                                   #dir translation, v=vec(O,origin)
        disk=np.apply_along_axis(lambda a, v: a+v, 0, disk, origin)# translate the cone base
        return  origin, cone, disk 
    
    else:  return (0, )
    

axis = dict(
showbackground=True, 
backgroundcolor="rgb(230, 230,230)",
gridcolor="rgb(255, 255, 255)",      
zerolinecolor="rgb(255, 255, 255)",  
    )

layout = dict(title='<br>A vector field along the central circle of the Moebius strip',
              font=dict(family='Balto'),
              autosize=False,
              width=700,
              height=700,
              showlegend=False,
              scene=dict(camera = dict(eye=dict(x=1.25, y=1.25, z=0.55)),
                         aspectratio=dict(x=1, y=1, z=0.5),
                         xaxis=axis,
                         yaxis=axis, 
                         zaxis=dict(axis, **{'tickvals':[-0.6, -0.3, 0, 0.3, 0.6]}),
                        )
               )

pl_balance=[[0.0, 'rgb(23, 28, 66)'], #  cmocean
            [0.1, 'rgb(41, 61, 150)'],
            [0.2, 'rgb(21, 112, 187)'],
            [0.3, 'rgb(89, 155, 186)'],
            [0.4, 'rgb(169, 194, 202)'],
            [0.5, 'rgb(240, 236, 235)'],
            [0.6, 'rgb(219, 177, 163)'],
            [0.7, 'rgb(201, 118, 90)'],
            [0.8, 'rgb(179, 56, 38)'],
            [0.9, 'rgb(125, 13, 41)'],
            [1.0, 'rgb(60, 9, 17)']
           ]

u=np.linspace(0, 2*np.pi, 36)
v=np.linspace(-0.5, 0.5, 10)
u,v=np.meshgrid(u,v)
tp=1+v*np.cos(u/2.)
x=tp*np.cos(u)
y=tp*np.sin(u)
z=v*np.sin(u/2.)
moebius=dict(type='surface',
            x=x,
            y=y,
            z=z,
            colorscale=pl_balance,
            showscale=True,
            colorbar=dict(thickness=20, lenmode='fraction', len=0.75, ticklen=4))

pl_c=[[0.0, 'rgb(179, 56, 38)'],
      [1.0, 'rgb(179, 56, 38)']]

def get_normals(start, origin, cone, disk, colorscale=pl_c):
    tr_cone=dict(type='surface', 
                 x=cone[0,:,:],
                 y=cone[1,:,:],
                 z=cone[2,:,:],
                 colorscale=colorscale,
                 showscale=False)
    tr_disk=dict(type='surface',
                 x=disk[0,:,:],
                 y=disk[1,:,:],
                 z=disk[2,:,:],
                 colorscale=colorscale,
                 showscale=False)
    tr_line=dict(type='scatter3d',
                 x=[start[0], origin[0]],
                 y=[start[1], origin[1]],
                 z=[start[2], origin[2]],
                 name='', 
                 mode='lines',
                 line=dict(width=3, color='rgb(60, 9, 17)')
                 )
    return [tr_line, tr_cone, tr_disk]#return a list that is concatenated to data            
                

u=np.linspace(0, 2*np.pi, 36)
xx=np.cos(u)
yy=np.sin(u)
zz=np.zeros(xx.shape)
starters=np.vstack((xx,yy,zz)).T
a=0.35
#Normal coordinates
Nx=2*np.cos(u)*np.sin(u/2)
Ny=np.cos(u/2)-np.cos(3*u/2)
Nz=-2*np.cos(u)
ends=starters+a*np.vstack((Nx,Ny, Nz)).T

data=[moebius]

for j in range(ends.shape[0]):
    arr=place_arrow3d(starters[j], ends[j], 0.15, np.pi/15)
    if len(arr)==3:# get normals at the regular points on a surface, i.e. where ||Normalvector|| not = 0
        data+=get_normals(starters[j], arr[0], arr[1], arr[2])

fig=dict(data=data, layout=layout)
py.sign_in('empet','api_key' )
py.iplot(fig, filename='normalsMob')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()


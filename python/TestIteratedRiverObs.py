# This is for making changes on the fly

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('pylab', 'inline')

import numpy as np
from numpy.random import normal
from Centerline import Centerline
from RiverObs import IteratedRiverObs

nobs = 5

# This is the truth

npoints = 100
x = np.arange(npoints)*2*np.pi/npoints
y = np.sin(x)
width = 0.2 + 0.1*x

xobs = []
yobs = []
for i in range(npoints):
    dx = normal(0.,width[i],nobs)
    dy = normal(0.,width[i],nobs)
    xobs += (x[i]+dx).tolist()
    yobs += (y[i]+dy).tolist()

# This is the first guess

nc = 25
xc = np.arange(nc)*2*np.pi/nc
yc = -1*np.ones(nc) + 0.5*xc

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.5,linewidth=2)

class reach: pass
reach.x = xc
reach.y = yc

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=1
max_iter=1
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=1
max_iter=1
smooth=1
weights=True
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2,
                  smooth=smooth,weights=weights)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=1
max_iter=2
smooth=1
weights=True
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2,
                  smooth=smooth,weights=weights)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=0.2
max_iter=10
smooth=1
weights=True
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2,
                  smooth=smooth,weights=weights)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=1
max_iter=1
smooth=1.e-1
weights=True
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2,
                  smooth=smooth,weights=weights)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=1
max_iter=2
smooth=1.e-1
weights=True
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2,
                  smooth=smooth,weights=weights)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=1
max_iter=1
smooth=1.e-2
weights=True
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2,
                  smooth=smooth,weights=weights)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=1
max_iter=2
smooth=1.e-2
weights=True
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2,
                  smooth=smooth,weights=weights)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs = IteratedRiverObs(reach,xobs,yobs)
alpha=1
max_iter=2
smooth=1.e-3
weights=True
river_obs.iterate(max_iter=max_iter,alpha=alpha,tol=1.e-2,
                  smooth=smooth,weights=weights)
xc,yc = river_obs.get_centerline_xy()

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'o',alpha=0.8,linewidth=2)
plot(x,y,'--k')

river_obs.add_centerline_obs(x,y,width,'width')

xw = river_obs.centerline_obs['width'].x
yw = river_obs.centerline_obs['width'].y
w = river_obs.centerline_obs['width'].v

plot(xobs,yobs,'.',alpha=0.1)
plot(xc,yc,'kx',alpha=1,linewidth=2)
scatter(xw,yw,c=w,s=50,alpha=1,edgecolor='none')




# This is for making changes on the fly

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('pylab', 'inline')

import numpy as np
from Centerline import Centerline

npoints = 10
x = np.arange(npoints)*2*np.pi/npoints
y = np.sin(x)

cl = Centerline(x,y)

x0 = x[5]*0.5 + x[6]*0.5
y0 = np.sin(x0) + 0.2
x1 = x0
y1 = np.sin(x0) - 0.2

xx = [x0,x1]
yy = [y0,y1]

i,d,xcl,ycl,s,n = cl(xx,yy)

print(i,d,s,n)

plot(x,y,'o')
plot(xx,yy,'rx')
plot(xcl,ycl,'ro')
plot(x[i],y[i],'gx')

cl2 = Centerline(x,y,ds=0.1)

i,d,xcl,ycl,s,n = cl2(xx,yy)

plot(cl2.x,cl2.y,'o')
plot(xx,yy,'rx')
plot(xcl,ycl,'ro')
plot(cl2.x[i],cl2.y[i],'gx')

width = 0.5 + 0.25*y
clw = Centerline(x,y,obs=[width],obs_names=['width'])

plot(clw.x,clw.y,'o')
plot(clw.x,clw.y+clw.width,'-k',alpha=0.5)
plot(clw.x,clw.y-clw.width,'-k',alpha=0.5)

width = 0.5 + 0.25*y
clw2 = Centerline(x,y,ds=0.1,obs=[width],obs_names=['width'])

plot(clw2.x,clw2.y,'o')
plot(clw2.x,clw2.y+clw2.width,'-k',alpha=0.5)
plot(clw2.x,clw2.y-clw2.width,'-k',alpha=0.5)


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import pi,linspace,cos,sin
from matplotlib.pyplot import plot,axis,title

t = linspace(0,pi,1000)
xx = cos(t)
yy = sin(t)
plot(xx,yy)

N = 10
theta = linspace(0,pi,N)
plot(cos(theta),sin(theta),'o')
for i in range(N):
    x1 = [cos(theta[i]), cos(theta[i])]
    y1 = [0.0, sin(theta[i])]
    plot(x1,y1,'k--',cos(theta[i]),0,'sr')
axis([-1.1, 1.1, 0.0, 1.1])
axis('equal')
title(str(N)+' Chebyshev points');


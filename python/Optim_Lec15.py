from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
x,y = np.meshgrid(np.linspace(0,6,200),np.linspace(0,3,200))
alpha = 0.95
beta = 0.95
z = x**alpha*y**beta
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(x, y, z, 
                rstride=5, 
                cstride=5, 
                alpha=0.3,            # transparency of the surface 
                cmap=cm.BuPu)         # colour map

ax.set_xlabel('$x$',fontsize=15)
ax.set_xlim(0, 6)
ax.set_ylabel('$y$',fontsize=15)
ax.set_ylim(0, 3)
ax.set_zlabel('$f(x,y)$',fontsize=15)
ax.set_zlim(0, 15)

ax.set_title('The graph of $x^{\\alpha} y^{\\beta}$ for $\\alpha$ = %.2f, $\\beta$ = %.2f'%(alpha,beta), va='bottom', fontsize=15)

ax.view_init(elev=18, azim=-50)           # elevation and angle
ax.dist=12   

plt.show()

fig.savefig("NonConcF.pdf")

import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(0,2,200)
a=0.5
Y = np.exp(-a*X)-0.5
f = plt.figure()
plt.plot(X,Y)
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$f(x)$',fontsize=15)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
x1lim = 6
x2lim = 10
x1,x2 = np.meshgrid(np.linspace(0,x1lim,200),np.linspace(0,x2lim,200))
A  = .5
alpha = 2
beta = 1.1
y = A*x1**alpha*x2**beta
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(x1, x2, y, 
                rstride=5, 
                cstride=5, 
                alpha=0.02,            # transparency of the surface 
                cmap=cm.BuPu)         # colour map

cset = ax.contourf(x1, x2, y, 
                   zdir='z',          # direction of contour projection
                   offset=0,          # how "far" to render the contour map
                   cmap=cm.BuPu)      # colour map
ax.set_xlabel('$x_1$',fontsize=15)
ax.set_xlim(0, x1lim)
ax.set_ylabel('$x_2$',fontsize=15)
ax.set_ylim(0, x2lim)
ax.set_zlabel('$f(x_1,x_2)$',fontsize=15)
ax.set_zlim(0, np.max(y))

ax.view_init(elev=30, azim=-80)           # elevation and angle
ax.dist=12   

plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
x1lim = 6
x2lim = 10
x1,x2 = np.meshgrid(np.linspace(0,x1lim,200),np.linspace(0,x2lim,200))
A  = .5
alpha = 2
beta = 1.1
y = A*x1**alpha*x2**beta
fig = plt.figure(figsize=plt.figaspect(0.3))
ax = fig.add_subplot(1, 2, 1,projection='3d')         # FIRST SUBPLOT
ax.plot_surface(x1, x2, y, 
                rstride=5, 
                cstride=5, 
                alpha=0.1,            # transparency of the surface 
                cmap=cm.BuPu)         # colour map

cset = ax.contourf(x1, x2, y, 
                   zdir='z',          # direction of contour projection
                   offset=0,          # how "far" to render the contour map
                   cmap=cm.BuPu)      # colour map
ax.set_xlabel('$x_1$',fontsize=15)
ax.set_xlim(0, x1lim)
ax.set_ylabel('$x_2$',fontsize=15)
ax.set_ylim(0, x2lim)
ax.set_zlabel('$f(x_1,x_2)$',fontsize=15)
ax.set_zlim(0, np.max(y))

ax.view_init(elev=20, azim=-60)           # elevation and angle
ax.dist=10   

ax = fig.add_subplot(1, 2, 2,projection='3d')         # SECOND SUBPLOT
ax.plot_surface(x1, x2, y, 
                rstride=5, 
                cstride=5, 
                alpha=0.1,            # transparency of the surface 
                cmap=cm.BuPu)         # colour map

cset = ax.contourf(x1, x2, y, 
                   zdir='z',          # direction of contour projection
                   offset=0,          # how "far" to render the contour map
                   cmap=cm.BuPu)      # colour map
ax.set_xlabel('$x_1$',fontsize=15)
ax.set_xlim(0, x1lim)
ax.set_ylabel('$x_2$',fontsize=15)
ax.set_ylim(0, x2lim)
ax.set_zlabel('$f(x_1,x_2)$',fontsize=15)
ax.set_zlim(0, np.max(y))

ax.view_init(elev=40, azim=10)           # elevation and angle
ax.dist=12   

plt.show()

fig.savefig("IRCobbDoug.pdf")


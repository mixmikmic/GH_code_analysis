import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

# set up a normalized grid:
dim= 10
xarray= np.arange(-dim,dim)
yarray= np.arange(-dim,dim)

x,y = np.meshgrid(xarray,yarray)

vx = -y
vy = x

# plot the flow lines:
plt.figure()
plt.quiver(x,y, vx, vy, pivot='mid')
plt.xlabel("$x$-axis")
plt.ylabel("$y$-axis")
plt.axis('equal')
plt.show()

xmin=3
xmax=4
ymin=3
ymax=4

bottom = -np.sum(vy[ymin,xmin:xmax])
right = np.sum(vx[ymin:ymax,xmax])
top = np.sum(vy[ymax,xmax:xmin:-1])
left = -np.sum(vx[ymax:ymin:-1,xmin])
area = (xmax-xmin)*(ymax-ymin)
total = (top+bottom+left+right)/area

plt.figure()
plt.title('The line intregral of the red box is {:.3}'.format(total))
plt.quiver(x,y, vx, vy, pivot='mid')

# bottom, top, left and right side of the box, respectively:
plt.plot(x[ymin,xmin:xmax+1],y[ymin,xmin:xmax+1],'r',alpha=0.5) # bottom
plt.plot(x[ymax,xmax:xmin-1:-1],y[ymax,xmax:xmin-1:-1],'r',alpha=0.5) # top
plt.plot(x[ymax:ymin-1:-1,xmin],y[ymax:ymin-1:-1,xmin],'r',alpha=0.5) # left
plt.plot(x[ymin:ymax+1,xmax],y[ymin:ymax+1,xmax],'r',alpha=0.5) # right
plt.xlabel("$x$-axis")
plt.ylabel("$y$-axis")
plt.axis('equal')
plt.show()

r= np.sqrt(x**2+y**2)
vx = -y/r**2
vy = x/r**2

plt.figure()
plt.quiver(x,y, vx, vy, pivot='mid')
plt.xlabel("$x$-axis")
plt.ylabel("$y$-axis")
plt.axis('equal')
plt.show()

threshold = 0.33
Mx = np.abs(vx) > threshold
My = np.abs(vy) > threshold
vx = np.ma.masked_array(vx, mask=Mx)
vy = np.ma.masked_array(vy, mask=My)

xmin=3
xmax=4
ymin=3
ymax=4

# closed line "integral".
bottom = -np.sum(vy[ymin,xmin:xmax])
right = np.sum(vx[ymin:ymax,xmax])
top = np.sum(vy[ymax,xmax:xmin:-1])
left = -np.sum(vx[ymax:ymin:-1,xmin])

area = (xmax-xmin)*(ymax-ymin)

total = (top+bottom+left+right)/area

plt.figure()
plt.title('The line intregral of the red box is {:.3}'.format(total))
plt.quiver(x,y, vx, vy, pivot='mid')

# bottom, top, left and right side of the box, respectively:
plt.plot(x[ymin,xmin:xmax+1],y[ymin,xmin:xmax+1],'r',alpha=0.5) # bottom
plt.plot(x[ymax,xmax:xmin-1:-1],y[ymax,xmax:xmin-1:-1],'r',alpha=0.5) # top
plt.plot(x[ymax:ymin-1:-1,xmin],y[ymax:ymin-1:-1,xmin],'r',alpha=0.5) # left
plt.plot(x[ymin:ymax+1,xmax],y[ymin:ymax+1,xmax],'r',alpha=0.5) # right
plt.xlabel("$x$-axis")
plt.ylabel("$y$-axis")
plt.axis('equal')
plt.show()




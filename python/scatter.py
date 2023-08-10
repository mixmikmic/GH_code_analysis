import numpy
from astropy.io import ascii
import matplotlib
from matplotlib import pyplot

get_ipython().run_line_magic('matplotlib', 'inline')

c90 = ascii.read('congress90.csv', format='csv')
c90

c114 = ascii.read('congress114.csv', format='csv')
c114

#define the subplots and figure size
f, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(10,7))

#plot the data with better colors
ax1.scatter(c90['x'], c90['alt'])
ax2.scatter(c114['x'], c114['alt'])

#define the subplots and figure size
f, (ax1, ax2) = pyplot.subplots(1, 2,  figsize = (14, 6.5), sharey = True)

#plot the data with better colors
ax1.scatter(c90['x'], c90['alt'], c = 'lightblue', edgecolors = 'darkblue', zorder = 3)
ax2.scatter(c114['x'], c114['alt'], c = 'lightblue', edgecolors = 'darkblue', zorder = 3)

#add axes labels,and define the limits
ax1.set_xlabel('x', fontsize = 20)
ax1.set_ylabel('alt',fontsize = 20)
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)

ax2.set_xlabel('x', fontsize = 20)
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)

#add titles
ax1.set_title('Congress 90', fontsize = 24)
ax2.set_title('Congress 114', fontsize = 24)

#add a grid? (and darker lines crossing the origin)
t = numpy.arange(9)/4. - 1
ax1.set_yticks(t)
ax1.set_xticks(t)
ax1.grid(color='gray', linestyle='-', linewidth=1, zorder = 1)
ax1.plot([0,0],[-2,2], color='black', zorder = 2)
ax1.plot([-2,2],[0,0], color='black', zorder = 2)

ax2.set_yticks(t)
ax2.set_xticks(t)
ax2.grid(color='gray', linestyle='-', linewidth=1, zorder = 1)
ax2.plot([0,0],[-2,2], color='black', zorder = 2)
ax2.plot([-2,2],[0,0], color='black', zorder = 2)

# Fine-tune figure; make subplots close to each other and hide x ticks for
f.subplots_adjust(wspace=0.02)

#also hide the ticks in the middle 
ax2.yaxis.set_ticks_position('none') 

f.savefig('scatter.pdf',format='pdf', bbox_inches = 'tight') 

#define the subplots and figure size
f = pyplot.figure(figsize=(7,7))

#plot the data with better colors
pyplot.scatter(c90['x'], c90['alt'], c = 'lightblue', edgecolors='darkblue', label='Congress 90', zorder = 3)
pyplot.scatter(c114['x'], c114['alt'], c = 'pink', edgecolors='red', label='Congress 114', zorder = 3)

#add axes labels, and define the limits
pyplot.xlabel('x', fontsize=20)
pyplot.ylabel('alt',fontsize=20)
pyplot.xlim(-1.1, 1.1)
pyplot.ylim(-1.1, 1.1)

#add a grid?
t = numpy.arange(9)/4. - 1
pyplot.axes().set_yticks(t)
pyplot.axes().set_xticks(t)
pyplot.grid(color='gray', linestyle='-', linewidth=1, zorder = 1)
pyplot.plot([0,0],[-2,2], color='black', zorder = 2)
pyplot.plot([-2,2],[0,0], color='black', zorder = 2)


#add a legend
pyplot.legend(loc = 'upper right', fontsize = 14)

from scipy import stats
import matplotlib.cm as cm

xmin= -1.
xmax = 1.
ymin = -1.
ymax = 1.

#I took this from here : https://stackoverflow.com/questions/33793701/pyplot-scatter-to-contour-plot
#which follows closely to the Gaussian KDE scipy page linked above
def density_estimation(x, y):
    xgrid, ygrid = numpy.mgrid[xmin:xmax:110j, ymin:ymax:110j]                                                     
    positions = numpy.vstack([xgrid.ravel(), ygrid.ravel()])                                                       
    values = numpy.vstack([x, y])                                                                        
    kernel = stats.gaussian_kde(values)                                                                 
    zgrid = numpy.reshape(kernel(positions).T, xgrid.shape)
    return xgrid, ygrid, zgrid

x90, y90, z90 = density_estimation(c90['x'], c90['alt'])
x114, y114, z114 = density_estimation(c114['x'], c114['alt'])

#maybe we want one of the contours to be filled
f = pyplot.figure(figsize = (15,15))
pyplot.axes().set_aspect('equal')
cs90 = pyplot.contourf(x90, y90, z90, 10, cmap = cm.Reds,  zorder = 1)
cs114 = pyplot.contour(x114, y114, z114, 10, cmap = cm.winter,  zorder = 4)  

#add a grid?
t = numpy.arange(9)/4. - 1
pyplot.axes().set_yticks(t)
pyplot.axes().set_xticks(t)
pyplot.grid(color='gray', linestyle='-', linewidth=1, zorder = 2)
pyplot.plot([0,0],[-2,2], color='black', zorder = 3)
pyplot.plot([-2,2],[0,0], color='black', zorder = 3)

#add color bars
cb90 = pyplot.colorbar(cs90, shrink=0.5, extend='both')
cb114 = pyplot.colorbar(cs114, shrink=0.5, extend='both', pad = 0.1)
cb90.ax.set_ylabel('Congress 90', labelpad=-80, fontsize=20)
cb114.ax.set_ylabel('Congress 114', labelpad=-80, fontsize=20)

#add axes labels, and define the limits
pyplot.xlabel('x', fontsize=20)
pyplot.ylabel('alt',fontsize=20)
pyplot.xlim(xmin, xmax)
pyplot.ylim(ymin, ymax)

f.savefig('contour.pdf',format='pdf', bbox_inches = 'tight') 

from scipy import stats
import matplotlib.cm as cm

xmin= -1.
xmax = 1.
ymin = -1.
ymax = 1.

#I took this from here : https://stackoverflow.com/questions/33793701/pyplot-scatter-to-contour-plot
#which follows closely to the Gaussian KDE scipy page linked above
def density_estimation(x, y):
    xgrid, ygrid = numpy.mgrid[xmin:xmax:110j, ymin:ymax:110j]                                                     
    positions = numpy.vstack([xgrid.ravel(), ygrid.ravel()])                                                       
    values = numpy.vstack([x, y])                                                                        
    kernel = stats.gaussian_kde(values)                                                                 
    zgrid = numpy.reshape(kernel(positions).T, xgrid.shape)
    return xgrid, ygrid, zgrid

x90, y90, z90 = density_estimation(c90['x'], c90['alt'])
x114, y114, z114 = density_estimation(c114['x'], c114['alt'])

#maybe we want one of the contours to be filled
f = pyplot.figure(figsize = (15,15))
pyplot.axes().set_aspect('equal')

zdiff = []
for z1, z2 in zip(z90, z114):
    zdiff.append(z2-z1)
cs = pyplot.contourf(x114, y114, zdiff, 10, cmap = cm.bwr,  zorder = 1)  
    
#add a grid?
t = numpy.arange(9)/4. - 1
pyplot.axes().set_yticks(t)
pyplot.axes().set_xticks(t)
pyplot.grid(color='gray', linestyle='-', linewidth=1, zorder = 2)
pyplot.plot([0,0],[-2,2], color='black', zorder = 3)
pyplot.plot([-2,2],[0,0], color='black', zorder = 3)

#add color bars
cb114 = pyplot.colorbar(cs, shrink=0.5, extend='both', pad = 0.1)
cb114.ax.set_ylabel('Congress 114 - 90', labelpad=-80, fontsize=20)

#add axes labels, and define the limits
pyplot.xlabel('x', fontsize=20)
pyplot.ylabel('alt',fontsize=20)
pyplot.xlim(xmin, xmax)
pyplot.ylim(ymin, ymax)

f.savefig('contour2.pdf',format='pdf', bbox_inches = 'tight') 




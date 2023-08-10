get_ipython().magic('pylab inline')

import sys, os

from ptha_paths import data_dir, events_dir

def fix_axes(ax=None):
    CClatitude = 41.75  # to rescale longitude
    if ax is None:
        ax = gca()   # get current axes
    ax.set_aspect(1. / cos(pi*CClatitude/180.)) 
    ax.ticklabel_format(format='plain',useOffset=False)

CCmap = imread('%s/MapsTopo/CCimage.png' % data_dir)
extent = (235.79781, 235.82087, 41.739671,41.762726)   #small region

def plot_CCmap():
    fig = figure(figsize=(6,6))
    ax = axes()
    imshow(CCmap,extent=extent)
    fix_axes(ax)
    return fig

fig = plot_CCmap()

fixed_grid_file = os.path.join(data_dir, 'MapsTopo', 'fixedgrid_xyB_small.npy')
d=load(fixed_grid_file)
x=d[:,0]
y=d[:,1]
B=d[:,2]
topo = reshape(B, (250,250), order='F')
X = reshape(x, (250,250), order='F')
Y = reshape(y, (250,250), order='F')
print "Max topography: %g m above MHW" % topo.max()
print "Min topography: %g m " % topo.min()

fig = figure(figsize=(6,6))

clines_land = arange(2,16,2)  
print "contour levels on land (green):", clines_land
contour(X,Y,topo,clines_land,colors='g')

clines_water = arange(-12,0,2) 
print "contour levels in water (blue):", clines_water
# specify solid lines '-' or negative topo will be dashed:
contour(X,Y,topo,clines_water,colors='b',linestyles='-')  

# shoreline (topo = 0 is MHW)
contour(X,Y,topo,[0],colors='r')

fix_axes()

plot_CCmap()
clines = linspace(0,20,11)
print "Contour levels:", clines
contour(X,Y,topo,clines,colors='m')

events_dir = os.path.join(data_dir, 'Events')

event = 'AASZa'
event_dir = os.path.join(events_dir, event)
hmax_file = os.path.join(event_dir, 'h_eta_small.npy')
hmax = load(hmax_file)
Hmax = hmax.reshape((250,250),order='F')

plot_CCmap()
clines = [0.01, 0.5, 1., 1.5, 2., 2.5, 3]
contourf(X,Y,Hmax,clines,alpha = 0.6)
colorbar()




get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import print_function
import sys, os

# local module to read an ascii raster file:
import data_tools

# module for making animations:
import JSAnimation
import animation_tools

data_dir = '/data/CCtimeframes'  # this location should work on Jupyterhub

# check that directory exists, raise an exception if not:
assert os.path.isdir(data_dir), '*** Data directory not found'

topo_file = os.path.join(data_dir,'topo.asc')
topo_data_dict = data_tools.read_asc_file(topo_file, verbose=True)

topo = topo_data_dict['values']
X = topo_data_dict['X']
Y = topo_data_dict['Y']

from scipy.interpolate import RegularGridInterpolator
topo_func = RegularGridInterpolator((X[0,:], Y[:,0]), topo.T)

def read_frame(frameno, verbose=False):
    time = 2*frameno  # minutes after earthquake
    fname = 'depth_%s.asc' % str(frameno).zfill(4)
    depth_file = os.path.join(data_dir, fname)
    if verbose:
        print("Reading %s at time %s minutes" % (depth_file, time))
        
    depth_data_dict = data_tools.read_asc_file(depth_file, verbose=False)

    depth = depth_data_dict['values']
    zeta = where(topo>0, depth, topo + depth)
    return depth, zeta, time

depth, zeta, time = read_frame(40, verbose=True)


def discrete_cmap_1(clines):
    """
    Construct a discrete color map for the regions between the contour lines
    given in clines. Colors go from turqouise through yellow to red.
    Good for flooding depth.
    """
    from numpy import floor, linspace, hstack, ones, zeros
    nlines = len(clines)
    n1 = int(floor((nlines-1)/2.))
    n2 = nlines - 1 - n1
    Green = hstack([linspace(1,1,n1),linspace(1,0,n2)])
    Red = hstack([linspace(0,0.8,n1), ones(n2)])
    Blue = hstack([linspace(1,0.2,n1), zeros(n2)])
    colors = list(zip(Red,Green,Blue))
    return colors

depth_contour_levels = np.arange(0.1,4.5,0.5)  # every 0.5 m up to 4 m
depth_colors = discrete_cmap_1(depth_contour_levels)


def plot_topo_and_depth(zeta, time, ax=None):
    if ax is None:
        fig = figure(figsize=(6,6))
        ax = axes()
    topo_clines = arange(0,20,2)
    ax.contour(X,Y,topo,topo_clines,colors='k')
    ax.contourf(X,Y,zeta, depth_contour_levels, 
             colors = depth_colors, extend='max')
    CClatitude = 41.75  # to rescale longitude
    ax.set_aspect(1. / cos(pi*CClatitude/180.)) 
    ax.ticklabel_format(format='plain',useOffset=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20 )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title('%4i minutes after earthquake' % time)

frameno = 30
depth, zeta, time = read_frame(frameno, verbose=True)
plot_topo_and_depth(zeta, time)

figs = []  # to accumulate figures

for frameno in range(10,41):
    fig = figure(figsize=(6,6))
    ax = axes()
    depth, zeta, time = read_frame(frameno)
    plot_topo_and_depth(zeta, time, ax);
    figs.append(fig)
    close(fig)
    

import animation_tools

animation_tools.interact_animate_figs(figs)

images = animation_tools.make_images(figs, dpi=150)

animation_tools.JSAnimate_images(images, figsize=(6,6))

get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DrawLine:
    def __init__(self, fig,ax1,ax2,depth_func,topo_func,zeta,time):
        self.figure = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.xprev = 0
        self.yprev = 0
        self.xnew = 0
        self.ynew = 0
        self.press = None
        self.depth_func = depth_func
        self.topo_func = topo_func

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)

    def on_press(self, event):
        self.xprev = event.xdata
        self.yprev = event.ydata
        self.press = 1
        ax1.plot(self.xprev,self.yprev,'bo')
        self.figure.canvas.draw()


    def on_release(self, event):
        self.press = None
        self.xnew = event.xdata
        self.ynew = event.ydata
        self.ax1.cla()  # clear the old transect

        # replot topo and water depth:
        plot_topo_and_depth(zeta, time, ax1)
        
        # add transect plot:
        self.plot_transect()
        
        # plot red line between points selected:
        self.ax1.plot([self.xprev,self.xnew],[self.yprev,self.ynew],'b-o',lw=3)

        self.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        
    def plot_transect(self):
        # points on transect:
        xi = linspace(self.xprev, self.xnew, 1000)
        yi = linspace(self.yprev, self.ynew, 1000)
        
        # evaulate topo and zeta on transect:
        Bi = self.topo_func(list(zip(xi,yi)))
        zi = self.depth_func(list(zip(xi,yi)))
                
        # define surface eta as water depth + topography 
        eta = zi+Bi
    
        # Clear axis 2
        self.ax2.cla()
    
        # plot vs. longitude or latitude depending on orientation:
        if (abs(self.xnew-self.xprev) > 0.5*abs(self.ynew-self.yprev)):
            ti = xi
            self.ax2.set_xlim(min(self.xprev,self.xnew),max(self.xprev,self.xnew))
            xtext = 'longitude'
        else:
            ti = yi
            self.ax2.set_xlim(min(self.yprev,self.ynew),max(self.yprev,self.ynew))
            xtext = 'latitude'
            
        BiPos = where(Bi<0, 0., Bi)
        BiNeg = where(Bi>0, 0, Bi)
        
        
        #self.ax2.fill_between(ti, BiPos, eta, color='b')       # flood water
        #self.ax2.fill_between(ti, BiNeg, 0, color=[.7,.7,1])   # original water
        self.ax2.fill_between(ti, -1e6, eta, color='b') 
        self.ax2.fill_between(ti, -1e6, Bi, color=[.5,1,.5])   # topography
        self.ax2.plot(ti, Bi, 'g', lw=1)                       # topography
        
        self.ax2.set_xlabel(xtext)
        self.ax2.set_ylabel('meters')
        self.ax2.set_title('Elevation vs. %s' % xtext)
    
        # choose limits of vertical axis to give nice plots:
        self.ax2.set_ylim(-10, 10)
    
        self.ax2.ticklabel_format(format='plain',useOffset=False)
        self.ax1.set_title('(%8.4f,%8.4f) to (%8.4f,%8.4f)' % (self.xprev,self.yprev,self.xnew,self.ynew))

depth, zeta, time = read_frame(frameno=40, verbose=True)
depth_func = RegularGridInterpolator((X[0,:], Y[:,0]), depth.T)

gs = gridspec.GridSpec(10, 3)
fig = plt.figure(figsize=(12,10))
ax1 = plt.subplot(gs[0:6,0:])

ax2 = plt.subplot(gs[7:,0:])


plot_topo_and_depth(zeta,time,ax1)
ax1.set_title("Click a point, drag, and release")

ax2 = plt.subplot(gs[7:,0:])

dr =  DrawLine(fig,ax1,ax2,depth_func,topo_func,zeta,time)
dr.connect()




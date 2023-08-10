get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')

from __future__ import print_function
import sys, os
from ipywidgets import interact
from IPython.display import Image, display

# local module to read an ascii raster file:
import data_tools

topo_file = '../geoclaw_output/chile2010/topo.asc'
topo_data_dict = data_tools.read_asc_file(topo_file, verbose=True)

X = topo_data_dict['X']
Y = topo_data_dict['Y']
topo = topo_data_dict['values']

frame_times = np.linspace(0,5.5,12)
print("The data is at the following times (in hours after the earthquake):\n   ", frame_times)

frameno = 9
filename = 'eta_%s.asc' % str(frameno).zfill(4)  # pads with zeros to fill 4 characters
eta_file = '../geoclaw_output/chile2010/%s' % filename
print('Reading file %s\nTsunami simulation at time %4.1f hours after quake\n' % (eta_file, frame_times[frameno]))
eta_data_dict = data_tools.read_asc_file(eta_file)
Eta = eta_data_dict['values']

from numpy import ma  # masked arrays

Eta_offshore = ma.masked_where(topo>0, Eta)
eta_colors = [[0,0,1], [.5,.5,1], [0,1,1], [1,.5,.5], [1,0,0]]
eta_levels = [-0.5,-0.1,-0.05,0.05,0.1,0.5]

def plot_topo_and_depth(ax=None):
    if ax is None:
        fig = figure(figsize=(6,6))
        ax = axes()
    topo_clines = linspace(-4000,0,5)
    ax.contour(X,Y,topo,topo_clines,colors='k',linestyles='-')
    ax.contourf(X,Y,Eta_offshore, eta_levels, 
             colors = eta_colors, extend='max')
    latitude = -30.  # to rescale longitude
    ax.set_aspect(1. / cos(pi*latitude/180.)) 
    ax.ticklabel_format(format='plain',useOffset=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20 )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude");

plot_topo_and_depth()

from scipy.interpolate import RegularGridInterpolator
topo_func = RegularGridInterpolator((X[0,:], Y[:,0]), topo.T)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DrawLine:
    def __init__(self, fig,ax1,ax2,ax3,eta_func,topo_func):
        self.figure = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.xprev = 0
        self.yprev = 0
        self.xnew = 0
        self.ynew = 0
        self.press = None
        self.eta_func = eta_func
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
        plot_topo_and_depth(ax1)
        
        # add transect plot:
        self.plot_transect(self.ax2, [-0.5, 0.5])
        self.plot_transect(self.ax3, [-6000,1000])
        
        # plot red line between points selected:
        self.ax1.plot([self.xprev,self.xnew],[self.yprev,self.ynew],'b-o',lw=3)

        self.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        
    def plot_transect(self, ax, ylim):
        # points on transect:
        xi = linspace(self.xprev, self.xnew, 1000)
        yi = linspace(self.yprev, self.ynew, 1000)
        
        # evaulate topo and zeta on transect:
        Bi = self.topo_func(list(zip(xi,yi)))
        eta = self.eta_func(list(zip(xi,yi)))
                
        # define surface eta as water depth + topography 
        #eta = zi+Bi
    
        # Clear axis
        ax.cla()
    
        # plot vs. longitude or latitude depending on orientation:
        if (abs(self.xnew-self.xprev) > 0.5*abs(self.ynew-self.yprev)):
            ti = xi
            ax.set_xlim(min(self.xprev,self.xnew),max(self.xprev,self.xnew))
            xtext = 'longitude'
        else:
            ti = yi
            self.ax.set_xlim(min(self.yprev,self.ynew),max(self.yprev,self.ynew))
            xtext = 'latitude'
            
        BiPos = where(Bi<0, 0., Bi)
        BiNeg = where(Bi>0, 0, Bi)
        
        
        #ax.fill_between(ti, BiPos, eta, color='b')       # flood water
        #ax.fill_between(ti, BiNeg, 0, color=[.7,.7,1])   # original water
        #ax.fill_between(ti, -1e6, Bi, color=[.5,1,.5])   # topography
        ax.plot(ti, eta, 'b', lw=2)
        ax.plot(ti, zeros(ti.shape), 'k')
        ax.plot(ti, Bi, 'g', lw=2)                       # topography
        
        ax.set_xlabel(xtext)
        ax.set_ylabel('meters')
        ax.set_title('Elevation vs. %s' % xtext)
    
        # choose limits of vertical axis to give nice plots:
        ax.set_ylim(ylim)
    
        ax.ticklabel_format(format='plain',useOffset=False)
        self.ax1.set_title('(%8.4f,%8.4f) to (%8.4f,%8.4f)' % (self.xprev,self.yprev,self.xnew,self.ynew))

eta_func = RegularGridInterpolator((X[0,:], Y[:,0]), Eta_offshore.T)

gs = gridspec.GridSpec(13, 3)
fig = plt.figure(figsize=(12,13))
ax1 = plt.subplot(gs[0:6,0:])

plot_topo_and_depth(ax1)
ax1.set_title("Click a point, drag, and release")

ax2 = plt.subplot(gs[7:9,0:])
ax3 = plt.subplot(gs[10:,0:])

plt.setp( ax1.xaxis.get_majorticklabels(), rotation=20 )

dr =  DrawLine(fig,ax1,ax2,ax3,eta_func,topo_func)
dr.connect()




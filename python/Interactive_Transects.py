get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')

from __future__ import print_function
import sys, os

# local module to read an ascii raster file:
import data_tools

ls

topo_file = 'topo_CC.asc.txt'
topo_data_dict = data_tools.read_asc_file(topo_file, verbose=True)

topo = topo_data_dict['values']

hmax_file = 'hmax_CC.asc.txt'
hmax_data_dict = data_tools.read_asc_file(hmax_file, verbose=False)

hmax = hmax_data_dict['values']
X = hmax_data_dict['X']
Y = hmax_data_dict['Y']

from numpy import ma  # masked arrays

# mask out the dry cells (e.g., where depth < 1 mm):
hmax_dry = ma.masked_where(hmax < 0.001, hmax)

# mask out the water region if we only care about onshore:
hmax_onshore = ma.masked_where(topo < 0, hmax_dry)

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

depth_contour_levels = np.arange(0,4.5,0.5)  # every 0.5 m up to 4 m
depth_colors = discrete_cmap_1(depth_contour_levels)


def plot_topo_and_depth(ax=None):
    if ax is None:
        fig = figure(figsize=(6,6))
        ax = axes()
    topo_clines = arange(0,20,2)
    ax.contour(X,Y,topo,topo_clines,colors='k')
    ax.contourf(X,Y,hmax_onshore, depth_contour_levels, 
             colors = depth_colors, extend='max')
    CClatitude = 41.75  # to rescale longitude
    ax.set_aspect(1. / cos(pi*CClatitude/180.)) 
    ax.ticklabel_format(format='plain',useOffset=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20 )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude");

plot_topo_and_depth()

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
ext = [-124.215,-124.175,41.735,41.77]
A = tiles = cimgt.OSM()
plt.figure(figsize=(7,7))
ax = plt.axes(projection=tiles.crs)
ax.set_extent(ext)
ax.add_image(tiles,14)
# ax.contour(X,Y,topo,topo_clines, colors='k',opacity=0.5,transform=ccrs.PlateCarree())

plt.savefig("street_view_2", dpi=500, bbox_inches='tight', pad_inches=0)

from scipy.interpolate import RegularGridInterpolator
topo_func = RegularGridInterpolator((X[0,:], Y[:,0]), topo.T)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DrawLine:
    def __init__(self, fig,ax1,ax2,ax3,hmax_func,topo_func):
        self.figure = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.xprev = 0
        self.yprev = 0
        self.xnew = 0
        self.ynew = 0
        self.press = None
        self.hmax_func = hmax_func
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
        
        self.ax1.lines = []
        if len(self.ax1.texts) > 0:
            self.ax1.texts.remove(self.ax1.texts[0])
        if len(self.ax1.texts) > 0:
            self.ax1.texts.remove(self.ax1.texts[0])
        plt.draw()
        
        self.ax1.plot(self.xprev,self.yprev,'bo')
        self.ax1.text(self.xprev,self.yprev-.0012, r'A', fontsize=12,fontweight='bold',
                      bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},ha='center', va='center')
        
        self.figure.canvas.draw()

    def on_release(self, event):
        self.press = None
        self.xnew = event.xdata
        self.ynew = event.ydata
              
        # self.plot1.remove()      
        # self.ax1.cla()  # clear the old transect

        # replot topo and water depth:
        # plot_topo_and_depth(ax1)
        
        # self.ax1.add_image(tiles,15)
        # self.ax1.contour(X,Y,topo,topo_clines, colors='k',opacity=0.5)
        # CClatitude = 41.75  # to rescale longitude
        # self = ax1.set_aspect(1. / cos(pi*CClatitude/180.)) 
        
        # add transect plot:
        self.plot_transect()
        
        # plot red line between points selected:
        self.ax1.plot([self.xprev,self.xnew],[self.yprev,self.ynew],'b-o',lw=3)
        
        self.ax1.text(self.xnew,self.ynew+.0012, r'B', fontsize=12,fontweight='bold',
                      bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},ha='center', va='center')

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
        zi = self.hmax_func(list(zip(xi,yi)))
                
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
        
        
        self.ax2.fill_between(ti, BiPos, eta, color='b')       # flood water
        self.ax2.fill_between(ti, BiNeg, 0, color=[.7,.7,1])   # original water
        self.ax2.fill_between(ti, -1e6, Bi, color=[.5,1,.5])   # topography
        self.ax2.plot(ti, Bi, 'g', lw=1)                       # topography
        
        self.ax2.set_xlabel(xtext)
        self.ax2.set_ylabel('meters')
        self.ax2.set_title('Cross-Section: Elevation vs. %s' % xtext)
    
        # choose limits of vertical axis to give nice plots:
        eta_wet_max = eta.max()  #where(zi>0, zi+Bi, 0).max()
        y2 = max(10,eta_wet_max)
        self.ax2.set_ylim(-5, y2)
        
        self.ax2.text(ti[0],y2+3, r'A', fontsize=12,fontweight='bold',
                      bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},ha='center', va='center')
        self.ax2.text(ti[-1],y2+3, r'B', fontsize=12,fontweight='bold',
                      bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},ha='center', va='center')
    
        self.ax2.ticklabel_format(format='plain',useOffset=False)
        
        self.ax3.cla() 
        # self.ax1.set_title('(%8.4f,%8.4f) to (%8.4f,%8.4f)' % (self.xprev,self.yprev,self.xnew,self.ynew))
        clust_data = np.array([[self.xprev,self.yprev],[self.xnew,self.ynew]])
        collabel=("Longitude", "Latitude")
        rowlabel=(" A ", " B ")
        ax3.axis('tight')
        ax3.axis('off')
        the_table = self.ax3.table(cellText=clust_data,colLabels=collabel,rowLabels=rowlabel,loc='center')
        the_table.set_fontsize(14)
        the_table.scale(1, 1.5)



hmax_func = RegularGridInterpolator((X[0,:], Y[:,0]), hmax.T)

# gs = gridspec.GridSpec(12, 4) # define a grid over which to plot 
# fig = plt.figure(figsize=(12,12))
# create figure 1 specifications 
# ax1 = plt.subplot(gs[0:9,0:2])

fig = plt.figure(figsize=(12,10))
ax1 = plt.axes([.01,.4,.7,.5])

# ax = plt.axes(projection=tiles.crs)
# ax1.set_extent(ext)
# ax1.add_image(tiles,15)
CCimage = plt.imread('street_view_2.png')
# image_extent = (X.min(),X.max(),Y.min(),Y.max()) 
plt.imshow(CCimage, extent = (X.min(),X.max(),Y.min(),Y.max()));
CClatitude = 41.75  # to rescale longitude
ax1.set_aspect(1. / cos(pi*CClatitude/180.)) 

topo_clines = arange(0,20,2)
# ax1.contour(X,Y,topo,topo_clines, colors='#808080',linewidth=1,opacity=0.5)
ax1.contourf(X,Y,hmax_onshore, depth_contour_levels,colors = depth_colors, extend='max',alpha = 0.5)

# topo_clines = arange(0,20,2)
# ax1.contour(X,Y,topo,topo_clines,colors='k')

# ax1.contourf(X,Y,hmax_onshore, depth_contour_levels,colors = depth_colors, extend='max')
ax1.ticklabel_format(format='plain',useOffset=False)
#plt.xticks(rotation=20)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Cresent City Inundation: click, drag, release")

# ax2 = plt.subplot(gs[10:,0:])
ax2 = plt.axes([.1,.1,.8,.2])

# ax3 = plt.subplot(gs[0:6,2:4])
ax3 = plt.axes([.62,.5,.3,.3])
clust_data = np.array([[1,2],[3,4]])
collabel=("Longitude", "Latitude")
rowlabel=(" A ", " B ")
ax3.axis('tight')
ax3.axis('off')
the_table = ax3.table(cellText=clust_data,colLabels=collabel,rowLabels=rowlabel,loc='center')
the_table.set_fontsize(14)
the_table.scale(1, 1.5)

plt.setp( ax1.xaxis.get_majorticklabels(), rotation=20 )
dr =  DrawLine(fig,ax1,ax2,ax3,hmax_func,topo_func)
dr.connect()

Y.max()

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
ext = [-124.215,-124.175,41.735,41.77]
tiles = cimgt.OSM()
plt.figure(figsize=(9,9))
ax = plt.axes(projection=tiles.crs)
ax.set_extent(ext)
ax.add_image(tiles,15)
ax.contour(X,Y,topo,topo_clines, colors='k',opacity=0.5,transform=ccrs.PlateCarree())

tiles

fig = plt.figure(figsize=(7,7))
ax = plt.axes()
CCimage = plt.imread('street_view.png')
# image_extent = (X.min(),X.max(),Y.min(),Y.max()) 
plt.imshow(CCimage, extent = (X.min(),X.max(),Y.min(),Y.max()));

fig = plt.figure(figsize=(10,10))
ax1 = plt.axes()
CCimage = plt.imread('street_view.png')
# image_extent = (X.min(),X.max(),Y.min(),Y.max()) 
plt.imshow(CCimage, extent = (X.min(),X.max(),Y.min(),Y.max()));




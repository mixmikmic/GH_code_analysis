get_ipython().magic('matplotlib notebook')

from pylab import *

import sys, os
from ipywidgets import interact,interact_manual, Image

from ptha_paths import data_dir, events_dir

# Read in topography data:

fixed_grid_file = os.path.join(data_dir, 'MapsTopo', 'fixedgrid_xyB_small.npy')
d=load(fixed_grid_file)
x=d[:,0]
y=d[:,1]
B=d[:,2]
topo = reshape(B, (250,250), order='F')
X = reshape(x, (250,250), order='F')
Y = reshape(y, (250,250), order='F')

def plot_topo():
    fig = figure(figsize=(6,6))
    ax = axes()
    topo_clines = arange(0,20,2)
    contour(X,Y,topo,topo_clines,colors='k')
    CClatitude = 41.75  # to rescale longitude
    ax.set_aspect(1. / cos(pi*CClatitude/180.)) 
    ax.ticklabel_format(format='plain',useOffset=False)
    return fig

CCmap = imread('%s/MapsTopo/CCimage.png' % data_dir)
extent = (235.79781, 235.82087, 41.739671,41.762726)   #small region

def plot_CCmap():
    fig = figure(figsize=(6,6))
    ax = axes()
    imshow(CCmap,extent=extent)
    CClatitude = 41.75  # to rescale longitude
    ax.set_aspect(1. / cos(pi*CClatitude/180.)) 
    ax.ticklabel_format(format='plain',useOffset=False)
    axis(extent)
    return fig

# these levels were used in original study:
#zeta = hstack((arange(0,2.,.1), arange(2.0,12.5,.5))) 

# you get nicer looking curves by using a denser set of exceedance values:
zeta = linspace(0,12,121)
nzeta = len(zeta)
print '%i exceedance values, \nzeta =  %s' % (nzeta,zeta)

all_events = ['AASZa', 'AASZb', 'AASZc', 'AASZd', 'CSZa', 'CSZb', 'CSZc', 'CSZd', 'CSZe',               'CSZf', 'KmSZa', 'KrSZa', 'SChSZa', 'TOHa']

event_prob = {}
event_prob['AASZa'] = 1./394.
event_prob['AASZb'] = 1./750.
event_prob['AASZc'] = 1./563.
event_prob['AASZd'] = 1./324.
event_prob['CSZa'] = 1./250. * .0125
event_prob['CSZb'] = 1./250. * .0125
event_prob['CSZc'] = 1./250. * .0750
event_prob['CSZd'] = 1./250. * .5000
event_prob['CSZe'] = 1./250. * .1750
event_prob['CSZf'] = 1./250. * .2250
event_prob['KmSZa'] = 1./50.
event_prob['KrSZa'] = 1./167.
event_prob['SChSZa'] = 1./300.
event_prob['TOHa'] = 1./103.

print "Annual probability of each event is set to:"
print event_prob

def combine_prob(p1,p2):
    """Returns the probability that event 1 or 2 happens"""
    return 1. - (1-p1)*(1-p2)

events = all_events

# Instead, to use a subset of the events, specify a list such as:
#events = ['AASZa', 'AASZb', 'AASZc']

nx, ny = X.shape  # note that X is a 2d array of longitude values at each point
exceed_prob = zeros((nx,ny,nzeta))  # initialize to zero

# loop over all events and update exceed_prob at each grid point by combining
# current value with the probability Pk of this event:

for event in events:
    event_dir = os.path.join(events_dir, event)
    hmax_file = os.path.join(event_dir, 'h_eta_small.npy')
    hmax = load(hmax_file)
    Hmax = hmax.reshape((nx,ny),order='F')
    for k in range(nzeta):
        Pk = exceed_prob[:,:,k]  # probabilities at all points for one exceedance value zeta_k
        exceed_prob[:,:,k] = where(Hmax > zeta[k], combine_prob(event_prob[event],Pk), Pk)
        
print "Computed exceedance probabilities.  \nMaximum over all grid points is %g" % exceed_prob.max()

dx = X[1,0] - X[0,0]
dy = Y[0,1] - Y[0,0]
nx, ny = X.shape
xmin = X.min(); xmax = X.max()
ymin = Y.min(); ymax = Y.max()

def plot_hcurve(longitude, latitude):
    i = (longitude - X[0,0]) / dx
    j = (latitude - Y[0,0]) / dy
    if (i<0) or (i>=nx) or (j<0) or (j>=ny):
        print "out of domain"
        return 
    fig = figure(figsize=(12,5))
    subplot(1,2,1)
    p = maximum(exceed_prob[i,j,:], 1e-10)
    semilogy(zeta, p, 'b')
    ylim(1e-5,1)
    xlabel('zeta in meters')
    ylabel('annual probability')
    title('Hazard Curve')
    
    # Also plot the CC image with a red dot showing the location:
    ax = subplot(1,2,2)
    imshow(CCmap,extent=extent)
    CClatitude = 41.75  # to rescale longitude
    ax.set_aspect(1. / cos(pi*CClatitude/180.)) 
    ax.ticklabel_format(format='plain',useOffset=False)
    xticks(rotation=20)
    plot([longitude], [latitude], 'ro')
    xlim(xmin,xmax)
    ylim(ymin,ymax)
    title('Location')
    #return fig

fig = plot_hcurve(235.805, 41.75)

dx = X[1,0] - X[0,0]
dy = Y[0,1] - Y[0,0]
nx, ny = X.shape
xmin = X.min(); xmax = X.max()
ymin = Y.min(); ymax = Y.max()

xmean = X.mean()
ymean = Y.mean()

def p_haz_curve(longitude,latitude):
    i = (longitude - X[0,0]) / dx
    j = (latitude - Y[0,0]) / dy
    if (i<0) or (i>=nx) or (j<0) or (j>=ny):
        print "out of domain"
    p = maximum(exceed_prob[i,j,:], 1e-10)
    return p
    
fig = figure(figsize=(12,5))
ax1 = subplot(1,2,1)
p = p_haz_curve(xmean, ymean)
line = semilogy(zeta, p, 'b')
ylim(1e-5,1)
xlabel('zeta in meters')
ylabel('annual probability')
title('Hazard Curve')

# Also plot the CC image with a red dot showing the location:
ax2 = subplot(1,2,2)
imshow(CCmap,extent=extent)
CClatitude = 41.75  # to rescale longitude
ax2.set_aspect(1. / cos(pi*CClatitude/180.)) 
ax2.ticklabel_format(format='plain',useOffset=False)
xticks(rotation=20)
point = ax2.plot([xmean], [ymean], 'ro')
xlim(xmin,xmax)
ylim(ymin,ymax)
title("Click on a point below")
      
def onclick(event):
    point[0].set_data([[event.xdata],[event.ydata]])
    line[0].set_data([zeta, p_haz_curve(event.xdata, event.ydata)])
    ax1.set_title("Hazard curve at \nx = %6.3f, y = %6.3f" % (event.xdata, event.ydata))
    draw()
    

cid = fig.canvas.mpl_connect('button_press_event', onclick)

interact_manual(plot_hcurve, longitude=(xmin,xmax,.001),latitude=(ymin,ymax,0.001))


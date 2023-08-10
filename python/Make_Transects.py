get_ipython().magic('pylab inline')

import sys, os
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

from scipy.interpolate import RegularGridInterpolator
topo_func = RegularGridInterpolator((X[:,0], Y[0,:]), topo)

def plot_transect(x1,y1, x2,y2, z_func):
    
    # points on transect:
    xi = linspace(x1, x2, 1000)
    yi = linspace(y1, y2, 1000)
    
    # plot the transect on top of topography:
    fig = plot_topo()
    plot([x1,x2], [y1,y2], 'r-', linewidth=3)
    title('Transect')

    # evaulate topo and zeta on transect:
    Bi = topo_func(zip(xi,yi))
    zi = z_func(zip(xi,yi))
    
    # define eta as zeta offshore or zeta + topo onshore:
    eta = where(Bi<0, zi, zi+Bi)
    
    # plot cross sections on topo
    # plot vs. longitude or latitude depending on orientation:
    
    if (abs(x2-x1) > 0.5*abs(y2-y1)):
        ti = xi
        xtext = 'longitude'
    else:
        ti = yi
        xtext = 'latitude'
        
    Bi0 = where(Bi<0, 0., Bi)
    
    figure(figsize=(15,4))
    fill_between(ti, Bi0, eta, color='b')
    plot(ti, Bi, 'g')
    
    xlabel(xtext)
    ylabel('meters')
    plt.title('Elevation vs. %s' % xtext)
    
    # choose limits of vertical axis to give nice plots:
    eta_wet_max = where(zi>0, zi+Bi, 0).max()
    ylim(-2, max(6,eta_wet_max))
    
    ticklabel_format(format='plain',useOffset=False)
    plt.xticks(rotation=20)

event = 'AASZb'
event_dir = os.path.join(events_dir, event)
hmax_file = os.path.join(event_dir, 'h_eta_small.npy')
hmax = load(hmax_file)
Hmax = hmax.reshape((250,250),order='F')
zeta_event_func = RegularGridInterpolator((X[:,0], Y[0,:]), Hmax)

x1 = 235.805;  y1 = 41.745
x2 = 235.800;  y2 = 41.762

print("Event: %s" % event)
plot_transect(x1,y1, x2,y2, zeta_event_func)

zeta = hstack((arange(0,2.,.1), arange(2.0,12.5,.5)))
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

def compute_zeta(p):

    # create boolean array K with K[i,j,k] == True only where exceed_prob[i,j,k] > p:
    K = exceed_prob > p

    K[:,:,0] = True
    zeta_p = zeros(X.shape)
    for i in range(nx):
        for j in range(ny):
            zeta_p[i,j] = zeta[K[i,j,:]][-1]
    return zeta_p

p = 0.002
zeta_p = compute_zeta(p)
zeta_p_func = RegularGridInterpolator((X[:,0], Y[0,:]), zeta_p)

# Set endpoints of transect:
x1 = 235.805;  y1 = 41.745
x2 = 235.800;  y2 = 41.762

print "Transect of zeta for given annual probability p = %g" % p
plot_transect(x1,y1, x2,y2, zeta_p_func)




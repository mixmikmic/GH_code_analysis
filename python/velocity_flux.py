import yt
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import edgefinder
import scipy.optimize as opt
from yt.units import kpc,km,s,cm
from yt import derived_field
from yt.analysis_modules.halo_analysis.api import HaloCatalog
import scipy
from scipy import ndimage
import os
from time import time
from glob import glob
from siman import *
get_ipython().magic('matplotlib inline')

# This is the base path to the directory containing the simulation snapshots
base = '/astro/simulations/FOGGIE/halo_008508/'

# 100% Feedback
#mid = 'nref10_track_2'
#L = [-0.37645994,  0.50723191,  0.77523784]
#Lx = [ 0.,          0.83679869, -0.54751068]

# 30% Feedback
mid = 'nref10_track_lowfdbk_1'
L = [-0.56197719,  0.56376017,  0.60527358]
Lx = [ 0.,          0.73175552, -0.6815672 ]

# 10% Feedback
#mid = 'nref10_track_lowfdbk_2'
#L = [-0.48868346,  0.6016812,   0.63179761]
#Lx = [ 0., 0.72415556, -0.68963666]

# 3% Feedback
#mid = 'nref10_track_lowfdbk_3'
#L = [-0.28229104, 0.68215233, 0.67452203]
#Lx = [ 0., 0.7031187, -0.71107249]

# 1% Feedback
#mid = 'nref10_track_lowfdbk_4'
#L = [-0.37819875, 0.57817821, 0.72296312]
#Lx = [ 0., 0.78097012, -0.62456839]

full_dir = base+mid+'/DD????/DD????'
all_dir = glob(full_dir)     # a list of each simulation snapshot found in the main simulation directory

# Select one snapshot for a test case, load that dataset, and derive the center of the galaxy
file_name = all_dir[50]
ds,center = galaxy_center(file_name)

# Define a sphere in which the velocity of the enitre galaxy can be calculated
sp = ds.sphere(center,(12,'kpc'))
# Calculates the bulk velocity of the galaxy
bulk_v = sp.quantities.bulk_velocity().in_units('km/s')
gx = bulk_v[0]
gy = bulk_v[1]
gz = bulk_v[2]

def _vflux(field,data):
    # Define x, y, and z coordinates
    x = data['x-velocity'].in_units('km/s')
    y = data['y-velocity'].in_units('km/s')
    z = data['z-velocity'].in_units('km/s')
    # Take dot product of bulk velocity and angular momentum vector
    bx = np.multiply(bulk_v[0],L[0])
    by = np.multiply(bulk_v[1],L[1])
    bz = np.multiply(bulk_v[2],L[2])
    leng = bx+by+bz
    # Take dot product of new velocity and angular momentum vector
    nx = x-leng
    ny = y-leng
    nz = z-leng
    Lxx = np.multiply(nx,L[0])
    Ly = np.multiply(ny,L[1])
    Lz = np.multiply(nz,L[2])
    return Lxx+Ly+Lz
ds.add_field('cyl_flux',function=_vflux,units='km/s',display_name='Velocity Flux')
ds.add_field('velocity_flux',function=_vflux,units='km/s',display_name='Velocity Flux')

def _rflux(field,data):
    x = data['x-velocity'].in_units('km/s')
    y = data['y-velocity'].in_units('km/s')
    z = data['z-velocity'].in_units('km/s')
    # Take dot product of bulk velocity and Lr vector
    Lr = np.cross(L,Lx)
    bx = np.multiply(bulk_v[0],Lr[0])
    by = np.multiply(bulk_v[1],Lr[1])
    bz = np.multiply(bulk_v[2],Lr[2])
    leng = bx+by+bz
    
    nx = x-leng
    ny = y-leng
    nz = z-leng
    Lxx = np.multiply(nx,Lr[0])
    Ly = np.multiply(ny,Lr[1])
    Lz = np.multiply(nz,Lr[2])
    return Lxx+Ly+Lz
ds.add_field('r_flux',function=_rflux,units='km/s')

oap=yt.SlicePlot(ds,Lx,'metallicity',center=center,width=(70,'kpc'),north_vector=L,data_source=bsp)
oap.annotate_quiver('r_flux','velocity_flux',16)
oap.show()

# Define a data cylinder in which we will perform calculations
cyl = ds.disk(center,L,(12.,'kpc'),(200.,'kpc'))

# Create a dictionary containing velocity flux calculations
results = calculate_vflux_profile(cyl,-100,100,2)

# Save the results from out dictionary as individual variables
nhl = results['height']
highperlist= results['highperlist']
lowperlist = results['lowperlist']
mean_flux_profile = results['mean_flux_profile']
median_flux_profile = results['median_flux_profile']

fig,axs=plt.subplots()

# Plots the 25th and 75th percentiles
axs.fill_between(nhl,highperlist,lowperlist,facecolor='r',alpha=.35)
# Plots mean velocity profile
axs.plot(nhl,mean_flux_profile,color='firebrick')
# Plots median velocity profile
axs.plot(nhl,median_flux_profile)

# Plots lines to make clear the difference between inflows and outflows
#     (positive or negative velocity), as well as above and below the
#     disk (positive (above) or negative (below) distance)
axs.plot([0,0],[-1000,1000],color='black',linewidth=.5,alpha=.6)
axs.plot([-1000,1000],[0,0],color='black',linewidth=.5,alpha=.6)
axs.set_xlim(-105,105)
axs.set_ylim(-160,200)

axs.set_xlabel('Distance from Galactic Plane (kpc)')
axs.set_ylabel('Mean Velocity Flux (km/s)')
axs.set_title(file_name[-6:]+': 30% Feedback')
plt.show()

# Define a data cylinder in which we will perform calculations
cyl = ds.disk(center,L,(12.,'kpc'),(200.,'kpc'))

# Create a dictionary containing velocity flux calculations
results1 = calculate_vflux_profile(cyl,-100,100,2,weight = True)

# Save the results from out dictionary as individual variables
nhl = results1['height']
highperlist= results1['highperlist']
lowperlist = results1['lowperlist']
mean_flux_profile = results1['mean_flux_profile']
weighted_profile = results1['weighted_profile']

fig,axs=plt.subplots()

# Plots the 25th and 75th percentiles
axs.fill_between(nhl,highperlist,lowperlist,facecolor='r',alpha=.35)
# Plots mean velocity profile
axs.plot(nhl,mean_flux_profile,color='firebrick')
# Plots mass weighted profile
axs.plot(nhl,weighted_profile)

# Plots lines to make clear the difference between inflows and outflows
#     (positive or negative velocity), as well as above and below the
#     disk (positive (above) or negative (below) distance)
axs.plot([0,0],[-1000,1000],color='black',linewidth=.5,alpha=.6)
axs.plot([-1000,1000],[0,0],color='black',linewidth=.5,alpha=.6)
axs.set_xlim(-105,105)
axs.set_ylim(-160,200)

axs.set_xlabel('Distance from Galactic Plane (kpc)')
axs.set_ylabel('Mean Velocity Flux (km/s)')
axs.set_title(file_name[-6:]+': 30% Feedback')
plt.show()

# Define a data cylinder in which we will perform calculations
cyl = ds.disk(center,L,(12.,'kpc'),(200.,'kpc'))

# Creates a dictionary containing the volume- and mass-weighted
#     velocity flux profiles for outflowing material
results2 = calculate_vflux_profile(cyl,0,100,2,weight=True,flow='out')

fig,axs=plt.subplots()

# Save the results from out dictionary as individual variables
nhl = results['height']
mean_flux_profile = results['mean_flux_profile']
wflux_profile = results['weighted_profile']

# Plots mean velocity profile
axs.plot(nhl,mean_flux_profile,color='firebrick')
# Plots mass weighted profile
axs.plot(nhl,wflux_profile)

# Plots lines to make clear the difference between inflows and outflows
#     (positive or negative velocity), as well as above and below the
#     disk (positive (above) or negative (below) distance)
axs.plot([0,0],[-1000,1000],color='black',linewidth=.5,alpha=.6)
axs.plot([-1000,1000],[0,0],color='black',linewidth=.5,alpha=.6)
axs.set_xlim(0,105)
axs.set_ylim(0,300)

axs.set_xlabel('Distance from Galactic Plane (kpc)')
axs.set_ylabel('Mean Velocity Flux (km/s)')
axs.set_title(file_name[-6:]+': 30% Feedback')
plt.show()




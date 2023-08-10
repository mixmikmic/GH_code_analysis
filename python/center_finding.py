import yt
import numpy as np
import matplotlib.pyplot as plt
from edgefinder import *
import scipy.optimize as opt
from yt.units import kpc,cm,km,s
from yt import derived_field
from astropy.table import Table
get_ipython().magic('matplotlib inline')

'''
Jason's method:
  Does some magic

pros:
  - quick
  - literally don't need any kind of starting point

cons:
  - not very accurate on its own, especially if merger is happening

'''

def galaxy_center(filename):
    track = Table.read('/astro/simulations/FOGGIE/'+filename[26:-14]+'/halo_track', format='ascii') 
    track.sort('col1')

    # load the snapshot 
    global ds
    ds = yt.load(filename)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    
    # interpolate the center from the track
    centerx = np.interp(zsnap, track['col1'], track['col2']) 
    centery = np.interp(zsnap, track['col1'], track['col3']) 
    centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) + np.interp(zsnap, track['col1'], track['col7'])) 

    cen = [centerx, centery+20. / 143886., centerz]
    return ds,cen

'''
Directional Profile Method
   Calculates velocity profile along x, y, and z axes, finds the maximum, and sets that
   as the center

pros:
  - fairly accurate
  
cons:
  - broken af
  - need Jason's method first
  - seriously it's like super broken don't use it

'''

# Makes density profile along one direction
def make_profile(ds,cen,rad,xyz):
    sph = ds.sphere(cen, (rad,'kpc'))
    rp = yt.create_profile(sph, xyz, 'density',
                          logs = {'density': True})
    return rp.x.value,rp["density"].value

# Finds max value of density profile along one direction
def find_max(ds,cen,rad,xyz):
    g,gdens = make_profile(ds,cen,rad,xyz)
    maxim = np.where(gdens == np.max(gdens))[0]
    return g[maxim][0]

# Does the two above things for each direction
def find_gal_center(ds,cen,rad):
    coord = ['x','y','z']
    newcenter = []
    for i in coord:
        newcenter.append(find_max(ds,cen,rad,i))
    return newcenter

'''
Dark Matter Density Method
   finds the point with the highest dark matter density and sets that as the center

pros:
   - super ez
   - works fairly well
   
cons:
   - sometimes picks wrong galaxy
   - needs Jason's method

'''

def dm_dens(file_name):
    ds,near_center = galaxy_center(file_name)
    sph = ds.sphere(near_center, (50,'kpc'))
    best = np.where(np.max(sph['dark_matter_density'])==sph['dark_matter_density'])[0][0]
    center = [sph['x'][best],sph['y'][best],sph['z'][best]]
    return ds,sph,center

filename = '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_1/DD0057/DD0057'
L = [-0.56197719,  0.56376017,  0.60527358]
Lx = [ 0.,          0.73175552, -0.6815672 ]

ds,jcent = galaxy_center(filename)
oap = yt.OffAxisProjectionPlot(ds,Lx,'density',center=jcent,width=(100,'kpc'),north_vector=L)
oap.show()

ds,sp,dmcent = dm_dens(filename)
oap = yt.OffAxisProjectionPlot(ds,Lx,'density',center=dmcent,width=(70,'kpc'),north_vector=L)
oap.show()




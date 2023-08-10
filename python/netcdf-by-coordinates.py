import netCDF4
filename = 'data/rtofs_glo_3dz_f006_6hrly_reg3.nc'
ncfile = netCDF4.Dataset(filename, 'r')
print ncfile                # shows global attributes, dimensions, and variables
ncvars = ncfile.variables   # a dictionary of variables
# print information about specific variables, including type, shape, and attributes
for varname in ['temperature', 'salinity', 'Latitude', 'Longitude']:
    print ncvars[varname]

import numpy as np
import netCDF4

def naive_slow(latvar,lonvar,lat0,lon0):
    '''
    Find "closest" point in a set of (lat,lon) points to specified point
    latvar - 2D latitude variable from an open netCDF dataset
    lonvar - 2D longitude variable from an open netCDF dataset
    lat0,lon0 - query point
    Returns iy,ix such that 
     (lonval[iy,ix] - lon0)**2 + (latval[iy,ix] - lat0)**2
    is minimum.  This "closeness" measure works badly near poles and
    longitude boundaries.
    '''
    # Read from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    ny,nx = latvals.shape
    dist_sq_min = 1.0e30
    for iy in range(ny):
        for ix in range(nx):
            latval = latvals[iy, ix]
            lonval = lonvals[iy, ix]
            dist_sq = (latval - lat0)**2 + (lonval - lon0)**2
            if dist_sq < dist_sq_min:
                iy_min, ix_min, dist_sq_min = iy, ix, dist_sq
    return iy_min,ix_min

ncfile = netCDF4.Dataset(filename, 'r')
latvar = ncfile.variables['Latitude']
lonvar = ncfile.variables['Longitude']
iy,ix = naive_slow(latvar, lonvar, 50.0, -140.0)
print 'Closest lat lon:', latvar[iy,ix], lonvar[iy,ix]
tempvar = ncfile.variables['temperature']
salvar = ncfile.variables['salinity']
print 'temperature:', tempvar[0, 0, iy, ix], tempvar.units
print 'salinity:', salvar[0, 0, iy, ix], salvar.units
ncfile.close()

import numpy as np
import netCDF4

def naive_fast(latvar,lonvar,lat0,lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    ny,nx = latvals.shape
    dist_sq = (latvals-lat0)**2 + (lonvals-lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min,ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return iy_min,ix_min

ncfile = netCDF4.Dataset(filename, 'r')
latvar = ncfile.variables['Latitude']
lonvar = ncfile.variables['Longitude']
iy,ix = naive_fast(latvar, lonvar, 50.0, -140.0)
print 'Closest lat lon:', latvar[iy,ix], lonvar[iy,ix]
ncfile.close()

import numpy as np
import netCDF4
from math import pi
from numpy import cos, sin

def tunnel_fast(latvar,lonvar,lat0,lon0):
    '''
    Find closest point in a set of (lat,lon) points to specified point
    latvar - 2D latitude variable from an open netCDF dataset
    lonvar - 2D longitude variable from an open netCDF dataset
    lat0,lon0 - query point
    Returns iy,ix such that the square of the tunnel distance
    between (latval[it,ix],lonval[iy,ix]) and (lat0,lon0)
    is minimum.
    '''
    rad_factor = pi/180.0 # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor
    ny,nx = latvals.shape
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    # Compute numpy arrays for all values, no loops
    clat,clon = cos(latvals),cos(lonvals)
    slat,slon = sin(latvals),sin(lonvals)
    delX = cos(lat0_rad)*cos(lon0_rad) - clat*clon
    delY = cos(lat0_rad)*sin(lon0_rad) - clat*slon
    delZ = sin(lat0_rad) - slat;
    dist_sq = delX**2 + delY**2 + delZ**2
    minindex_1d = dist_sq.argmin()  # 1D index of minimum element
    iy_min,ix_min = np.unravel_index(minindex_1d, latvals.shape)
    return iy_min,ix_min

ncfile = netCDF4.Dataset(filename, 'r')
latvar = ncfile.variables['Latitude']
lonvar = ncfile.variables['Longitude']
iy,ix = tunnel_fast(latvar, lonvar, 50.0, -140.0)
print 'Closest lat lon:', latvar[iy,ix], lonvar[iy,ix]
ncfile.close()

import numpy as np
import netCDF4
from math import pi
from numpy import cos, sin
from scipy.spatial import cKDTree

def kdtree_fast(latvar,lonvar,lat0,lon0):
    rad_factor = pi/180.0 # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor
    ny,nx = latvals.shape
    clat,clon = cos(latvals),cos(lonvals)
    slat,slon = sin(latvals),sin(lonvals)
    # Build kd-tree from big arrays of 3D coordinates
    triples = zip(np.ravel(clat*clon), np.ravel(clat*slon), np.ravel(slat))
    kdt = cKDTree(triples)
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    clat0,clon0 = cos(lat0_rad),cos(lon0_rad)
    slat0,slon0 = sin(lat0_rad),sin(lon0_rad)
    dist_sq_min, minindex_1d = kdt.query([clat0*clon0, clat0*slon0, slat0])
    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
    return iy_min,ix_min
                
ncfile = netCDF4.Dataset(filename, 'r')
latvar = ncfile.variables['Latitude']
lonvar = ncfile.variables['Longitude']
iy,ix = kdtree_fast(latvar, lonvar, 50.0, -140.0)
print 'Closest lat lon:', latvar[iy,ix], lonvar[iy,ix]
ncfile.close()

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename,'r');latvar = ncfile.variables['Latitude'];lonvar = ncfile.variables['Longitude']", 'naive_slow(latvar, lonvar, 50.0, -140.0)')

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename,'r');latvar = ncfile.variables['Latitude'];lonvar = ncfile.variables['Longitude']", 'naive_fast(latvar, lonvar, 50.0, -140.0)')

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename,'r');latvar = ncfile.variables['Latitude'];lonvar = ncfile.variables['Longitude']", 'tunnel_fast(latvar, lonvar, 50.0, -140.0)')

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename,'r');latvar = ncfile.variables['Latitude'];lonvar = ncfile.variables['Longitude']", 'kdtree_fast(latvar, lonvar, 50.0, -140.0)')

ncfile.close()

# Split naive_slow into initialization and query, so we can time them separately
import numpy as np
import netCDF4

class Naive_slow(object):
    def __init__(self, ncfile, latvarname, lonvarname):
        self.ncfile = ncfile
        self.latvar = self.ncfile.variables[latvarname]
        self.lonvar = self.ncfile.variables[lonvarname]
        # Read latitude and longitude from file into numpy arrays
        self.latvals = self.latvar[:]
        self.lonvals = self.lonvar[:]
        self.shape = self.latvals.shape

    def query(self,lat0,lon0):
        ny,nx = self.shape
        dist_sq_min = 1.0e30
        for iy in range(ny):
            for ix in range(nx):
                latval = self.latvals[iy, ix]
                lonval = self.lonvals[iy, ix]
                dist_sq = (latval - lat0)**2 + (lonval - lon0)**2
                if dist_sq < dist_sq_min:
                    iy_min, ix_min, dist_sq_min = iy, ix, dist_sq
        return iy_min,ix_min

ncfile = netCDF4.Dataset(filename, 'r')
ns = Naive_slow(ncfile,'Latitude','Longitude')
iy,ix = ns.query(50.0, -140.0)
print 'Closest lat lon:', ns.latvar[iy,ix], ns.lonvar[iy,ix]
ncfile.close()

# Split naive_fast into initialization and query, so we can time them separately
import numpy as np
import netCDF4

class Naive_fast(object):
    def __init__(self, ncfile, latvarname, lonvarname):
        self.ncfile = ncfile
        self.latvar = self.ncfile.variables[latvarname]
        self.lonvar = self.ncfile.variables[lonvarname]        
        # Read latitude and longitude from file into numpy arrays
        self.latvals = self.latvar[:]
        self.lonvals = self.lonvar[:]
        self.shape = self.latvals.shape
        

    def query(self,lat0,lon0):
        dist_sq = (self.latvals-lat0)**2 + (self.lonvals-lon0)**2
        minindex_flattened = dist_sq.argmin()                             # 1D index
        iy_min, ix_min = np.unravel_index(minindex_flattened, self.shape) # 2D indexes
        return iy_min,ix_min

ncfile = netCDF4.Dataset(filename, 'r')
ns = Naive_fast(ncfile,'Latitude','Longitude')
iy,ix = ns.query(50.0, -140.0)
print 'Closest lat lon:', ns.latvar[iy,ix], ns.lonvar[iy,ix]
ncfile.close()

# Split tunnel_fast into initialization and query, so we can time them separately
import numpy as np
import netCDF4
from math import pi
from numpy import cos, sin

class Tunnel_fast(object):
    def __init__(self, ncfile, latvarname, lonvarname):
        self.ncfile = ncfile
        self.latvar = self.ncfile.variables[latvarname]
        self.lonvar = self.ncfile.variables[lonvarname]        
        # Read latitude and longitude from file into numpy arrays
        rad_factor = pi/180.0 # for trignometry, need angles in radians
        self.latvals = self.latvar[:] * rad_factor
        self.lonvals = self.lonvar[:] * rad_factor
        self.shape = self.latvals.shape
        clat,clon,slon = cos(self.latvals),cos(self.lonvals),sin(self.lonvals)
        self.clat_clon = clat*clon
        self.clat_slon = clat*slon
        self.slat = sin(self.latvals)
 
    def query(self,lat0,lon0):
        # for trignometry, need angles in radians
        rad_factor = pi/180.0 
        lat0_rad = lat0 * rad_factor
        lon0_rad = lon0 * rad_factor
        delX = cos(lat0_rad)*cos(lon0_rad) - self.clat_clon
        delY = cos(lat0_rad)*sin(lon0_rad) - self.clat_slon
        delZ = sin(lat0_rad) - self.slat;
        dist_sq = delX**2 + delY**2 + delZ**2
        minindex_1d = dist_sq.argmin()                              # 1D index 
        iy_min, ix_min = np.unravel_index(minindex_1d, self.shape)  # 2D indexes
        return iy_min,ix_min

ncfile = netCDF4.Dataset(filename, 'r')
ns = Tunnel_fast(ncfile,'Latitude','Longitude')
iy,ix = ns.query(50.0, -140.0)
print 'Closest lat lon:', ns.latvar[iy,ix], ns.lonvar[iy,ix]
ncfile.close()

# Split kdtree_fast into initialization and query, so we can time them separately
import numpy as np
import netCDF4
from math import pi
from numpy import cos, sin
from scipy.spatial import cKDTree

class Kdtree_fast(object):
    def __init__(self, ncfile, latvarname, lonvarname):
        self.ncfile = ncfile
        self.latvar = self.ncfile.variables[latvarname]
        self.lonvar = self.ncfile.variables[lonvarname]        
        # Read latitude and longitude from file into numpy arrays
        rad_factor = pi/180.0 # for trignometry, need angles in radians
        self.latvals = self.latvar[:] * rad_factor
        self.lonvals = self.lonvar[:] * rad_factor
        self.shape = self.latvals.shape
        clat,clon = cos(self.latvals),cos(self.lonvals)
        slat,slon = sin(self.latvals),sin(self.lonvals)
        clat_clon = clat*clon
        clat_slon = clat*slon
        triples = zip(np.ravel(clat*clon), np.ravel(clat*slon), np.ravel(slat))
        self.kdt = cKDTree(triples)

    def query(self,lat0,lon0):
        rad_factor = pi/180.0 
        lat0_rad = lat0 * rad_factor
        lon0_rad = lon0 * rad_factor
        clat0,clon0 = cos(lat0_rad),cos(lon0_rad)
        slat0,slon0 = sin(lat0_rad),sin(lon0_rad)
        dist_sq_min, minindex_1d = self.kdt.query([clat0*clon0,clat0*slon0,slat0])
        iy_min, ix_min = np.unravel_index(minindex_1d, self.shape)
        return iy_min,ix_min

ncfile = netCDF4.Dataset(filename, 'r')
ns = Kdtree_fast(ncfile,'Latitude','Longitude')
iy,ix = ns.query(50.0, -140.0)
print 'Closest lat lon:', ns.latvar[iy,ix], ns.lonvar[iy,ix]
ncfile.close()

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename, 'r')", "ns = Naive_slow(ncfile,'Latitude','Longitude')")

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename, 'r')", "ns = Naive_fast(ncfile,'Latitude','Longitude')")

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename, 'r')", "ns = Tunnel_fast(ncfile,'Latitude','Longitude')")

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename, 'r')", "ns = Kdtree_fast(ncfile,'Latitude','Longitude')")

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename, 'r'); ns = Naive_slow(ncfile,'Latitude','Longitude')", 'iy,ix = ns.query(50.0, -140.0)')

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename, 'r'); ns = Naive_fast(ncfile,'Latitude','Longitude')", 'iy,ix = ns.query(50.0, -140.0)')

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename, 'r'); ns = Tunnel_fast(ncfile,'Latitude','Longitude')", 'iy,ix = ns.query(50.0, -140.0)')

get_ipython().run_cell_magic('timeit', "ncfile = netCDF4.Dataset(filename, 'r'); ns = Kdtree_fast(ncfile,'Latitude','Longitude')", 'iy,ix = ns.query(50.0, -140.0)')

ns0,nf0,tf0,kd0 = 3.76, 3.8, 27.4, 2520  # setup times in msec
nsq,nfq,tfq,kdq = 7790, 2.46, 5.14, .0738 # query times in msec

N = 10000
print N, "queries using naive_slow:", round((ns0 + nsq*N)/1000,1), "seconds"
print N, "queries using naive_fast:", round((nf0 + nfq*N)/1000,1), "seconds"
print N, "queries using tunnel_fast:", round((tf0 + tfq*N)/1000,1), "seconds"
print N, "queries using kdtree_fast:", round((kd0 + kdq*N)/1000,1), "seconds"
print
print "kd_tree_fast outperforms naive_fast above:", int((kd0-nf0)/(nfq-kdq)), "queries"
print "kd_tree_fast outperforms tunnel_fast above:", int((kd0-tf0)/(tfq-kdq)), "queries"


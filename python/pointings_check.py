get_ipython().magic('matplotlib inline')
import numpy as np
import healpy as hp
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.utils import hpid2RaDec, _healbin

# LSST has almost 10 sq degree FOv
nside = 128
pixArea = hp.nside2pixarea(nside, degrees=True)
npix_per_fov = 10./pixArea
print npix_per_fov

# Let's make an array where all the southern hemisphere is pointed at this many times
ra, dec = hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
good = np.where(dec < 30)
ra = ra[good]
dec = dec[good]

ra.size

names = ['fieldRA', 'fieldDec']
types = [float] * 2
simdata = np.zeros(ra.size, dtype=zip(names, types))
simdata['fieldRA'] = np.radians(ra)
simdata['fieldDec'] = np.radians(dec)

metric = metrics.CountMetric(col='fieldRA')
slicer_nside = nside
slicer = slicers.HealpixSlicer(nside=slicer_nside)
sql=''
plotDict = {'colorMin':41, 'colorMax': 50}
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict)

bgroup = metricBundles.MetricBundleGroup({0:bundle}, None, saveEarly=False)
bgroup.setCurrent('')
bgroup.runCurrent('', simData=simdata)
bgroup.plotAll(closefigs=False)

simdata.size

# from https://people.sc.fsu.edu/~jburkardt/py_src/sphere_fibonacci_grid/sphere_fibonacci_grid_points.py
def sphere_fibonacci_grid_points ( ng ):

#*****************************************************************************80
#
## SPHERE_FIBONACCI_GRID_POINTS: Fibonacci spiral gridpoints on a sphere.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 May 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Richard Swinbank, James Purser,
#    Fibonacci grids: A novel approach to global modelling,
#    Quarterly Journal of the Royal Meteorological Society,
#    Volume 132, Number 619, July 2006 Part B, pages 1769-1793.
#
#  Parameters:
#
#    Input, integer NG, the number of points.
#
#    Output, real XG(3,N), the grid points.
#
  import numpy as np

  phi = ( 1.0 + np.sqrt ( 5.0 ) ) / 2.0

  theta = np.zeros ( ng )
  sphi = np.zeros ( ng )
  cphi = np.zeros ( ng )

# Jubus, why is there a loop here?
  for i in range ( 0, ng ):
    i2 = 2 * i - ( ng - 1 ) 
    theta[i] = 2.0 * np.pi * float ( i2 ) / phi
    sphi[i] = float ( i2 ) / float ( ng )
    cphi[i] = np.sqrt ( float ( ng + i2 ) * float ( ng - i2 ) ) / float ( ng )

  xg = np.zeros ( ( ng, 3 ) )

  for i in range ( 0, ng ) :
    xg[i,0] = cphi[i] * np.sin ( theta[i] )
    xg[i,1] = cphi[i] * np.cos ( theta[i] )
    xg[i,2] = sphi[i]

  return xg

def fib_sphere_grid(npoints):
    
    phi = ( 1.0 + np.sqrt ( 5.0 ) ) / 2.0

    # theta = np.zeros(npoints)
    # sphi = np.zeros(npoints)
    # cphi = np.zeros(npoints)
    
    i = np.arange(npoints, dtype=float)
    i2 = 2*i - (npoints-1)
    theta = (2.0*np.pi * i2/phi) % (2.*np.pi)
    sphi = i2/npoints
    # cphi = ((npoints+i2) * (npoints - i2))**0.5 / npoints
    #x = cphi*np.sin(theta)
    #y = cphi*np.cos(theta)
    #z = sphi
    phi = np.arccos(sphi)
    dec = np.pi/2. - phi
    return theta, dec

theta, phi = fib_sphere_grid(100000)





gridMap = _healbin(theta, phi, phi*0+1, reduceFunc=np.sum, nside=8)

hp.mollview(gridMap)




# Basic notebook imports
get_ipython().magic('matplotlib inline')

import matplotlib
import pylab as plt
import numpy as np
import healpy as hp

# Import skymap and some of it's basic map classes
import skymap

smap = skymap.Skymap()

def skymap_test(smap):
    """ Some simple test cases. """
    plt.gca()
    # Draw some scatter points
    smap.scatter([0,45,-30],[0,-45,-30],latlon=True)

    # Draw a tissot (projected circle)
    smap.tissot(-60,30,10,100,facecolor='none',edgecolor='b',lw=2)

    # Draw a color mesh image (careful, basemap is quirky)
    x = y = np.arange(30,60)
    xx,yy = np.meshgrid(x,y)
    z = xx*yy
    smap.pcolormesh(xx,yy,data=z,cmap='gray_r',latlon=True)
    
skymap_test(smap)

fig,axes = plt.subplots(2,2,figsize=(14,8))

# A nice projection for plotting the visible sky
plt.sca(axes[0,0])
smap = skymap.Skymap(projection='ortho',lon_0=0, lat_0=0)
skymap_test(smap)
plt.title('Orthographic')

# A common equal area all-sky projection
plt.sca(axes[1,0])
smap = skymap.Skymap(projection='hammer',lon_0=0, lat_0=0)
skymap_test(smap)
plt.title("Hammer-Aitoff")

# Something wacky that I've never used
plt.sca(axes[0,1])
smap = skymap.Skymap(projection='sinu',lon_0=0, lat_0=0)
skymap_test(smap)
plt.title("Sinusoidal")

# My favorite projection for DES
plt.sca(axes[1,1])
smap = skymap.Skymap(projection='mbtfpq',lon_0=0, lat_0=0)
skymap_test(smap)
plt.title("McBryde-Thomas Flat Polar Quartic")


from astroquery.sdss import SDSS
from astroquery.ned import Ned
from astropy import coordinates as coords
import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

result_table = Ned.query_object('NGC 5353')

print result_table.columns

print result_table['RA(deg)'][0]

pos = coords.SkyCoord(ra=result_table['RA(deg)'][0]*u.deg,dec=result_table['DEC(deg)'][0]*u.deg, frame='icrs')

xid = SDSS.query_region(pos)

print xid

print xid['ra'][0:-1]

sdsscoords = coords.SkyCoord(ra = xid['ra']*u.deg, dec=xid['dec']*u.deg,frame='icrs')

distance = pos.separation(sdsscoords)
match = (distance == min(distance))
print match
print sdsscoords[match]

im = SDSS.get_images(matches = xid[match], band='g')

print im

plt.imshow(im[0][1])




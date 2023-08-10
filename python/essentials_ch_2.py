import numpy as np
import pandas as pd
deg2rad=np.pi/180. # converts degrees to radians

def dir2cart(data):
    decs,incs,ints=data[0]*deg2rad,data[1]*deg2rad,data[2]
    X=ints*np.cos(decs)*np.cos(incs)
    Y=ints*np.sin(decs)*np.cos(incs)
    Z=ints*np.sin(incs)
    cart=np.array([X,Y,Z]).transpose()
    return cart

# read in the data and transpose it to rows of dec, inc, int
data=np.loadtxt('Chapter_2/ps2_prob1_data.txt').transpose()
print dir2cart(data)

import pmagpy.pmag as pmag
print pmag.get_unf.__doc__

places=pmag.get_unf(10)
print places

import pmagpy.ipmag
print ipmag.igrf.__doc__

for place in places:
    print ipmag.igrf([2006,0,place[1],place[0]])

data=[] # make a blank list
for place in places:
    Dir=ipmag.igrf([2006,0,place[1],place[0]])
    data.append(Dir) # append to the data list
data=np.array(data).transpose() # dir2cart takes arrays of data
print dir2cart(data)

import pmagplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

ipmag.plot_net(1) # make an equal angle net
ipmag.plot_di(data[0],data[1]) # put on the dots

lat = 36.*deg2rad # remember to convert to radians!
inc = np.arctan(2.*np.tan(lat)) /deg2rad # and back! 
print '%7.1f'%(inc) # and print it out

print pmag.dia_vgp.__doc__

vgp_lat,vgp_lon,dp,dp= pmag.dia_vgp(345,47,0.,36,-112) 
print '%7.1f %7.1f'%(vgp_lat,vgp_lon)




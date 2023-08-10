get_ipython().run_line_magic('run', './python/nb_init.py')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import geography as geo
import data_handling as dh
from property_map import pmap
from MCSim import SimFamily, PostGroup, PostData

# Unit conversions
#0.82*(75*unit.inch).to(unit.m).to(unit.inch)
(12*unit.m/unit.s).to(unit.knot)

# UTM to LatLon
import utm
utm.to_latlon(444005.64156725001521, 5889683.58956928178668+2000, 32, 'U')

# Rate of descent
(10*unit.m).to(unit.ft).magnitude / 105

# Other
theta = np.radians(15)
V = 12

P1 = (-400, 400*np.tan(theta), 0)
P2 = (2800, -2800*np.tan(theta), 0)
vV = (V*np.cos(theta), -V*np.sin(theta), 0)

print(P1, P2, vV, sep='\n')

8000*10/3600

D = {'3': [0,1,3], '5': [8,4,6], '0': [3,6,1]}
for key, val in D.items(): D[key] = np.array(val)
    
inds = D['0'].argsort()

for key, val in D.items(): D[key] = val[inds]
    
D['0']

100/126




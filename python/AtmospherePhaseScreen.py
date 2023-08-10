import math
import numpy as np
import ceo
get_ipython().magic('pylab inline')

atm = ceo.Atmosphere(0.15,30,altitude=10e3,wind_speed=10,wind_direction=0)

n = 64
src = ceo.Source("V",resolution=(n,n))

D = 4.0
zen_oa = D/10e3
src_oa = ceo.Source("V",zenith=zen_oa,azimuth=0,resolution=(n,n))

L = 8.0
p = L/(n-1)
atm.get_phase_screen(src,p,n,p,n,0.0)
atm.get_phase_screen(src_oa,p,n,p,n,0.0)

fig,ax = subplots()
img1 = ax.imshow(src.phase.host(units='micron'),aspect='equal',extent=[0,L,0,L])
img1.set_clim((-5,1.5))
img2 = ax.imshow(src_oa.phase.host(units='micron'),aspect='equal',extent=[0,L,-D,L-D])
img2.set_clim((-5,1.5))
ax.set_xlim(0,L)
ax.set_ylim(-D,L)
fig.colorbar(img2, ax=ax)

srcCombo = ceo.Source("V",zenith=[0,zen_oa],azimuth=[0,0],resolution=(n,n))
atm.get_phase_screen(srcCombo,p,n,p,n,0.0)
fig,ax = subplots()
img = ax.imshow(srcCombo.phase.host(units='micron'))
img.set_clim((-5,1.5))
fig.colorbar(img, ax=ax)




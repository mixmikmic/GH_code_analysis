get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd 
from pylab import *

#Standard NDVI
def compute_ndvi(RED,NIR):
    NDVI=(NIR-RED)/(NIR+RED)
    return NDVI

#Modified NDVI
def modified_ndvi(RED,NIR):
    RED=RED+0.05
    NDVI_MOD=(NIR-RED)/(NIR+RED)
    return NDVI_MOD

nb=10000

#random simulations of RED and NIR surface reflectances (uniform probability law)
RED=np.random.random_sample(nb)*0.1+0.01
NIR=np.random.random_sample(nb)*0.5+0.15

#Random simulation of noise with a gauwwian law with std=0.02 (a little bit exaggerated)
NoiseR=np.random.normal(0,0.02,nb)
NoiseN=np.random.normal(0,0.02,nb)

#Add noise to reflectance
RED_noise=RED+NoiseR
NIR_noise=NIR+NoiseN

#Cast negative values to zero
RED_noise=np.where(RED_noise<0,0,RED_noise)
NIR_noise=np.where(NIR_noise<0,0,NIR_noise)

ndvi_ref=compute_ndvi(RED,NIR)
ndvi_noise=compute_ndvi(RED_noise,NIR_noise)


plot(ndvi_ref,ndvi_noise,'.')
ylim(-0.1,1.2)
xlim(-0.1,1.2)
title('Standard NDVI')
xlabel('NDVI')
ylabel('NDVI+Noise')
show()

ndvi_ref=modified_ndvi(RED,NIR)
ndvi_noise=modified_ndvi(RED_noise,NIR_noise)


plot(ndvi_ref,ndvi_noise,'.')
ylim(-0.1,1.2)
xlim(-0.1,1.2)
title('Modified NDVI')
xlabel('NDVI')
ylabel('NDVI+Noise')
show()


from skimage.exposure import rescale_intensity
import numpy as np
import holoviews as hv
hv.extension('bokeh')

volumeseries = np.random.randint(0,255,(10,20,300,300))

def select_t_z(t,z):
    tmp = volumeseries[t,z,: , :]
    return  hv.Image(tmp)

hv.DynamicMap(select_t_z, kdims=['t','z', ]).redim.values(t=range(10), z=range(20))

# The larger the range of the intensity sliders, the longer this takes ages to run ...
# 

volumeseries = np.random.randint(0,255,(10,20,600,300))

def select_t_z(t,z,cmap, vmin, vmax):
    tmp_xy = rescale_intensity(volumeseries[t,z,: , :], (vmin,vmax), (0,255) )
    
    return  hv.Image(tmp_xy).options(cmap=cmap) 
    
hv.DynamicMap(select_t_z, kdims=['t','z', 'cmap', 'vmin','vmax']).redim.values(t=range(10), z=range(20) ,cmap=['viridis','Greys'], vmin=range(128), vmax=range(128,256))

hv.__version__


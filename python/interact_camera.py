get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from ctapipe import io, visualization
import numpy as np
from ctapipe.utils.datasets import get_example_simtelarray_file
import astropy.units as u
from ipywidgets import interact

src=io.hessio.hessio_event_source(get_example_simtelarray_file())


evt=next(src)
evt=next(src)
tels=list(evt.dl0.tels_with_data)

#for evt in src:
#    print(evt)

def plot_evt(t,tel_id):
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(15,6))
    
    cube=evt.dl0.tel[int(tel_id)].adc_samples[0]
    x, y = evt.meta.pixel_pos[int(tel_id)]
    geom = io.CameraGeometry.guess(x, y, 2.3*u.meter)
    disp = visualization.CameraDisplay(geom,ax=ax1)
    disp.image = cube[:,t]
    disp.update()

    ax2.plot( np.sum(cube,axis=0) )
    ax2.axvline(t,ls=':',color='k')
    ax2.set_ylabel('Summed ADC')    
    ax2.set_xlabel('Time slice')
    plt.show()

interact(plot_evt,t=(0,24,1),tel_id=tels)

get_ipython().magic('load_ext line_profiler')
get_ipython().magic('lprun -f plot_evt plot_evt(12,11)')




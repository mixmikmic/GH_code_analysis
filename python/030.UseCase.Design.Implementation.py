# some python modules for live data simulation (DataSource)
import math
import random
import numpy as np
import logging
# SOLEIL's scanning API
from datastream.plots import ChannelData, DataSource, BoxSelectionManager
from datastream.plots import Scale, SpectrumChannel, ImageChannel
from datastream.plots import DataStream, DataStreamer, DataStreamerController

# 1D data source for the X axis (here X scale is one of the DataSource)
class XDS(DataSource):
    
    def __init__(self, name, num_points=128):
        DataSource.__init__(self, name)
        self._l = num_points
    
    def pull_data(self): 
        cd = ChannelData(self.name)
        start = random.uniform(-math.pi/2, math.pi/2)
        end = 2 * math.pi + start
        cd.buffer = np.linspace(start, end, self._l)
        return cd
    
# 1D data source for the Y axis
class YDS(DataSource):
    
    def __init__(self, name, channels=None, num_points=128):
        DataSource.__init__(self, name)
        self._l = num_points
    
    def pull_data(self): 
        cd = ChannelData(self.name)
        p = random.uniform(-math.pi/2, math.pi/2)
        start = 0 + p
        end = 2 * (math.pi + p)
        x = np.linspace(start, end, self._l)
        d = random.uniform(1.0, 4.0) * np.sin(x)
        cd.buffer = random.uniform(1.0, 4.0) * np.sin(x)
        return cd

# 2D (i.e. image data source)
class XYDS(DataSource):
    
    def __init__(self, name, num_points=128):
        DataSource.__init__(self, name)
        self._selection = None
        self._inc = 1
        self._current_index = 0
        self._iw, self._ih = num_points, num_points
        x, y = np.linspace(0, 10, self._iw), np.linspace(0, 10, self._ih)
        xx, yy = np.meshgrid(x, y)
        self._full_image = np.sin(xx) * np.cos(yy)

    def pull_data(self):
        cd = ChannelData(self.name)    
        i = self._current_index
        cd.buffer = self._full_image[0:i+1, 0:i+1]
        self._current_index += self._inc
        if self._current_index > self._ih:
            self._current_index = self._ih
            self._inc *= -1 
        elif self._current_index < 0:
            self._current_index = 0
            self._inc *= -1
        return cd
    
    def scb(self, selection):
        self._selection = selection
        
    def rcb(self):
        self._selection = None

# Model (i.e plot) properties for the spectrum channel
def spc_model_props():
    shsp = dict()
    shsp['label'] = 'angle'
    shsp['unit'] = 'rad'
    shsp['channel'] = 'x_scale'
    x_scale = Scale(**shsp)
    spsp = dict()
    spsp['label'] = 'amplitude'
    spsp['unit'] = 'a.u.'
    y_scale = Scale(**spsp)
    smp = dict()
    smp['width'] = 900
    smp['height'] = 250
    smp['x_scale'] = x_scale
    smp['y_scale'] = y_scale
    return smp

# Model (i.e plot) properties for the image channel
def img_model_props(s=-100, e=100, this=None):
    xshsp = dict()
    xshsp['start'] = s
    xshsp['end'] = e
    xshsp['num_points'] = abs(e - s)
    xshsp['label'] = 'x-axis'
    xshsp['unit'] = 'mm'
    yshsp = dict()
    yshsp['start'] = 0
    yshsp['end'] = max(s, e)
    yshsp['num_points'] = abs(e - s)
    yshsp['label'] = 'y-axis'
    yshsp['unit'] = 'mm'
    imp = dict()    
    imp['width'] = 900
    imp['height'] = 250
    imp['x_scale'] = Scale(**xshsp)
    imp['y_scale'] = Scale(**yshsp)
    if this:
        imp['selection_manager'] = BoxSelectionManager(selection_callback=this.scb, reset_callback=this.rcb)
    return imp

npts = 128
# SpectrumChannel (has multiple DataSource support)
sds = list()
sds.append(XDS('x_scale', num_points=npts))
sds.extend([YDS(n,  num_points=npts) for n in ['y1', 'y2', 'y3']])
sc = SpectrumChannel('sc', data_sources=sds, model_properties=spc_model_props())
# ImageChannel (supports only one DataSource)
ids = XYDS("is0")
ic = ImageChannel("ic", data_source=ids, model_properties=img_model_props(this=ids))
# DataStream (has with multiple Channel support)
s1 = DataStream('s1', channels=[sc,ic])
# DataStreamer (has with multiple DataStream support)
m1 = DataStreamer('m1', data_streams=[s1], update_period=0.25)
# DataStreamerController (optional widgets to control the DataStreamer)
c1 = DataStreamerController('c1', m1)

logging.getLogger('bokeh').setLevel(logging.ERROR)
logging.getLogger('tornado').setLevel(logging.ERROR)
logging.getLogger('fs.client.jupyter').setLevel(logging.DEBUG)


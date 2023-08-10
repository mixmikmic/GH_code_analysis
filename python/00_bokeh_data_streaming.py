# some python modules for live data simulation (DataSource)
import math
import random
import numpy as np
import logging
# bokeh plots data streaming API (see plots.py)
from common.plots import ChannelData, DataSource, BoxSelectionManager
from common.plots import Scale, SpectrumChannel, ImageChannel
from common.plots import DataStream, DataStreamer, DataStreamerController

# 1D data source for the X axis (here X scale is one of the DataSource)
class XDS(DataSource):
    
    def __init__(self, name, num_points=128):
        DataSource.__init__(self, name)
        self._l = num_points
        self._cnt = 0
        
    def pull_data(self):
        cd = ChannelData(self.name)
        if self._cnt < 10:
            cd.set_error("Don't worry we are simply testing the error handling widgets", None)
        else:
            start = random.uniform(-math.pi/2, math.pi/2)
            end = 2 * math.pi + start
            cd.buffer = np.linspace(start, end, self._l)
        self._cnt += 1
        return cd
    
# 1D data source for the Y axis
class YDS(DataSource):
    
    def __init__(self, name, channels=None, num_points=128):
        DataSource.__init__(self, name)
        self._l = num_points
        self._cnt = 0
    
    def pull_data(self): 
        cd = ChannelData(self.name)
        if self._cnt < 10:
            cd.set_error("Don't worry we are simply testing the error handling widgets", None)
        else:
            p = random.uniform(-math.pi/2, math.pi/2)
            start = 0 + p
            end = 2 * (math.pi + p)
            x = np.linspace(start, end, self._l)
            d = random.uniform(1.0, 4.0) * np.sin(x)
            cd.buffer = random.uniform(1.0, 4.0) * np.sin(x)
        self._cnt += 1
        return cd

# 2D (i.e. image data source)
class XYDS(DataSource):
    
    def __init__(self, name, nx=128, ny=128):
        DataSource.__init__(self, name)
        self._selection = None
        self._inc = 1
        self._current_index = 0
        self._iw, self._ih = nx, ny
        x, y = np.linspace(0, 10, self._iw), np.linspace(0, 10, self._ih)
        xx, yy = np.meshgrid(x, y)
        self._full_image = np.sin(xx) * np.cos(yy)
        self._cnt = 0

    def pull_data(self):
        cd = ChannelData(self.name) 
        if self._cnt in range(10,15) or self._cnt in range(20,25):
            cd.set_error("Don't worry we are simply testing the error handling widgets", None)
        else:
            i = self._current_index
            cd.buffer = self._full_image[0:i+1, :]
            self._current_index += self._inc
            if self._current_index > self._ih:
                self._current_index = self._ih
                self._inc *= -1 
            elif self._current_index < 0:
                self._current_index = 0
                self._inc *= -1
        self._cnt += 1
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
def img_model_props(s=-100, e=100, nx=128, ny=128, this=None):
    xshsp = dict()
    xshsp['start'] = s
    xshsp['end'] = e
    xshsp['num_points'] = nx
    xshsp['label'] = 'x-axis'
    xshsp['unit'] = 'mm'
    yshsp = dict()
    yshsp['start'] = 0
    yshsp['end'] = max(s, e)
    yshsp['num_points'] = ny
    yshsp['label'] = 'y-axis'
    yshsp['unit'] = 'mm'
    imp = dict()    
    imp['width'] = 900
    imp['height'] = 250
    imp['x_scale'] = Scale(**xshsp)
    imp['y_scale'] = Scale(**yshsp)
    imp['images_size_threshold'] = 100000
    if this:
        imp['selection_manager'] = BoxSelectionManager(selection_callback=this.scb, reset_callback=this.rcb)
    return imp

i = 0
def open_plot(output):
    global i
    i += 1
    # num. of points in ColumnDataSource
    npts = 128
    # SpectrumChannel (has multiple DataSource support)
    sds = list()
    sds.append(XDS('x_scale', num_points=npts))
    sds.extend([YDS(n,  num_points=npts) for n in ['y1.1', 'y1.2', 'y1.3']])
    sc1 = SpectrumChannel('sc1', data_sources=sds, model_properties=spc_model_props())
    # SpectrumChannel (has multiple DataSource support)
    sds = list()
    sds.append(XDS('x_scale', num_points=npts))
    sds.extend([YDS(n,  num_points=npts) for n in ['y2.1', 'y2.2', 'y2.3']])
    sc2 = SpectrumChannel('sc2', data_sources=sds, model_properties=spc_model_props())
    # ImageChannel (supports only one DataSource)
    ids = XYDS("is0", nx=npts, ny=npts)
    #import IPython.core.debugger as debugger; debugger.set_trace()
    ic = ImageChannel('ic{}'.format(i), data_source=ids, model_properties=img_model_props(nx=npts, ny=npts, this=ids))
    # DataStream (has with multiple Channel support)
    s1 = DataStream('s{}'.format(i), channels=[sc1 if i % 2 else ic]) #[sc1, sc2, ic])
    # DataStreamer (has with multiple DataStream support)
    ds1 = DataStreamer('ds{}'.format(i), data_streams=[s1], update_period=1.0,  auto_start=False)
    # DataStreamerController
    kwargs =  {'output':output, 'title':'DataStreamerController #{}'.format(i), 'auto_start':True}
    dc1 = DataStreamerController('c{}'.format(i), data_streamer=ds1, **kwargs)
    return dc1

import logging
logging.getLogger('fs.client.jupyter').setLevel(logging.DEBUG)

import time
import ipywidgets as ipw
from IPython.display import display, clear_output
from common.tools import *

class OpenClosePlots(NotebookCellContent):
    
    def __init__(self):
        NotebookCellContent.__init__(self)
        self._p1 = None
        self._p2 = None
        self._o = ipw.Output()
        self._o.layout.border = "1px solid green"
        self._b = ipw.Button(description="Open Plots")
        self._b.on_click(self.open_plots_clicked)
        self._c = ipw.Button(description="Close Plots")
        self._c.on_click(self.close_plots_clicked)
        buttons = ipw.HBox(children=[self._b, self._c])
        self._l = ipw.VBox(children=[buttons, self._o])
        display(self._l)
        
    def plot_closed_by_user(self, **kwargs):
        try:
            self.printout(kwargs)
        except Exception as e:
            self.error(e)
            
    def open_plots_clicked(self, b):
        try:
            self._p1 = open_plot(self._o)
            self._p2 = open_plot(self._o)
        except Exception as e:
            self.error(e)

    def close_plots_clicked(self, b):
        try:
            if self._p1:    
                self._p1.close()
                self._p1 = None
            if self._p2:    
                self._p2.close()
                self._p2 = None
            self._o.clear_output()
        except Exception as e:
            self.error(e)

# ugly but mandatory: select the context in which we are running: NOTEBOOK or LAB
import os
os.environ["JUPYTER_CONTEXT"] = "LAB"

w0 = OpenClosePlots()


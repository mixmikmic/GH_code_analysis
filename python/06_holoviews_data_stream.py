import sys
import time
import numpy as np
from functools import partial
import ipywidgets as ipw
import holoviews as hv
from holoviews.operation.datashader import regrid
from IPython.display import display, clear_output
from tornado.ioloop import IOLoop, PeriodicCallback
from bokeh.models import DatetimeTickFormatter

import logging
logging.getLogger('bokeh').setLevel(logging.ERROR)
logging.getLogger('tornado').setLevel(logging.ERROR)
logging.getLogger('fs').setLevel(logging.ERROR)

hv.extension('bokeh', logo=None)

class Output(object):
    def __init__(self):
        k = get_ipython().kernel
        self._ident = k._parent_ident
        self._header = k._parent_header
        self._save_context = None

    def __enter__(self):
        kernel = get_ipython().kernel
        self._save_context = (kernel._parent_ident, kernel._parent_header)
        sys.stdout.flush()
        sys.stderr.flush()
        kernel.set_parent(self._ident, self._header)

    def __exit__(self, etype, evalue, tb):
        sys.stdout.flush()
        sys.stderr.flush()
        kernel = get_ipython().kernel
        kernel.set_parent(*self._save_context)
        return False

    def clear_output(self):
        with self:
            clear_output()

class Scanner(object):
    
    def __init__(self, imw=100, imh=100, fill=np.nan): 
        '''
        imw: image width
        imh: image height
        '''
        # scan progress: num of row added to the image at each iteration
        self._inc = max(1, int(0.05 * imh))
        # scan progress: row at which new data is injected at next iteration
        self._row_index = 0
        # scan image width and height
        self._iw = int(imw)
        self._ih = int(imh)
        self._x = np.linspace(-imw/2, imw/2, imw, True)  
        self._y = np.linspace(-imh/2, imh/2, imh, True)  
        # image buffer (from which scan data is extracted - for simulation purpose)
        xx, yy = np.meshgrid(np.linspace(-10, 10, imw), np.linspace(-10, 10, imh)) 
        self._data_source = np.sin(xx) * np.cos(yy)
        # empty image (full frame)
        self._empty_image = self.empty_image((int(imh), int(imw)), fill)
        # full image (full frame)
        self._full_image = self.__acquire(0, imh)
        
    def empty_image(self, shape=None, fill=np.nan):
        # produce an empty scanner image
        if not shape:
            empty_img = self._empty_image
        else:
            empty_img = np.empty(shape if shape else (self._ih, self._iw))
            empty_img.fill(fill)
        return empty_img
    
    def full_image(self):
        # return 'full' scanner image (simulate scan progress done)
        return self._full_image
    
    def image(self):
        # return 'current' scanner image (simulate scan progress)
        end = self._row_index + self._inc
        image = self.__acquire(None, end)
        self._row_index = end % self._ih
        return image
    
    def __acquire(self, start, end):
        # return scanner image (simulate scan progress)
        s1, s2 = slice(start, end), slice(None)
        image = self._empty_image.copy()
        image[s1, s2] = self._data_source[s1, s2]
        self._row_index = end % self._ih
        return image

    @property
    def speed(self):
        return self._inc
    
    @speed.setter
    def speed(self, s):
        self._inc = max(1, int(min(s, self._ih) / 20))
    
    @property
    def x_scale(self):
        return self._x
    
    @property
    def y_scale(self):
        return self._y

from bokeh.layouts import widgetbox
from bokeh.models.widgets import Slider, Button, TextInput
from functools import partial

PY2=True
if PY2:
    def partial(func, *args, **keywords):
        def newfunc(*fargs, **fkeywords):
            newkeywords = keywords.copy()
            newkeywords.update(fkeywords)
            return func(*(args + fargs), **newkeywords)
        newfunc.func = func
        newfunc.args = args
        newfunc.keywords = keywords
        return newfunc

class ScannerDisplay(object):
    
    def __init__(self, imw=100, imh=100, upp=1.):
        '''
        imw: image width
        imh: image height
        upp: plot update period in seconds  
        '''
        # output (in jupyterlab, this will be a ipywidget.Output)
        self._output = Output()
        # the underlying scanner
        self._scanner = Scanner(imw, imh)
        # async activity
        self._pcb = PeriodicCallback(self.__periodic_task, int(1000.* upp))
        # setup data stream and dynamic interactions
        self._pipe = hv.streams.Pipe(data=(self._scanner.x_scale, self._scanner.y_scale, np.zeros((imh,imw))))
        # dimensions
        dim_x = hv.Dimension('m1', label='Actuator: m1', unit='um')
        dim_y = hv.Dimension('m2', label='Actuator: m2', unit='um')
        self._dmap = hv.DynamicMap(partial(hv.Image, kdims=[dim_x, dim_y]), streams=[self._pipe])
        self._rgxy = hv.streams.RangeXY(source=self._dmap)
        self._rgrd = regrid(self._dmap, streams=[self._pipe, self._rgxy], dynamic=True)
        # start async. activity and display plot
        self.resume()
    
    def __periodic_task(self):
        self._pipe.send((self._scanner.x_scale, self._scanner.y_scale, self._scanner.image()))
    
    @property
    def period(self):
        return self._pcb.callback_time / 1000.
   
    @period.setter
    def period(self, p):
        assert(p >= 1.e-2)
        self._pcb.callback_time = 1000. * p
    
    @property
    def element(self):
        return self._rgrd
        
    def open(self):  
        with self._output:
            display(self._rgrd)
        self.resume()

    def close(self):  
        self.pause()
        self._output.clear_output()
    
    def pause(self):  
        self._pcb.stop()

    def resume(self):  
        self._pcb.start()

hv.opts("Image  (cmap='viridis') [width=550 height=500]")
# scanner image width & height
img_width, img_height = 1000, 2000
# instanciate the ScannerDisplay
sd1 = ScannerDisplay(img_width, img_height)
sd1.open()

sd1.period = 0.1

sd1.pause()

sd1.close()

sd11 = ScannerDisplay(img_width, img_height)
sd12 = ScannerDisplay(img_width, img_height)
sd21 = ScannerDisplay(img_width, img_height)
sd22 = ScannerDisplay(img_width, img_height)

hv.opts("NdLayout [tabs=False] Image (cmap='viridis') [width=450 height=400]")
lo1 = hv.Layout([sd11.element, sd12.element], kdims=['Scanner']).cols(2)
lo2 = hv.Layout([sd21.element, sd22.element], kdims=['Scanner']).cols(2)
hv.Layout(lo1 + lo2).cols(2)

sd11.pause(); sd11.close()
sd12.pause(); sd12.close()
sd21.pause(); sd21.close()
sd22.pause(); sd22.close()

def apply_formatter(plot, element):
    plot.handles['xaxis'].axis_label = 'time'
    plot.handles['xaxis'].formatter = DatetimeTickFormatter()

plot_opts = {
    'Curve':{
        'style':{
            'line_width':1.0
        },
        'plot':{
            'width':950, 
            'height':300, 
            'show_grid':True, 
            'show_legend':True, 
            'finalize_hooks':[apply_formatter]
        }
    }, 
    'Curve.rand1':{
        'style':{
            'color':'darkblue'
        }
    }, 
    'Curve.rand2':{
        'style':{
            'color':'crimson'
        }
    }
}
                                         
data_src_1 = hv.streams.Buffer(np.zeros((0, 2)), length=1024)
data_mon_1 = hv.DynamicMap(partial(hv.Curve, kdims=['time'], vdims=['amplitude'], label='rand1'), streams=[data_src_1])

data_src_2 = hv.streams.Buffer(np.zeros((0, 2)), length=1024)
data_mon_2 = hv.DynamicMap(partial(hv.Curve, kdims=['time'], vdims=['amplitude'], label='rand2'), streams=[data_src_2])

layout = (data_mon_1 * data_mon_2)

def push_data():
    data_src_1.send(np.array([[time.time() * 1000.,  np.random.rand()]]))
    data_src_2.send(np.array([[time.time() * 1000.,  10. * np.random.rand()]]))
    
pcb = PeriodicCallback(push_data, int(250.* 1))

def suspend_resume(b=None):
    if pcb.is_running():
        pcb.stop()
        srb.description = 'Resume'
    else:
        pcb.start()
        srb.description = 'Suspend'
        
def close(b=None):
    pcb.stop()
    out.clear_output()
    
srb = ipw.Button()
srb.on_click(suspend_resume)
cls = ipw.Button(description='Close')
cls.on_click(close)
hbx = ipw.HBox([srb, cls])

out = Output()
with out:
    display(hbx)
    display(layout.opts(plot_opts))
    
suspend_resume()

from common.tools import NotebookCellContent
from common.session import BokehSession

class Monitor(BokehSession, NotebookCellContent):
    
    def __init__(self):
        BokehSession.__init__(self, uuid='mon1')
        NotebookCellContent.__init__(self, name='mon1')
        self._data_src_1 = None
        self._data_src_2 = None
        self._srb = None
        self._widgets = None
        self._plot = None
        self._reset_handler = None
        self.callback_period = 1.0
    
    def periodic_callback(self):
        try:
            self._data_src_1.send(np.array([[time.time() * 1000.,  np.random.rand()]]))
            self._data_src_2.send(np.array([[time.time() * 1000.,  10. * np.random.rand()]]))
        except Exception as e:
            print(e)
            
    def __apply_formatter(plot, element):
        plot.handles['xaxis'].axis_label = 'time'
        plot.handles['xaxis'].formatter = DatetimeTickFormatter()

    def __plot_opts(self):
        def apply_axis_formatter(plot, element):
            plot.handles['xaxis'].axis_label = 'time'
            plot.handles['xaxis'].formatter = DatetimeTickFormatter()
        def twinx(plot, element):
            ax = plot.handles['axis']
            twinax = ax.twinx()
            twinax.set_ylabel(str(element.last.get_dimension(1)))
            plot.handles['axis'] = twinax
        return {
            'Curve': {
                'style': {
                    'line_width':1.0
                },
                'plot': {
                    'width':950, 
                    'height':300, 
                    'show_grid':True, 
                    'show_legend':True,
                    'finalize_hooks':[apply_axis_formatter]
                }
            }, 
            'Curve.rand1': {
                'style': {
                    'color':'darkblue'
                }
            }, 
            'Curve.rand2': {
                'style': {
                    'color':'crimson'
                },
                'plot': {
                    'init_hooks':[twinx]
                }
            }
        }

    def __suspend_resume(self, b):
        if not self.suspended:
            self.pause()
            self._srb.description = 'Resume'
        else:
            self.resume()
            self._srb.description = 'Suspend'
        
    def __close(self, b):
        self.clear_output()
        self.close_output()
        self.cleanup()
        
    def __change_period(self, c):
        self.update_callback_period(c.new)
        
    def __setup_widgets(self):
        self._srb = ipw.Button(description = 'Suspend')
        self._srb.on_click(self.__suspend_resume)
        cls = ipw.Button(description='Close')
        cls.on_click(self.__close)
        sld = ipw.FloatSlider(description='Resfresh (s)', min=0.1, max=1., value=0.5, continuous_update=False)
        sld.observe(self.__change_period, names='value')
        self._widgets = ipw.HBox([sld, self._srb, cls])
        self.display(self._widgets)
            
    def __reset_plot(self, *args, **kwargs):
        print(args)
        print(kwargs)
        self._data_src_1.clear()
        self._data_src_2.clear()
    
    def setup_document(self):
        try:
            self.__setup_widgets()
            self._data_src_1 = s1 = hv.streams.Buffer(np.zeros((0, 2)), length=1024)
            data_mon_1 = hv.DynamicMap(partial(hv.Curve, kdims=['time'], vdims=['a.u.'], label='rand1'), streams=[s1])
            self._data_src_2 = s2 = hv.streams.Buffer(np.zeros((0, 2)), length=1024)
            data_mon_2 = hv.DynamicMap(partial(hv.Curve, kdims=['time'], vdims=['a.u.'], label='rand2'), streams=[s2])
            self._plot = (data_mon_1 * data_mon_2)
            self._rst = hv.DynamicMap(self.__reset_plot, streams = [hv.streams.PlotReset(source=self._plot)])
            self.display(self._plot.opts(self.__plot_opts()))
            #bk_plot = hv.renderer('bokeh').get_plot(self._plot)
            #self.document.add_root(bk_plot, setter=self.bokeh_session_id)
            self.resume()
        except Exception as e:
            print(e)

# ugly but mandatory: select the context in which we are running: NOTEBOOK or LAB
import os
os.environ["JUPYTER_CONTEXT"] = "LAB"

m = Monitor()
m.open()


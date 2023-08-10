from bokeh.io import output_notebook 
from bokeh.resources import INLINE
output_notebook(resources=INLINE)

from common.session import BokehSession

import time
import numpy as np

from IPython.display import clear_output

from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from bokeh.models.glyphs import Rect
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, Button
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import Plasma256, Viridis256
from bokeh.layouts import row, layout, widgetbox


class PatchedImage(BokehSession):
    
    def __init__(self, iw=100, ih=100, bs=1, up=1):
        BokehSession.__init__(self)
        # suspend/resume button
        self._suspend_resume_button = None
        # plot update period
        self.callback_period = up
        # column data source
        self._cds = None
        # bunch size (num of rows in a bunch of data)
        self._inc = bs
        # next row at which we'll patch incoming data
        self._row_index = 0
        # image width and height
        self._iw, self._ih = iw, ih
        # image buffer (from which bunches of data are extracted)
        x, y = np.linspace(0, 10, self._iw), np.linspace(0, 10, self._ih)
        xx, yy = np.meshgrid(x, y) 
        self._full_image = np.sin(xx) * np.cos(yy)
    
    def __empty_buffer(self):
        b = np.empty((self._ih, self._iw))
        b.fill(np.nan)
        return b
    
    def __setup_cds(self):
        if self._cds is None:
            self._cds = ColumnDataSource(data=dict(img=[self.__empty_buffer()]))
        return self._cds
    
    def __reset(self):
        if self._cds is not None:
            self._cds.data.update(img=[self.__empty_buffer()])
        self._row_index = 0
        self.resume()
    
    def __on_update_period_change(self, attr, old, new):
        """called when the user changes the refresh period using the dedicated slider"""
        self.update_callback_period(new)

    def __on_slice_size_change(self, attr, old, new):
        """called when the user changes the slice size using the dedicated slider"""
        self._inc = new
        
    def __suspend_resume(self): 
        """suspend/resume preriodic activity"""
        if self.suspended:
            self._suspend_resume_button.label = 'suspend'
            self.resume()
        else:
            self._suspend_resume_button.label = 'resume'
            self.pause()
       
    def open(self):  
        self._open_time = time.time()
        super(PatchedImage, self).open()
        
    def __close(self):  
        """tries to cleanup everything properly"""
        # celear cell ouputs
        clear_output()
        # cleanup the session
        self.close()
        
    def setup_document(self):
        """setup the session document"""
        # close button
        rb = Button(label='reset')
        rb.on_click(self.__reset)
        # close button
        cb = Button(label='close')
        cb.on_click(self.__close)
        # suspend/resume button
        self._suspend_resume_button = Button(label='suspend')
        self._suspend_resume_button.on_click(self.__suspend_resume)
        # a slider to control the update period
        upp = Slider(start=0.1, end=2, step=0.01, value=self.callback_period, title="Updt.period [s]",)
        upp.on_change("value", self.__on_update_period_change)
        # a slider to control the update period
        max_val = max(1, self._ih / 10)
        inc = Slider(start=1, end=max_val, step=1, value=self._inc, title="Slice size [rows]",)
        inc.on_change("value", self.__on_slice_size_change)
        # the figure and its content
        f = figure(plot_width=400, plot_height=350, x_range=(0, self._iw), y_range=(0, self._ih))
        ikwargs = dict()
        ikwargs['x'] = 0
        ikwargs['y'] = 0
        ikwargs['dw'] = self._iw
        ikwargs['dh'] = self._ih
        ikwargs['image'] = 'img'
        ikwargs['source'] = self.__setup_cds()
        ikwargs['color_mapper'] = LinearColorMapper(Viridis256)
        f.image(**ikwargs)
        # widgets are placed into a dedicated layout
        w = widgetbox(upp, inc, rb, self._suspend_resume_button, cb)
        # arrange all items into a layout then add it to the document
        self.document.add_root(layout([[w, f]]), setter=self.bokeh_session_id) 
        # start the periodic activity
        self.resume()
    
    def periodic_callback(self):
        """periodic activity"""
        if self._row_index >= self._ih:
            # done: full image received
            self.pause()
            return
        start = self._row_index
        start_wr6545 = start if start else None # workaround for bokeh bug #6545 
        end = self._row_index + self._inc
        s1, s2 = slice(start_wr6545, end), slice(None)
        index = [0, s1, s2]
        new_data = self._full_image[s1, s2].flatten()
        self._cds.patch({ 'img' : [(index, new_data)] })
        self._row_index = end

# ugly but mandatory: select the context in which we are running: NOTEBOOK or LAB
import os
os.environ["JUPYTER_CONTEXT"] = "LAB"

s1 = PatchedImage(1000, 900, 10, 2)
s1.open()


from bokeh.io import output_notebook 
from bokeh.resources import INLINE
output_notebook(resources=INLINE)

from common.session import BokehSession

import numpy as np

class Scanner(object):
    
    def __init__(self, imw=100, imh=100): 
        '''
        imw: image width
        imh: image height
        '''
        # scan progress: num of row added to the image at each iteration
        self._inc = max(1, int(imh / 20))
        # scan progress: row at which new data is injected at next iteration
        self._row_index = 0
        # scan image width and height
        self._iw, self._ih = int(imw), int(imh)
        # image buffer (from which scan data is extracted - for simulation purpose)
        x, y = np.linspace(0, 10, imw), np.linspace(0, 10, imh)
        xx, yy = np.meshgrid(x, y) 
        self._data_source = np.sin(xx) * np.cos(yy)
        # empty image (full frame)
        self._empty_image = self.empty_image((int(imh), int(imw)))
        # full image (full frame)
        self._full_image = self.__acquire(0, imh)
        
    def empty_image(self, shape=None):
        # produce an empty scanner image
        if not shape:
            empty_img = self._empty_image
        else:
            empty_img = np.empty(shape if shape else (self._ih, self._iw))
            empty_img.fill(np.nan)
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
    
    def reset(self):
        # reset 'current' image
        self._row_index = 0
        
    @property
    def x_range(self):
        return (-self._iw / 2., self._iw / 2.)

    @property
    def y_range(self):
        return (-self._ih / 2., self._ih / 2.)
    
    @property
    def image_width(self):
        return self._iw
    
    @property
    def image_height(self):
        return self._ih
    
    @property
    def num_pixels(self):
        return self._iw * self._ih
    
    @property
    def inc(self):
        return self._inc
 
    @inc.setter
    def inc(self, inc):
        self._inc = max(1, inc)
  

import time
import math as mt
import numpy as np
import uuid

from IPython.display import clear_output

from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from bokeh.models.glyphs import Rect
from bokeh.models.ranges import Range1d
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Slider, Button, TextInput
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import Plasma256, Viridis256
from bokeh.layouts import row, layout, widgetbox
from bokeh.models.tools import BoxZoomTool

import bokeh.events

from skimage.transform import rescale

class InteractionsManager(object):
    
    def __init__(self, bks):
        assert(isinstance(bks, BokehSession))
        self._session = bks
        self._callback = None
        self._range_change_notified = False
        
    def setup(self, figure, callback):
        self._callback = callback
        figure.x_range.on_change('start', self.__on_range_change)
        figure.x_range.on_change('end', self.__on_range_change)
        figure.y_range.on_change('start', self.__on_range_change)
        figure.y_range.on_change('end', self.__on_range_change)
        
    def __on_range_change(self, attr, old, new):
        if not self._range_change_notified and self._callback:
            self._range_change_notified = True
            try:
                # InteractionsManager.__on_range_change is called with 'document' locked
                # we consequently have to call the owner's handler asynchrounously so that 
                # it will be able to update things (i.e. to modify any plot data or property) 
                self._session.timeout_callback(self._callback, 0.25)
            except Exception as e:
                print(e)

    def range_change_handled(self):
        self._range_change_notified = False

class ScannerDisplay(BokehSession):
    
    def __init__(self, imw=100, imh=100, ist=1e5, upp=1.):
        '''
        imw: image width
        imh: image height
        ist: image size threshold above which we rescale the image
        upp: plot update period in seconds  
        '''
        BokehSession.__init__(self)
        # the underlying scanner
        self._scanner = Scanner(imw, imh)
        # last image acquire from scanner
        self._image = None
        # image size above which a rescale occurs
        self._image_size_threshold = ist
        # image plot update period in seconds
        self.callback_period = upp
        # bokeh plot
        self._rdr = None
        self._plt = None
        # bokeh column data source
        self._cds = None
        # an InteractionsManager
        self._itm = InteractionsManager(self)
        # suspend/resume button
        self._suspend_resume_button = None
        # image width & height +  x & y ranges 
        siw = self._scanner.image_width
        sih = self._scanner.image_height
        xrg = self._scanner.x_range
        yrg = self._scanner.y_range
        # the following will be used for linear interpolation (point coords -> pixel index) 
        self._xx = np.linspace(xrg[0], xrg[1], num=siw, dtype=float)
        self._xy = np.linspace(0, siw - 1, num=siw, dtype=int)
        self._yx = np.linspace(yrg[0], yrg[1], num=sih, dtype=float)
        self._yy = np.linspace(0, sih - 1, num=sih, dtype=int)
        
    def __setup_cds(self):
        self._image = self._scanner.empty_image((2,2))
        self._cds = ColumnDataSource(data=dict(img=[self._image]))
        return self._cds
    
    def __reset(self):
        self._image = self._scanner.empty_image((2,2))                                       
        self._cds.data.update(img=[self._image])
        self._scanner.reset()
        self.resume()
    
    @property
    def figure(self):
        return self._plt
    
    def __on_update_period_change(self, attr, old, new):
        """called when the user changes the refresh period using the dedicated slider"""
        self.update_callback_period(new)

    def __on_slice_size_change(self, attr, old, new):
        """called when the user changes the slice size using the dedicated slider"""
        self._scanner.inc = int(new)
     
    def __on_rescaling_factor_change(self, attr, old, new):
        """called when the user changes the rescaling factor using the dedicated slider"""
        self._rescaling_factor = new
        
    def __suspend_resume(self): 
        """suspend/resume preriodic activity"""
        if self.suspended:
            self._suspend_resume_button.label = 'suspend'
            self.resume()
        else:
            self._suspend_resume_button.label = 'resume'
            self.pause()
        
    def __close(self):  
        """tries to cleanup everything properly"""
        # celear cell ouputs
        clear_output()
        # cleanup the session
        self.close()
        
    def setup_document(self):
        """setup the session document"""
        # close button
        cb = Button(label='close')
        cb.on_click(self.__close)
        # reshape time text input
        self._rt_ti = TextInput(value="n/a", title="Rescaling time [s]:")
        self._rt_ti.disabled = True
        # before reshape shape text input 
        self._brs_ti = TextInput(value="n/a", title="Initial shape:")
        self._brs_ti.disabled = True
        #  rescale factor text input 
        self._rf_ti = TextInput(value="n/a", title="Rescale factor:")
        self._rf_ti.disabled = True
        # after reshape shape text input 
        self._ars_ti = TextInput(value="n/a", title="Rescale shape:")
        self._ars_ti.disabled = True
        # x-range text input 
        self._xrg_ti = TextInput(value="n/a", title="x-range:")
        self._xrg_ti.disabled = True
        # y-range text input 
        self._yrg_ti = TextInput(value="n/a", title="y-range:")
        self._yrg_ti.disabled = True
        # suspend/resume button
        self._suspend_resume_button = Button(label='suspend')
        self._suspend_resume_button.on_click(self.__suspend_resume)
        # a slider to control the update period
        upp = Slider(start=0.25, end=2, step=0.01, value=self.callback_period, title="Updt.period [s]")
        upp.on_change("value", self.__on_update_period_change)
        # a slider to control the scanner increment
        max_val = max(1, self._scanner.y_range[1] / 10)
        inc = Slider(start=1, end=max_val, step=1, value=self._scanner.inc, title="Slice size [rows]")
        inc.on_change("value", self.__on_slice_size_change)
        # tools
        tools="box_zoom,box_select,reset"
        # ranges
        xrg = self._scanner.x_range
        yrg = self._scanner.y_range
        # the figure and its content
        f = figure(plot_width=550, 
                   plot_height=500, 
                   x_range=Range1d(xrg[0], xrg[1]),
                   y_range=Range1d(yrg[0], yrg[1]),
                   tools=tools)
        ikwargs = dict()
        ikwargs['x'] = xrg[0]
        ikwargs['y'] = yrg[0]
        ikwargs['dw'] = abs(xrg[1] - xrg[0])
        ikwargs['dh'] = abs(yrg[1] - yrg[0])
        ikwargs['image'] = 'img'
        ikwargs['source'] = self.__setup_cds()
        ikwargs['color_mapper'] = LinearColorMapper(Viridis256)
        self._rdr = f.image(**ikwargs)
        # limit zoom out on both x & y axes
        f.x_range.min_interval = f.x_range.start
        f.x_range.max_interval = f.x_range.end
        f.y_range.min_interval = f.y_range.start
        f.y_range.max_interval = f.y_range.end
        # keep a ref. to the plot
        self._plt = f
        # setup InteractionsManager
        self._itm.setup(f, self.handle_range_change)
        # widgets are placed into a dedicated layout
        w = widgetbox(upp, 
                      inc, 
                      self._rt_ti,
                      self._rf_ti,
                      self._brs_ti, 
                      self._ars_ti, 
                      self._xrg_ti, 
                      self._yrg_ti, 
                      self._suspend_resume_button, 
                      cb)
        # arrange all items into a layout then add it to the document
        self.document.add_root(layout([[w, f]]), setter=self.bokeh_session_id) 
        # start periodic activity
        self.resume()
    
    def handle_range_change(self):
        #print("ScannerDisplay.handle_range_change <<")
        try:
            xrg = (self._plt.x_range.start, self._plt.x_range.end)
            yrg = (self._plt.y_range.start, self._plt.y_range.end)
            self.periodic_callback(update_image=False)
        finally:
            self._itm.range_change_handled()
        #print("ScannerDisplay.handle_range_change >>")

    def periodic_callback(self, update_image=True):
        try:
            # get image from scanner
            if update_image:
                self._image = self._scanner.image()
            self._xrg_ti.value = '{:.02f} : {:.02f}'.format(self._plt.x_range.start, self._plt.x_range.end)
            self._yrg_ti.value = '{:.02f} : {:.02f}'.format(self._plt.y_range.start, self._plt.y_range.end)
            # compute indexes (in original image) to (potentially new) x & y ranges
            xsi = int(mt.floor(np.interp(self._plt.x_range.start, self._xx, self._xy)))
            xei = int(mt.ceil(np.interp(self._plt.x_range.end, self._xx, self._xy))) + 1
            ysi = int(mt.floor(np.interp(self._plt.y_range.start, self._yx, self._yy)))
            yei = int(mt.ceil(np.interp(self._plt.y_range.end, self._yx, self._yy))) + 1
            #print('submatrix: x_start:{} - x_end:{} : y_start:{} - y_end:{}'.format(xsi, xei, ysi, yei))
            # extract part of the image corresponding to new x & y ranges
            image = self._image[ysi:yei, xsi:xei]
            #print('image.shape: {}'.format(image.shape))
            # rescale image in case its size is above threshold
            num_pixels = image.shape[0] * image.shape[1]
            if num_pixels > self._image_size_threshold:
                rescaling_factor = self.compute_rescaling_factor(num_pixels)
                image = self.rescale_image(image, rescaling_factor)
                #print('image.(re)shape: {}'.format(image.shape))
            else:
                self.reset_rescaling_info()
            # update bokeh plot
            self._cds.data.update(img=[image])
            # update plot scaling
            self._rdr.glyph.update(x=self._plt.x_range.start, 
                                   y=self._plt.y_range.start, 
                                   dw=abs(self._plt.x_range.end - self._plt.x_range.start), 
                                   dh=abs(self._plt.y_range.end - self._plt.y_range.start))
        except Exception as e:
            print(e)
         
    def compute_rescaling_factor(self, num_pixels):
        rescaling_factor = 1.0
        img_size = intial_img_size = num_pixels
        if img_size <= self._image_size_threshold:
            return rescaling_factor
        for inc in [0.1, 0.01, 0.001, 0.0001]:
            while img_size > self._image_size_threshold:
                rescaling_factor -= inc
                img_size = int(intial_img_size * rescaling_factor)
            rescaling_factor += inc
            img_size = int(intial_img_size * rescaling_factor)
        rescaling_factor = mt.sqrt(rescaling_factor)
        self._rf_ti.value = "{:.04f}".format(rescaling_factor)
        return rescaling_factor
    
    def rescale_image(self, in_img, rescaling_factor):
        t = time.time()
        out_img = rescale(in_img, rescaling_factor, mode='constant', cval=np.nan)
        self._rt_ti.value = '{:.04f}'.format(time.time() - t)
        self._brs_ti.value = '{}'.format(in_img.shape)
        self._ars_ti.value = '{}'.format(out_img.shape)
        return out_img
    
    def reset_rescaling_info(self):
        self._rt_ti.value = 'n/a'

# ugly but mandatory: select the context in which we are running: NOTEBOOK or LAB
import os
os.environ["JUPYTER_CONTEXT"] = "LAB"

# scanner image width
img_width = 4000
# scanner image height
img_height = 4000
# scanner image size above which we want to enable vaex binnning
img_size_threshold = 10000
# instanciate than open scanner image display 
d = ScannerDisplay(img_width, img_height, img_size_threshold)
d.open()

d.close()


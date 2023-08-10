from common.session import BokehSession

import numpy as np

from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from bokeh.models.glyphs import Rect
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.layouts import layout, widgetbox

class MySession(BokehSession):
    
    def __init__(self, uuid=None):
        BokehSession.__init__(self, uuid)
        self.callback_period = 1.
        self._np = 100
        self._widgets_layout = None
        columns = dict()
        columns['x'] = self._gen_x_scale()
        columns['y'] = self._gen_y_random_data()
        self._cds = ColumnDataSource(data=columns)

    def _gen_x_scale(self):
        """x data"""
        return np.linspace(1, self._np, num=self._np, endpoint=True)
    
    def _gen_y_random_data(self):
        """y data"""
        return np.random.rand(self._np)
    
    def __on_update_period_change(self, attr, old, new):
        """called when the user changes the refresh period using the dedicated slider"""
        self.update_callback_period(new)
        
    def __on_num_points_change(self, attr, old, new):
        """called when the user changes the number of points using the dedicated slider"""
        self._np = int(new)

    def setup_document(self):
        """setup the session model then return it"""
        # a slider to control the update period
        rrs = Slider(start=0.25, 
                     end=2, 
                     step=0.25, 
                     value=self.callback_period, 
                     title="Updt.period [s]",)
        rrs.on_change("value", self.__on_update_period_change)
        # a slider to control the number of points
        nps = Slider(start=0, 
                     end=1000, 
                     step=10, 
                     value=self._np, 
                     title="Num.points")
        nps.on_change("value", self.__on_num_points_change)
        # the figure and its content
        p = figure(plot_width=650, plot_height=200)
        p.toolbar_location = 'above'
        p.line(x='x', y='y', source=self._cds, color="navy", alpha=0.5)
        # widgets are placed into a dedicated layout
        self._widgets_layout = widgetbox(nps, rrs)
        # arrange all items into a layout then return it as the session model
        self.document.add_root(layout([[self._widgets_layout, p]]))
        # start periodic activity
        self.start()

    def periodic_callback(self):
        """periodic activity"""
        self._cds.data.update(x=self._gen_x_scale(), y=self._gen_y_random_data())

# ugly but mandatory: select the context in which we are running: NOTEBOOK or LAB
import os
os.environ["JUPYTER_CONTEXT"] = "LAB"

import logging
logging.getLogger('bokeh.server').setLevel(logging.DEBUG)
logging.getLogger('bokeh.server.tornado').setLevel(logging.ERROR)
logging.getLogger('fs.client.jupyter.session').setLevel(logging.DEBUG)

s1 = MySession('s1')
s1.open()

logging.getLogger('bokeh.server.tornado').setLevel(logging.DEBUG)

BokehSession.print_repository_status()

s1.close()

BokehSession.print_repository_status()


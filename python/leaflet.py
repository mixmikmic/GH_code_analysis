import warnings
# Warnings make for ugly notebooks, ignore them
warnings.filterwarnings('ignore')

from ipyleaflet import Map, DrawControl, ImageOverlay

import iris

fn = 'data/2017-11-21/work01/output/netcdf/discharge_dailyTot_output.nc'

cube = iris.load_cube(fn)

m = Map(center=center, zoom=zoom)

center = [52.3252978589571, 4.94580993652344]
zoom = 10

dc = DrawControl(polyline={}, polygon={}, edit=True, remove=False)
m.add_control(dc)

m

dc.last_draw['geometry']['coordinates']

from IPython.display import display
from ipywidgets import SelectionSlider

dts = [d.point for d in cube.coord('time').cells()]
dt_slider = SelectionSlider(options=dts)
display(dt_slider)

dt_slider.value

c = cube[1]

loc = cube.interpolate([('longitude', 4.799652), ('latitude', 52.331815)], iris.analysis.Nearest())

loc.data




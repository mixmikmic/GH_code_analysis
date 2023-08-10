import ipywidgets as widgets

slider = widgets.FloatSlider(min=0, max=10, step=0.5)
slider

text = widgets.FloatText(value=1)
text

text.value

text.value = 5
slider.value = 2

widgets.jslink((text, 'value'), (slider, 'value'))

slider.value = 3.0

import bqplot.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 50)
y = x**2

fig = plt.figure()
scatter = plt.scatter(x, y)
plt.show()

fig.animation_duration = 500
scatter.y = x**0.5

scatter.selected_style = {'stroke':'red', 'fill': 'orange'}
plt.brush_selector();

scatter.selected

scatter.selected = [1,2,10,40]

import ipyleaflet
ipyleaflet.Map(center = [53.2388975, 6.5317301], zoom = 15)

_.add_layer(ipyleaflet.ImageOverlay(url='https://jupyter.org//assets/nav_logo.svg', bounds=_.bounds, opacity=0.8))

import vaex
import numpy as np
np.warnings.filterwarnings('ignore')
dstaxi = vaex.open('/Users/maartenbreddels/datasets/nytaxi/nyc_taxi2015.hdf5') # mmapped, doesn't cost extra memory

dstaxi.plot_widget("pickup_longitude", "pickup_latitude", f="log", backend="ipyleaflet", shape=600)

dstaxi.plot_widget("dropoff_longitude", "dropoff_latitude", f="log", backend="ipyleaflet",
                   z="dropoff_hour", type="slice", z_shape=24, shape=400, z_relative=True,
                   limits=[None, None, (-0.5, 23.5)])

ds = vaex.datasets.helmi_de_zeeuw.fetch()

ds.plot_widget("x", "y", f="log", limits=[-20, 20])

ds.plot_widget("Lz", "E", f="log")

import ipyvolume as ipv
import numpy as np
np.warnings.filterwarnings('ignore')

ipv.example_ylm();

N = 1000
x, y, z = np.random.random((3, N))

fig = ipv.figure()
scatter = ipv.scatter(x, y, z, marker='box')
ipv.show()

scatter.x = scatter.x + 0.1

scatter.color = "green"
scatter.size = 5

scatter.color = np.random.random((N,3))

scatter.size = 2

ipv.figure()
ipv.style.use('dark')
quiver = ipv.quiver(*ipv.datasets.animated_stream.fetch().data[:,::,::4], size=5)
ipv.animation_control(quiver, interval=200)
ipv.show()

ipv.style.use('light')

quiver.size = np.random.random(quiver.x.shape) * 10

quiver.color = np.random.random(quiver.x.shape + (3,))

quiver.geo = "cat"

# stereo

quiver.geo = "arrow"

N = 1000*1000
x, y, z = np.random.random((3, N)).astype('f4')

ipv.figure()
s = ipv.scatter(x, y, z, size=0.2)
ipv.show()

s.size = 0.1

#ipv.screenshot(width=2048, height=2048)

plot3d = ds.plot_widget("x", "y", "z", vx="vx", vy="vy", vz="vz",
                        backend="ipyvolume", f="log1p", shape=100, smooth_pre=0.5)

plot3d.vcount_limits = [50, 100000]

plot3d.backend.quiver.color = "red"

import ipywidgets as widgets

widgets.ColorPicker()

widgets.jslink((plot3d.backend.quiver, 'color'), (_, 'value'))

ipv.save("kapteyn-lunch-talk-2018.html")

get_ipython().system('open kapteyn-lunch-talk-2018.html')

# webrtc demo if time permits

import vaex
#gaia = vaex.open("ws://gaia:9000/gaia-dr1")
gaia = vaex.open('/Users/maartenbreddels/datasets/gaia/gaia-dr1-minimal_f4.hdf5')
get_ipython().run_line_magic('matplotlib', 'inline')

f"{len(gaia):,}"

ra_dec_limits = [[0, 360], [-90, 90]]

gaia.set_active_fraction(0.01)

gaia.plot_widget("ra", "dec", limits=ra_dec_limits)

gaia.mean("phot_g_mean_mag", selection=True)

gaia.plot1d("phot_g_mean_mag", selection=False, n=True, limits=[10, 22])
gaia.plot1d("phot_g_mean_mag", selection=True, show=True, n=True, limits=[10, 22])










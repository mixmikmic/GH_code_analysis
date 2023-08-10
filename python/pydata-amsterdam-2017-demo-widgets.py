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
scatter.y = x**1.5

scatter.selected_style = {'stroke':'red', 'fill': 'orange'}
plt.brush_selector()

scatter.selected

scatter.selected = [1,2,10,40]

import ipyleaflet
ipyleaflet.Map(center = [52.3655343,4.8963377], zoom = 14)

_.add_layer(ipyleaflet.ImageOverlay(url='http://localhost:8888/static/base/images/logo.png', bounds=_.bounds, opacity=0.8))

import vaex
dstaxi = vaex.open("/Users/maartenbreddels/vaex/data/nytaxi/nyc_taxi2015.hdf5") # mmapped, doesn't cost extra memory

# dstaxi.plot_widget(dstaxi.col.pickup_longitude, dstaxi.col.pickup_latitude, f="log1p")

dstaxi.plot_widget("pickup_longitude", "pickup_latitude", f="log", backend="ipyleaflet", shape=400)

dstaxi.count("dropoff_hour")

dstaxi.plot_widget("dropoff_longitude", "dropoff_latitude", f="log", backend="ipyleaflet",
                   z="dropoff_hour", type="slice", z_shape=24, shape=400, z_relative=True,
                   limits=[None, None, (-0.5, 23.5)])

ds = vaex.datasets.helmi_de_zeeuw.fetch()

ds.plot_widget("x", "y", f="log", limits=[-20, 20])

ds.plot_widget("Lz", "E", f="log")

import ipyvolume
import ipyvolume.pylab as plt3d
import numpy as np

ipyvolume.example_ylm()

N = 1000
x, y, z = np.random.random((3, N))

fig = plt3d.figure()
scatter = plt3d.scatter(x, y, z, marker='box')
plt3d.show()

scatter.color = "green"
scatter.size = 10

scatter.color = np.random.random((N,3))

scatter.size = 2

plt3d.figure()
plt3d.style.use('dark')
quiver = plt3d.quiver(*ipyvolume.datasets.animated_stream.fetch().data[:,::,::4], size=5)
plt3d.animate_glyphs(quiver, interval=200)
plt3d.show()

quiver.geo = "cat"

quiver.geo = "arrow"

# ds.plot_widget("x", "y", "z", backend="ipyvolume", f="log1p", shape=100, smooth_pre=0.5)

plot3d = ds.plot_widget("x", "y", "z", vx="vx", vy="vy", vz="vz",
                        backend="ipyvolume", f="log1p", shape=100, smooth_pre=0.5)

plot3d.vcount_limits = [70, 100000]

plot3d.backend.quiver.color = "red"

widgets.ColorPicker()

widgets.jslink((plot3d.backend.quiver, 'color'), (_, 'value'))

plt3d.save("pydata.html")

get_ipython().system('open pydata.html')

import vaex
gaia = vaex.open("ws://gaia:9000/gaia-dr1")

f"{len(gaia):,}" # (python3.6 f-string)

ra_dec_limits = [[0, 360], [-90, 90]]

gaia.plot_widget("ra", "dec", limits=ra_dec_limits)

gaia.plot1d("phot_g_mean_mag", selection=False, n=True)
gaia.plot1d("phot_g_mean_mag", selection=True, show=True, n=True)


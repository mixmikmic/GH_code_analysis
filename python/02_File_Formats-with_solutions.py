#%matplotlib inline
from __future__ import print_function
import matplotlib.pylab as plt
plt.switch_backend("nbagg")
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

import obspy

# ObsPy automatically detects the file format.
st = obspy.read("data/example.mseed")
print(st)

# Fileformat specific information is stored here.
print(st[0].stats.mseed)

st.plot()

# This is a quick interlude to teach you basics about how to work
# with Stream/Trace objects.

# Most operations work in-place, e.g. they modify the existing
# objects. We'll create a copy here.
st2 = st.copy()

# To use only part of a Stream, use the select() function.
print(st2.select(component="Z"))

# Stream objects behave like a list of Trace objects.
tr = st2[0]

tr.plot()

# Some basic processing. Please note that these modify the
# existing object.
tr.detrend("linear")
tr.taper(type="hann", max_percentage=0.05)
tr.filter("lowpass", freq=0.5)

tr.plot()

# You can write it again by simply specifing the format.
st.write("temp.mseed", format="mseed")

st = obspy.read("data/example.sac")
print(st)
st[0].stats.sac.__dict__

st.plot()

# You can once again write it with the write() method.
st.write("temp.sac", format="sac")

get_ipython().system('head data/all_stations.xml')

import obspy

# Use the read_inventory function to open them.
inv = obspy.read_inventory("data/all_stations.xml")
print(inv)

# ObsPy is also able to plot a map of them.
inv.plot(projection="local");

# As well as a plot the instrument response.
inv.select(network="IV", station="SALO", channel="BH?").plot_response(0.001);

# Coordinates of single channels can also be extraced. This function
# also takes a datetime arguments to extract information at different
# points in time.
inv.get_coordinates("IV.SALO..BHZ")

# And it can naturally be written again, also in modified state.
inv.select(channel="BHZ").write("temp.xml", format="stationxml")

get_ipython().system('cat data/GCMT_2014_04_01__Mw_8_1')

# Read QuakeML files with the read_events() function.
cat = obspy.read_events("data/GCMT_2014_04_01__Mw_8_1.xml")
print(cat)

print(cat[0])

cat.plot(projection="ortho");

# Once again they can be written with the write() function.
cat.write("temp_quake.xml", format="quakeml")

import obspy

cat = obspy.read_events("data/2014.ndk")

print(cat)

cat.plot();

cat.filter("depth > 100000", "magnitude > 7")


get_ipython().magic('matplotlib inline')
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

from obspy import readEvents

catalog = readEvents("./data/event_tohoku_with_big_aftershocks.xml")
print(catalog)

print(type(catalog))
print(type(catalog[0]))

event = catalog[0]
print(event)

print(type(event.origins))
print(type(event.origins[0]))
print(event.origins[0])

print(type(event.magnitudes))
print(type(event.magnitudes[0]))
print(event.magnitudes[0])

# try event.<Tab> to get an idea what "children" elements event has

largest_magnitude_events = catalog.filter("magnitude >= 7.8")
print(largest_magnitude_events)

catalog.plot(projection="local");

largest_magnitude_events.write("/tmp/large_events.xml", format="QUAKEML")
get_ipython().system('ls -l /tmp/large_events.xml')

from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude
from obspy.core.util.geodetics import FlinnEngdahl

cat = Catalog()
cat.description = "Just a fictitious toy example catalog built from scratch"

e = Event()
e.event_type = "not existing"

o = Origin()
o.time = UTCDateTime(2014, 2, 23, 18, 0, 0)
o.latitude = 47.6
o.longitude = 12.0
o.depth = 10000
o.depth_type = "operator assigned"
o.evaluation_mode = "manual"
o.evaluation_status = "preliminary"
o.region = FlinnEngdahl().get_region(o.longitude, o.latitude)

m = Magnitude()
m.mag = 7.2
m.magnitude_type = "Mw"

m2 = Magnitude()
m2.mag = 7.4
m2.magnitude_type = "Ms"

# also included could be: custom picks, amplitude measurements, station magnitudes,
# focal mechanisms, moment tensors, ...

# make associations, put everything together
cat.append(e)
e.origins = [o]
e.magnitudes = [m, m2]
m.origin_id = o.resource_id
m2.origin_id = o.resource_id

print(cat)
cat.write("/tmp/my_custom_events.xml", format="QUAKEML")
get_ipython().system('cat /tmp/my_custom_events.xml')


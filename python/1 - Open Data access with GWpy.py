import sys
get_ipython().system('{sys.executable} -m pip install "lalsuite" "gwpy>=0.8.0[hdf5]"')

import gwpy
print(gwpy.__path__)
print(gwpy.__version__)

from gwopensci.datasets import event_gps
gps = event_gps('GW150914')
print(gps)

segment = (int(gps)-5, int(gps)+5)

from gwpy.timeseries import TimeSeries
data = TimeSeries.fetch_open_data('H1', *segment, verbose=True)
print(data)

gps = event_gps('GW170817')
segment = (int(gps) - 5, int(gps) + 5)
hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True, tag='CLN')
print(data)

get_ipython().run_line_magic('matplotlib', 'inline')

plot = hdata.plot()

get_ipython().system('curl -O https://losc.ligo.org//s/events/GW170817/H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5')
hdata2 = TimeSeries.read('H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5', start=segment[0], end=segment[1], format='hdf5.losc')
plot2 = hdata2.plot()
plot2.show()


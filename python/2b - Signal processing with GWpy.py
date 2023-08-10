get_ipython().run_line_magic('matplotlib', 'inline')

from gwopensci.datasets import event_gps
from gwpy.timeseries import TimeSeries

gps = event_gps('GW150914')
segment = (int(gps) - 5, int(gps) + 5)
hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True)
plot = hdata.plot()
plot.show()

hasd = hdata.asd(fftlength=2)
plot2 = hasd.plot(color='gwpy:ligo-hanford')
ax = plot2.gca()
ax.set_xlim(10, 1400)
ax.set_ylim(3e-24, 5e-21)
plot2.show()

ax = plot.gca()
ax.set_epoch(gps)  # set T=0 on the X-axis
ax.set_xlim(gps, gps+.6)
plot

hbp = hdata.bandpass(50, 250, filtfilt=True)

plot = hbp.plot()

hbp = hbp.crop(int(gps) - 4, int(gps) + 4)
plot = hbp.plot()

ax = plot.gca()
ax.set_epoch(int(gps))
ax.set_xlim(gps-.4, gps+.2)
plot

from gwpy.signal import filter_design
notches = [filter_design.notch(f, hdata.sample_rate) for f in (60, 120, 180)]
powernotch = filter_design.concatenate_zpks(*notches)

hclean = hbp.filter(powernotch, filtfilt=True)
plot = hclean.plot()
ax = plot.gca()
ax.set_epoch(int(gps))
ax.set_xlim(gps-.4, gps+.2)

plot = hclean.plot(figsize=(12, 4), color='gwpy:ligo-hanford')
ax = plot.gca()
ax.set_xscale('seconds', epoch=int(gps))
ax.set_xlim(int(gps), int(gps)+.6)
ax.set_xlabel('Time (seconds) from 2015-09-14 09:50:45')
ax.set_ylim(-1e-21, 1e-21)
ax.set_ylabel('Strain amplitude')
ax.set_title('LIGO-Hanford strain data around GW150914')




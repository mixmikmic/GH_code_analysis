get_ipython().run_line_magic('matplotlib', 'inline')

from gwopensci.datasets import event_gps
from gwpy.timeseries import TimeSeries

gps = event_gps('GW150914')
igps = int(gps)
segment = (igps - 5, igps + 5)
hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True)
plot = hdata.plot()

fft = hdata.fft()
print(fft)

plot = fft.abs().plot()
plot.show()

from scipy.signal import get_window
window = get_window('hann', hdata.size)
hwin = hdata * window

fftamp = hwin.fft().abs()
plot = fftamp.plot()
plot.show()

asd = hdata.asd(fftlength=4)
plot = asd.plot()
plot.show()

ax = plot.gca()
ax.set_xlim(10, 1400)
ax.set_ylim(2e-24, 1e-20)
plot

hdata2 = TimeSeries.fetch_open_data('H1', igps-512, igps+512, cache=True)
hasd2 = hdata2.asd(fftlength=4)
plot = hasd2.plot()
ax = plot.gca()
ax.set_xlim(10, 1400)
ax.set_ylim(5e-24, 1e-20)
plot.show()

# get Livingston data
ldata2 = TimeSeries.fetch_open_data('L1', igps-512, igps+512, cache=True)
lasd2 = ldata2.asd(fftlength=4)

# and plot using standard colours
ax.plot(lasd2, label='LIGO-Livingston', color='gwpy:ligo-livingston')

# update the Hanford line to use standard colour, and have a label
hline = ax.lines[0]
hline.set_color('gwpy:ligo-hanford')  # change colour of Hanford data
hline.set_label('LIGO-Hanford')

ax.set_ylabel(r'Strain noise [$1/\sqrt{\mathrm{Hz}}$]')
ax.legend()
plot

specgram = hdata2.spectrogram2(fftlength=4, overlap=2, window='hann') ** (1/2.)
plot = specgram.plot()
plot.show()

plot = specgram.plot(vmin=1e-24, vmax=1e-20, norm='log')
ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(10, 1400)
plot.add_colorbar(label=r'Strain noise [$1/\sqrt{\mathrm{Hz}}$]')


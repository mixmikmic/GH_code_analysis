get_ipython().run_line_magic('matplotlib', 'inline')

# format our times
injstart, injend = 1126310499.75, 1126310500.25
datastart = int(injstart) - 4
dataend = int(injend) + 4

# fetch the data
from gwpy.timeseries import TimeSeries
hdata = TimeSeries.fetch_open_data('H1', datastart, dataend, cache=True)

# generate the Q-transform
qtrans = hdata.q_transform(outseg=(injstart-1, injend+1))

# plot
plot = qtrans.plot(vmin=0, vmax=20, yscale='log')
plot.show()

idx = qtrans.argmax()

# get indices for times and frequencies
from numpy import unravel_index
timeid, freqid = unravel_index(idx, qtrans.shape)

# extract values (using .value to extract as a float)
peakgps = qtrans.times[timeid].value
peakfrequency = qtrans.frequencies[freqid].value
print(peakgps, peakfrequency)

ax = plot.gca()
ax.scatter([peakgps], [peakfrequency], marker='*', color='red', s=100)
plot


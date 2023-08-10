get_ipython().magic('matplotlib inline')
from rf import read_rf, rfstats

stream = read_rf()
print stream
print '\nStats:\n', stream[0].stats
stream[:3].plot()

rfstats(stream)
stream.filter('bandpass', freqmin=0.4, freqmax=1)
stream.trim2(5, 95, 'starttime')
print stream
stream[:3].plot(type='relative', reftime=stream[0].stats.onset)

stream.rf()
stream.moveout()
stream.trim2(-5, 22, 'onset')
print stream
stream.select(component='L').plot_rf()    
stream.select(component='Q').plot_rf()


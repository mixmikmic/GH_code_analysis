import numpy as np
import pandas as pd
import pylab as pl
pl.style.use('ggplot')

df = pd.read_csv('../data/scan_data_gain200_thumb.csv', header=None, index_col=0)
df.plot(legend=False)
pl.xlabel('time [ms]')
pl.show()

df.loc[15000:25000].rolling(window=10).median().mean(axis=1).plot(
        title='avg signal, touch')
pl.xlabel('time [ms]')
pl.show()

pd.tools.plotting.autocorrelation_plot(df.loc[15000:25000].mean(axis=1))
pl.xlim((0,160))
pl.xticks([35,70,105,140])
pl.grid(which='major',axis='x')
pl.title('touch signal, autocorrelation')
pl.show()

avg_sample_rate = np.mean(np.diff(df.index.values))
print('Avg sampling period: %.2f ms, (%d ms – %d ms)' % (
        avg_sample_rate,
        np.min(np.diff(df.index.values)),
        np.max(np.diff(df.index.values))))
print('Period: 35 samples * 21.35 ms = %.2f sec --> %.1f beats / min' % 
      (35*0.001*avg_sample_rate, 60/(35*0.001*avg_sample_rate)))

df.loc[:10000].rolling(window=10).median().mean(axis=1).plot(
        title='avg signal, no touch')
pl.xlabel('time [ms]')
pl.show()

pd.tools.plotting.autocorrelation_plot(df.loc[:10000].mean(axis=1))
pl.xlim((0,160))
pl.xticks([35, 70, 105, 140])
pl.grid(which='major',axis='x')
pl.title('no touch signal, autocorrelation')
pl.show()


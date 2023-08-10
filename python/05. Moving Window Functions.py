get_ipython().magic('matplotlib inline')
import matplotlib.pylab

get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (10, 6)

import numpy as np
import pandas as pd

ts = pd.Series(np.random.randn(20), pd.date_range('7/1/16', freq = 'D', periods = 20))
ts_lagged = ts.shift(-5)

plt.plot(ts, color = 'blue')
plt.plot(ts_lagged, color = 'red')

# %load snippets/shift_future.py

# Window functions are like aggregation functions
# You can use them in conjunction with .resample()

df = pd.DataFrame(np.random.randn(600, 3), index = pd.date_range('5/1/2016', freq = 'D', periods = 600), columns = ['A', 'B', 'C'])

r = df.rolling(window = 20)
r

df['A'].plot(color = 'gray')
r.mean()['A'].plot(color = 'red')

r.quantile(0.5).plot()

# %load snippets/custom_rolling.py
df.rolling(window = 10, center = False).apply(lambda x: x[1]/x[2])[10:30]

# %load snippets/resample_rolling.py
ts_long = pd.Series(np.random.randn(200),pd.date_range('7/1/16', freq = 'D', periods=200))
ts_long.resample('M').mean().rolling(window = 3).mean().plot()

df.expanding(min_periods = 1).mean()[1:5]

df.expanding(min_periods = 1).mean().plot()

df.expanding(min_periods = 1).mean()[1:5]

df.expanding(min_periods = 1)

# import pandas as pd
# import numpy as np
# # %load snippets/window_funcs_try.py
# #1
# ts = pd.Series(np.random.randn(1000), index = pd.date_range(start = '1/1/16', periods = 1000, freq = 'D'))
# ts.ewm(zspan = 60, frezq = 'D', min_periods = 0, adjust = True).mean().plot()
# ts.rolling(window = 60z).mean().plot()

# #2 
# # To get a more reliable statistic if it makes logical sense





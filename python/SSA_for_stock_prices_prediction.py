from ssa_core import ssa, ssa_predict, ssaview, inv_ssa, ssa_cutoff_order
from mpl_utils import set_mpl_theme

import matplotlib.pylab as plt
import quandl
import pandas as pd
import datetime
from datetime import timedelta
from dateutil import parser


get_ipython().magic('matplotlib inline')

# customize mpl a bit
set_mpl_theme('light')

# some handy functions
def fig(w=16, h=5, dpi=96, facecolor=None, edgecolor=None):
    return plt.figure(figsize=(w, h), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)

def mape(f, t):
    return 100*((f - t)/t).abs().sum()/len(t)

def mae(f, t):
    return 100*((f - t)).abs().sum()/len(t)

instrument = 'MSFT'
data = quandl.get('WIKI/%s' % instrument, start_date='2012-01-01', end_date='2017-02-01')
closes = data['Adj. Close'].rename('close')

test_date = '2017-01-01'

train_d = closes[:test_date]
test_d = closes[test_date:]

fig(16, 3); plt.plot(train_d, label='Train'); plt.plot(test_d, 'r', label='Test')
plt.title('%s adjusted daily close prices' % instrument)
plt.legend()

fig()
ssaview(train_d.values, 120, [0,1,2,3])

pc, _, v = ssa(train_d.values, 120)
reconstructed = inv_ssa(pc, v, [0,1,2,3])
noise = train_d.values - reconstructed
plt.hist(noise, 50);

MAX_LAG_NUMBER = 120 # 4*30 = 1 quarter max
n_co = ssa_cutoff_order(train_d.values, dim=MAX_LAG_NUMBER, show_plot=True)

days_to_predict = 15
forecast = ssa_predict(train_d.values, n_co, list(range(8)), days_to_predict, 1e-5)

fig(16, 5)

prev_ser = closes[datetime.date.isoformat(parser.parse(test_date) - timedelta(120)):test_date]
plt.plot(prev_ser, label='Train Data')

test_d = closes[test_date:]
f_ser = pd.DataFrame(data=forecast, index=test_d.index[:days_to_predict], columns=['close'])
orig = pd.DataFrame(test_d[:days_to_predict])

plt.plot(orig, label='Test Data')
plt.plot(f_ser, 'r-', marker='.', label='Forecast')
plt.legend()
plt.title('Forecasting %s for %d days, MAPE = %.2f%%' % (instrument, days_to_predict, mape(f_ser, orig)));








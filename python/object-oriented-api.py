import matplotlib.pyplot as plt
import pandas as pd

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

aapl = pd.read_csv('data/aapl.csv', parse_dates=['Date'])
aapl['ticker'] = 'AAPL'
aaba = pd.read_csv('data/aaba.csv', parse_dates=['Date'])
aaba['ticker'] = 'AABA'
msft = pd.read_csv('data/msft.csv', parse_dates=['Date'])
msft['ticker'] = 'MSFT'
goog = pd.read_csv('data/goog.csv', parse_dates=['Date'])
goog['ticker'] = 'GOOG'



data = pd.concat([aapl, aaba, msft, goog])
data.set_index('Date', inplace=True)
data.head(20)

plt.figure()
plt.plot(aapl['Close'])
plt.title('AAPL')

plt.figure()
plt.plot(aaba['Close'])
plt.title('AABA')

plt.figure()
plt.plot(msft['Close'])
plt.title('MSFT')

plt.figure()
plt.plot(goog['Close'])
plt.title('GOOG')

fig = plt.figure(figsize=(10,6))

ax1 = fig.add_subplot(2,2,1)
ax1.plot(aapl['Close'])
ax1.set_title('AAPL')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(aaba['Close'])
ax2.set_title('AABA')

ax3 = fig.add_subplot(2,2,3)
ax3.plot(msft['Close'])
ax3.set_title('MSFT')

ax3 = fig.add_subplot(2,2,4)
ax3.plot(goog['Close'])
ax3.set_title('GOOG')

plt.tight_layout()
plt.show()

symbols = ['AAPL', 'AABA', 'MSFT', 'GOOG']

def plot_symbol(data, symb, ax):
    cond = data['ticker'] == symb
    ax.plot(data[cond]['Close'])
    return ax

fig = plt.figure(figsize=(10, 6))
from matplotlib.gridspec import GridSpec

gs = GridSpec(2, 2)
for i, symbol in enumerate(symbols):
    ax = fig.add_subplot(gs[i])
    ax = plot_symbol(data, symbol, ax)
    ax.set_title(symbol)
    
plt.tight_layout()
plt.show()




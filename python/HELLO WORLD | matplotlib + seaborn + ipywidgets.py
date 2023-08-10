get_ipython().magic('pylab inline')

t = arange(0.0, 1.0, 0.01)

y1 = sin(2*pi*t)
y2 = sin(2*2*pi*t)

import pandas as pd

df = pd.DataFrame({'t': t, 'y1': y1, 'y2': y2})
df.head(10)

fig = figure(1, figsize = (10,10))

ax1 = fig.add_subplot(211)
ax1.plot(t, y1)
ax1.grid(True)
ax1.set_ylim((-2, 2))
ax1.set_ylabel('Gentle Lull')
ax1.set_title('I can plot waves')

for label in ax1.get_xticklabels():
    label.set_color('r')


ax2 = fig.add_subplot(212)
ax2.plot(t, y2,)
ax2.grid(True)
ax2.set_ylim((-2, 2))
ax2.set_ylabel('Getting choppier')
l = ax2.set_xlabel('Hi PyLadies')
l.set_color('g')
l.set_fontsize('large')

show()

import seaborn as sns

sns.set(color_codes=True)
sns.distplot(y1)
sns.distplot(y2)

from ipywidgets import widgets
from IPython.html.widgets import *

t = arange(0.0, 1.0, 0.01)

def pltsin(f):
    plt.plot(t, sin(2*pi*t*f))
    
interact(pltsin, f=(1,10,0.1))

def pltsin(f):
    sns.distplot(sin(2*pi*t*f))
    
interact(pltsin, f=(1,10,0.1))


import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt

print('\nPython version: ', sys.version)
print('Pandas version: ', pd.__version__, '\n')

url = 'http://www.ggdc.net/maddison/maddison-project/data/mpd_2013-01.xlsx'
mpd = pd.read_excel(url, skiprows=2, index_col=0, na_values=[' '])
# strip trailing blanks in country names
# use comprehension instead? string methods?
mpd.columns = map(str.rstrip, mpd.columns)

print('Dataframe dimensions:', mpd.shape)

countries = ['England/GB/UK', 'USA', 'Japan', 'China', 'India', 'Argentina']
mpd = mpd[countries]
mpd = mpd.rename(columns={'England/GB/UK': 'UK'})
mpd = np.log(mpd)/np.log(2)

print('Dataframe dimensions:', mpd.shape)

subset = mpd.dropna().copy()

fig, ax = plt.subplots(figsize=(12,10))
subset.plot(lw=2, ax=ax)
ax.set_title('GDP per person', fontsize=18, loc='center')
ax.set_ylabel('GDP Per Capita (1990 USD, log2 scale)')
ax.legend(loc='upper left', fontsize=10, handlelength=2, labelspacing=0.15)

ax.set_xlim(1870,2010) # This sets the y-limits

ax.spines["right"].set_visible(False) # This removes the ``spines'', just the right and top
ax.spines["top"].set_visible(False) # ones...

plt.show()




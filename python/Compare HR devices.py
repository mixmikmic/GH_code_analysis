import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xml.etree.ElementTree as ET
import datetime

get_ipython().magic('matplotlib inline')

mpl.rcParams['figure.figsize'] = (16, 8)
mpl.rcParams['font.size'] = 14

def trackpoint_to_dict(tp):
    """Save interesting info from Trackpoint element as dict"""
    return {
        'time': tp.find('g:Time', ns).text,
        'hr': int(tp.find('g:HeartRateBpm/g:Value', ns).text)
    }

def tcx_to_dataframe(tcx_filename):
    """Open TCX file, pull info for each trackpoint in time.
    Returns info as pandas Dataframe
    """
    tree = ET.parse(tcx_filename)
    root = tree.getroot()
    ns = {"g": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}  # garmin namespace, bleurgh
    trackpoints = root.findall(".//g:Trackpoint", ns)
    dicts = [trackpoint_to_dict(tp) for tp in trackpoints] 
    df = pd.DataFrame(dicts)
    df.index = pd.to_datetime(df['time'])
    df = df.drop(['time'], axis=1)
    df = df.resample('1S').mean()  # resample to align to exact seconds for easier comparison
    return df

df_strap = tcx_to_dataframe("strap.tcx")

df_watch = tcx_to_dataframe("watch.tcx")

# Convert to time from start of activity
start_time = min(df_strap.index[0], df_watch.index[0])
df_watch['rel_time'] = df_watch.index - start_time
df_strap['rel_time'] = df_strap.index - start_time

df_strap.dtypes

plt.plot(df_strap.rel_time, df_strap.hr, label='chest strap');
ax = plt.gca()
ax.plot(df_watch.rel_time, df_watch.hr, label='optical');
plt.legend(loc='best');
plt.ylabel('HR [bpm]');

ml = ax.xaxis.get_major_locator()
ml.set_params(nbins=20, min_n_ticks=10)
ax.xaxis.set_major_locator(ml)

# This doesn't work for timedelta, no idea why
# locator = mdates.AutoDateLocator()
# ax.xaxis.set_major_locator(locator)

def timeTicks(x, pos):                                                                                                                                                                                                                                                         
    d = datetime.timedelta(seconds=x/1E9)                                                                                                                                                                                                                                          
    return str(d)                                                                                                                                                                                                                                                              
formatter = mpl.ticker.FuncFormatter(timeTicks)                                                                                                                                                                                                                         
ax.xaxis.set_major_formatter(formatter)  

plt.grid()

df_watch.head()

df_strap.head()

diff = (df_watch - df_strap).dropna().drop(['rel_time'], axis=1)

# Get standard quantiles (95%, 68%, 50%)
quantiles = diff.hr.quantile([0.025, 0.16, 0.25, 0.5, 0.75, 0.84, 0.975])
print(quantiles)

mask = df_strap.hr < 140
diff_filtered = (df_watch - df_strap).drop(['rel_time'], axis=1).dropna()

mask.shape

diff_filtered.shape

diff_filtered = diff_filtered[mask]
diff_filtered = diff_filtered.dropna()
quantiles_filtered = diff_filtered.hr.quantile([0.025, 0.16, 0.25, 0.5, 0.75, 0.84, 0.975])
print(quantiles_filtered)

plt.plot(diff_filtered)

# with quantiles
plt.plot(diff);
plt.ylabel('Optical HR - strap HR [bpm]');
plt.axhline(quantiles[0.5], linestyle='solid', color='black', label='0.5 (%d)' % quantiles[0.5])
plt.axhline(quantiles[0.25], linestyle='dashed', color='black', label='0.25 (%d)' % quantiles[0.25])
plt.axhline(quantiles[0.75], linestyle='dashed', color='black', label='0.75 (%d)' % quantiles[0.75])
plt.axhline(quantiles[0.16], linestyle='dashed', color='grey', label='0.16 (%d)' % quantiles[0.16])
plt.axhline(quantiles[0.84], linestyle='dashed', color='grey', label='0.84 (%d)' % quantiles[0.84])
plt.axhline(quantiles[0.025], linestyle='dotted', color='grey', label='0.025 (%d)' % quantiles[0.025])
plt.axhline(quantiles[0.975], linestyle='dotted', color='grey', label='0.975 (%d)' % quantiles[0.975])
plt.legend(title="Quantiles");
# plt.grid();

# with fixed reasonable horizontal lines (±5, ±10 bpm)
plt.plot(diff);
plt.ylabel('Optical HR - strap HR [bpm]');
plt.axhline(0, linestyle='dashed', color='black')
plt.axhline(-5, linestyle='dashed', color='grey')
plt.axhline(5, linestyle='dashed', color='grey')
plt.axhline(-10, linestyle='dotted', color='grey')
plt.axhline(10, linestyle='dotted', color='grey')

plt.hist(diff.hr, bins=np.arange(-30, 30, 1));
plt.xlabel('Optical HR - strap HR [bpm]');

rel_diff = ((df_watch.hr - df_strap.hr)/df_strap.hr).dropna()
rel_diff.head()

# Get standard quantiles (95%, 68%, 50%)
quantiles = rel_diff.quantile([0.025, 0.16, 0.25, 0.5, 0.75, 0.84, 0.975])
print(quantiles)

plt.hist(rel_diff, bins=50);
plt.xlabel('Fractional HR difference (Optical - strap / strap)');

plt.hist(df_strap.hr, bins=50);




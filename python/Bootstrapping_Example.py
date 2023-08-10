get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from collections import defaultdict, Counter, OrderedDict

pd.set_option('max_columns', 50)
mpl.rcParams['lines.linewidth'] = 4

ja = '#D12325' # cubs red ... cubs blue = '#0E3386'
zg = '#005596' # dodger blue
kc = '#005596'

arrietaPitches = pd.read_csv('https://raw.githubusercontent.com/gjreda/cy-young-NL-2015/master/data/pitchfx/arrieta.csv', parse_dates=['game_date'])
greinkePitches = pd.read_csv('https://raw.githubusercontent.com/gjreda/cy-young-NL-2015/master/data/pitchfx/greinke.csv', parse_dates=['game_date'])
kershawPitches = pd.read_csv('https://raw.githubusercontent.com/gjreda/cy-young-NL-2015/master/data/pitchfx/kershaw.csv', parse_dates=['game_date'])

arrietaPitches.tail()

# if it's not a ball, it's a strike
ball_vals = ['Ball', 'Ball In Dirt', 'Intent Ball', 'Hit By Pitch']
swing_and_miss = ['Swinging Strike', 'Swinging Strike (Blocked)', 'Missed Bunt']
hit_vals = ['Single', 'Double', 'Triple', 'Home Run']

arrietaPitches.loc[arrietaPitches.pitch_result.isin(ball_vals), 'is_strike'] = 0
arrietaPitches.loc[arrietaPitches.is_strike != 0, 'is_strike'] = 1
arrietaPitches.loc[arrietaPitches.pitch_result.isin(swing_and_miss), 'swing_and_miss'] = 1
arrietaPitches.loc[arrietaPitches.atbat_result.isin(hit_vals), 'is_hit'] = 1
arrietaPitches.loc[arrietaPitches.atbat_result == 'Single', 'total_bases'] = 1
arrietaPitches.loc[arrietaPitches.atbat_result == 'Double', 'total_bases'] = 2
arrietaPitches.loc[arrietaPitches.atbat_result == 'Triple', 'total_bases'] = 3
arrietaPitches.loc[arrietaPitches.atbat_result == 'Home Run', 'total_bases'] = 4

greinkePitches.loc[greinkePitches.pitch_result.isin(ball_vals), 'is_strike'] = 0
greinkePitches.loc[greinkePitches.is_strike != 0, 'is_strike'] = 1

kershawPitches.loc[kershawPitches.pitch_result.isin(ball_vals), 'is_strike'] = 0
kershawPitches.loc[kershawPitches.is_strike != 0, 'is_strike'] = 1

arrietaPitches.head()

fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(14,5))

arrietaPitches.batted_ball_velocity.hist(ax=axes[0], bins=20, label='Arrieta', alpha=.5, color=ja)
axes[0].set_title('Arrieta BB Velocity')
greinkePitches.batted_ball_velocity.hist(ax=axes[1], bins=20, label='Greinke', alpha=.5, color=zg)
axes[1].set_title('Greinke BB Velocity')
kershawPitches.batted_ball_velocity.hist(ax=axes[2], bins=20, label='Kershaw', alpha=.5, color='grey')
axes[2].set_title('Kershaw BB Velocity');

# set random seed for consistency
np.random.seed(49)

arrietaBBs = arrietaPitches[arrietaPitches.batted_ball_velocity > 0].batted_ball_velocity
greinkeBBs = greinkePitches[greinkePitches.batted_ball_velocity > 0].batted_ball_velocity
kershawBBs = kershawPitches[kershawPitches.batted_ball_velocity > 0].batted_ball_velocity
arrietaSamples = []
greinkeSamples = []
kershawSamples = []

for i in range(1000):
    arrietaSamples.append(np.random.choice(arrietaBBs, size=len(arrietaBBs), replace=True))
    greinkeSamples.append(np.random.choice(greinkeBBs, size=len(greinkeBBs), replace=True))
    kershawSamples.append(np.random.choice(kershawBBs, size=len(kershawBBs), replace=True))

arrietaMeans = [np.mean(obs) for obs in arrietaSamples]
greinkeMeans = [np.mean(obs) for obs in greinkeSamples]
kershawMeans = [np.mean(obs) for obs in kershawSamples]

fig, ax = plt.subplots(figsize=(10, 4))
plt.hist(arrietaMeans, alpha=.5, label='Arrieta', color=ja)
plt.hist(greinkeMeans, alpha=.6, label='Greinke', color=zg)
plt.hist(kershawMeans, alpha=.3, label='Kershaw', color=kc)
plt.legend(loc='best')
plt.xlabel('Avg. Batted Ball Velocity', fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
plt.tick_params(axis='both', which='major', labelsize=13)
ax.get_yaxis().set_ticks([])
plt.show()


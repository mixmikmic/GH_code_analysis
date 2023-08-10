#
#    1. import libraries and files about performances
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib
from IPython.display import display
matplotlib.rcParams['pdf.fonttype'] = 42

selcols = ['nom', 'temps net']

df_list = []
years = range(2006, 2016)
for year in years:
    print('Loading year {}…'.format(year))
    df_list.append(pd.read_csv('runners_{}.csv'.format(year), usecols=selcols))
    print('Shape: {}'.format(df_list[-1].shape))

#
#    2. merge performances
#

df = df_list[0]
c = 0
for item in df_list[1:]:
    #df = merge_years(df, item)
    df = df.merge(item, on='nom', how='outer', suffixes=('_{}'.format(years[c]), '_{}'.format(years[c+1])))
    print('Merge done, new shape:', df.shape)
    display(df.head())
    c += 1
    
df

#
#    3. import and merge years of birth
#

df_years_15 = pd.read_csv('Data_ages_2015.csv', usecols=['catégorie', 'nom', 'année'])
df_years_14 = pd.read_csv('Data_ages_2014.csv', usecols=['catégorie', 'nom', 'année'])
df_years = df_years_15.append(df_years_14)
df_years = df_years.drop_duplicates(subset='nom')
print('Birth years collected, {} duplicates dropped.'.format(827-len(df_years)))

df = df.merge(df_years, on='nom', how='outer')

#
#    4. time conversion
#

#m.ss,c -> (m / 60) + ss,c

from datetime import timedelta
from math import nan

# works well, but ugly for the y axis ticks
def timeToSeconds(value):
    if not isinstance(value, str):
        return value
    try:
        minutes, seconds = value.split('.')
        return (int(minutes) * 60) + float(str.replace(seconds, ',', ''))
    except:
        return None

# doesn't work well
def stringToTime(value):
    if not isinstance(value, str):
        return value
    try:
        _minutes, _seconds = value.split('.')
        return timedelta(minutes=int(_minutes), seconds=float(str.replace(_seconds, ',', '')))
    except:
        return nan
    
for i in range(2006, 2016):
    df['temps s{}'.format(i)] = df['temps net_{}'.format(i)].apply(timeToSeconds)

#
#    5. add other columns: nan values, evolution ratio, sudden drop, age (by dizens)
#

df['nan_values'] = df.nan_values

# waste of time (but, finally.... it worked)
columns = ['temps s{}'.format(i) for i in range(2006, 2016)]

def evolution_ratio(row):
    # we get all float time values -- except the nan ones
    times = [row[i] for i in columns]
    times = [i for i in times if i == i]
    
    if len(times) > 1:
        # return last year - first year
        return times[-1] - times[0]
    else:
        return nan
    
df['time_evolution'] = df.apply(evolution_ratio, axis=1)
df

def get_sudden_drop(row):
    times = [i for i in row if isinstance(i, float)]
    time_keys = [y for y in row.keys() if isinstance(row[y], float)]
    
    sudden_drops = {}
    
    if len(times[:-1]) > 1:
        lasttime = None
        for item, key in zip(times[:-1], time_keys[:-1]):
            if key in ('time_evolution', 'sudden_drop_key', 'année'):
                continue
            if not lasttime: # first item
                if item == item:
                    lasttime = item
            else:
                if (item == item):
                    if (item - lasttime) < -100:# -480: == 8 minutes, too much
                        sudden_drops[key] = item - lasttime
#                        print("Big gap here!", row['nom'], key, item - lasttime)
                    lasttime = item
    if len(sudden_drops) > 0:
        drop_key = max(sudden_drops, key=sudden_drops.get)
        return int(drop_key[-4:])
    else:
        return nan
    
# 480 seconds = 8 minutes
df['sudden_drop_key'] = df.apply(get_sudden_drop, axis=1)

from math import floor
def add_age(value):
    if value == value:
        age = 2015 - value
        if age < 20:
            return 20
        return floor(age/10)*10
    else:
        return nan
    

df['age_10'] = df['année'].apply(add_age)
df['age_10'].value_counts()

#
#    6. two runners made all 10 races
#

df_most_regulars = df[df['time_rows']>9]
df_most_regulars

# Dutoit: 1978
# Gattone: 1976

counter = 0
columns = ['temps net_{}'.format(i) for i in range(2006, 2016)]
xticklabels = ['{} {}'.format(i, 1978-i) for i in years]
print(xticklabels)
fig, ax = plt.subplots(figsize=(12,6))

for idx, runner in df_most_regulars.iterrows():
    runner[columns].plot(label=runner['nom'])

    values = runner[columns].tolist()

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels(range(2006, 2016))
    ax.legend(loc='upper right')
    

    for x, y, year in zip(range(0, 10), values, years):
        ax.text(x+.1, y+15, "{} y.o.".format(year-runner['année']))
    
plt.savefig('two_most_regulars.pdf')

#
#    7. correlation age / sudden drop?
#

df[df['nan_values'] < 6].corr()['année']['sudden_drop_key']

df_6 = df[df['nan_values'] < 6]

lm = smf.ols(formula="sudden_drop_key~année",data=df_6).fit()
fig, ax = plt.subplots()

ax.plot(df_6['année'], df_6['sudden_drop_key'], 'o', label="Data (26 values)")
ax.plot(df_6['année'], lm.fittedvalues, '-', color='red', label="(No-) Prediction")
#ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_yticklabels([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015])

ax.legend(loc="best");
ax.set_ylim([2006, 2016])
ax.set_title('Does your age determine when your performance drop?')
ax.set_ylabel('Worst intra-year drop')
ax.set_xlabel('Year of birth')
plt.savefig('Prediction.pdf')

#
#    8. people of four ages (with the most represented years of birth)
#

nrows = 2; ncols = 2
num_plots = nrows * ncols  # number of subplots
columns = ['temps s{}'.format(i) for i in range(2006, 2016)]

fig = plt.figure(figsize=(14, 10))

axes = [plt.subplot(nrows,ncols,i) for i in range(1,num_plots+1)]

plt.tight_layout(pad=0, w_pad=3, h_pad=1)
plt.subplots_adjust(hspace=.5)

colors = ['#FF2700', '#F6B900', '#77AB43', '#3EA8DC', 'red', 'orange', 'green', 'violet']

count = 0

for age_10, group in df[df['nan_values'] < 10].groupby('année'):
    if len(group) < 5:
        continue
        
    ax = axes[count]

    for idx, runner in group.iterrows():
        ax.plot(runner[columns].tolist())
    
    drop_sentence = ''
    drop_mode = group['sudden_drop_key'].mode()
    if len(drop_mode > 0):
        drop_year = drop_mode[0]
        drop_sentence = '. Most had their worst decline in ' + str(int(drop_year))
        drop_x_tick = drop_year - 2006
        ax.axvline(x=drop_x_tick, ymin=0.3, ymax = 0.9, linewidth=.5, color='#cacaca')

#        ax.plot([drop_x_tick, 0], [drop_x_tick, 3000], color='k', linestyle='-', linewidth=2)
    
    ax.set_title("People born in {}{}".format(int(age_10), drop_sentence))

    ax.set_xlim([-.5, 9.5])
    ax.set_ylim([0, 3000])

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
        


    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015])
    count += 1
    if count > num_plots:
        break

plt.savefig('1969-1970-1973-1976.pdf')

#
#    9. four age groups (18-29, 30-39, 40-49, 50-59)
#

nrows = 2; ncols = 2
num_plots = nrows * ncols  # number of subplots
columns = ['temps s{}'.format(i) for i in range(2006, 2016)]

fig = plt.figure(figsize=(14, 10))

axes = [plt.subplot(nrows,ncols,i) for i in range(1,num_plots+1)]

plt.tight_layout(pad=0, w_pad=3, h_pad=1)
plt.subplots_adjust(hspace=.5)

colors = ['#FF2700', '#F6B900', '#77AB43', '#3EA8DC', 'red', 'orange', 'green', 'violet']

count = 0

for age_10, group in df[df['nan_values'] < 8].groupby('age_10'):

    ax = axes[count]

    for idx, runner in group.iterrows():
        ax.plot(runner[columns].tolist())
    
    drop_sentence = 'No bad year here.'
    drop_mode = group['sudden_drop_key'].mode()
    if len(drop_mode > 0):
        print(drop_mode[0])
        drop_year = drop_mode[0]
        drop_sentence = 'Most had their worst decline in ' + str(int(drop_year))
        drop_x_tick = drop_year - 2006
        ax.axvline(x=drop_x_tick, ymin=0.3, ymax = 0.9, linewidth=.5, color='y')


#    ax.plot([drop_mode, 0], [drop_mode, 3000], color='k', linestyle='-', linewidth=2)
    #ax.axvline(drop_mode, color='r')
    
    ax.set_title("People in their {}es. {}".format(int(age_10), drop_sentence))

    ax.set_xlim([-.5, 9.5])
    ax.set_ylim([0, 3000])

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
        


    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015])
    count += 1
    if count > num_plots:
        break

plt.savefig('20-30-40-50.pdf')

# No.

df[df['time_evolution'] == df['time_evolution']]['time_evolution'].describe()
# min      -662.000000
# 25%      -206.000000
# 50%       -21.000000
# 75%       142.000000
# max       670.000000

#df_bad_evolution = df[df['time_evolution'] < -206]['time_evolution']

lower_limit = df[df['time_evolution'] == df['time_evolution']]['time_evolution'].quantile(q=.3)
upper_limit = df[df['time_evolution'] == df['time_evolution']]['time_evolution'].quantile(q=.7)

df_bad_evolution = df[df['time_evolution'] < lower_limit]
df_medium_evolution = df[(df['time_evolution'] >= lower_limit) & (df['time_evolution'] <= upper_limit) ]
df_good_evolution = df[df['time_evolution'] > upper_limit]

df_regulars = df[df['nan_values'] <= 8]
df_very_regulars = df[df['nan_values']<=2]

df_very_regulars.sort_values(by='time_evolution', ascending=False)

# Useless attempt
#

counter = 0
columns = ['temps net_{}'.format(i) for i in range(2006, 2016)]
xticklabels = ['{} {}'.format(i, 1978-i) for i in years]
print(xticklabels)
fig, ax = plt.subplots(figsize=(18,12))

for idx, runner in df.iterrows():
    runner[columns].plot(label=runner['nom'])

    values = runner[columns].tolist()

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels(range(2006, 2016))
#    ax.legend(loc='best')
    

#    for x, y, year in zip(range(0, 10), values, years):
#        ax.text(x+.1, y+15, "{} y.o.".format(year-runner['Birth']))
    
plt.savefig('very_regulars.pdf')

# For debugging

columns = ['temps s{}'.format(i) for i in range(2006, 2016)]
col_ext = columns
col_ext.append('time_evolution')
df_very_regulars.sort_values(by='time_evolution', ascending=False)[col_ext]

#
#    10. Twelve most regular runners and their evolution, sorted from the better to the worst evolution
#


nrows = 4; ncols = 3
num_plots = nrows * ncols  # number of subplots
columns = ['temps s{}'.format(i) for i in range(2006, 2016)]

fig = plt.figure(figsize=(14, 10))

axes = [plt.subplot(nrows,ncols,i) for i in range(1,num_plots+1)]

plt.tight_layout(pad=0, w_pad=3, h_pad=1)
plt.subplots_adjust(hspace=.5)

colors = ['#FF2700', '#F6B900', '#77AB43', '#3EA8DC', 'red', 'orange', 'green', 'violet']

count = 0

for idx, runner in df_very_regulars.sort_values(by='time_evolution', ascending=True).iterrows():
    ax = axes[count]

    name = runner['nom']
    values = runner[columns].tolist()
    ax.plot(values)
    ax.set_title(name)

    ax.set_xlim([-.5, 9.5])
    ax.set_ylim([0, 2500])

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
        


    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015])
    count += 1
    if count >= num_plots:
        break

plt.savefig('Small_multiples_12runners.pdf')


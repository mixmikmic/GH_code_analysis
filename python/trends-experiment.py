import pandas as pd
import glob
from datetime import datetime

data=[]
for f in glob.glob('data/*.csv'):
    print(f)
    if 'week' not in f:
        name = f.split('data/')[1][:-4]
        df = pd.read_csv(f)
        #print(name)
        dates = list(df['Day'])
        dates = [datetime.strptime(x,'%m/%d/%y') for x in dates]
        df['date'] = dates
        df = df.set_index(pd.Series(df['date']))
        del df['Day']
        del df['date']
        data.append((name,df))

for i,d in enumerate(data):
    print(i,d[0])

thirty=data[0][1]
ninety=data[1][1]
thirty.columns = ['ukip_','bf_']
merged = pd.concat([thirty,ninety],axis=1)
merged = merged['20170118':]

merged

merged.describe()

merged = merged+1 #Adding one to every cell to prevent division errors

merged['ukip_ratio'] = merged['ukip']/merged['ukip_'] #ratio of thirty day to ninety day
merged['ukip_ratio_2'] = merged['ukip_']/merged['ukip'] #ratio of ninety to thirty day

merged

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#Here are the two original series
merged['ukip_'].plot(label='thirty day')
merged['ukip'].plot(label='ninety day')

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

merged['ukip_'].plot(label='thirty day',color='green')
merged['ukip'].plot(label='ninety day',color='blue')
(merged['ukip_']*merged['ukip_ratio'].mean()).plot(label='transformed',color='red')
(merged['ukip']*merged['ukip_ratio_2'].mean()).plot(label='transformed',color='red')
legend = ax.legend(loc='upper left', shadow=False,fontsize=14)

merged['dist30'] = abs(merged['bf_']-merged['ukip_'])

merged['dist90'] = abs(merged['bf']-merged['ukip'])

merged

from scipy.stats import zscore
merged = merged.apply(zscore)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

merged['dist30'].plot(label='normalized thirty day distance between groups',color='green')
merged['dist90'].plot(label='normalized ninety day distance between groups',color='blue')
legend = ax.legend(loc='upper left', shadow=False,fontsize=14)

merged['dist30_ratio'] = merged['dist30']/merged['dist90'] #ratio of thirty day to ninety day
#merged['ukip_ratio_2'] = merged['ukip_']/merged['ukip'] #ratio of ninety to thirty day

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

merged['dist30'].plot(label='normalized thirty day distance between groups',color='green')
merged['dist90'].plot(label='normalized ninety day distance between groups',color='blue')
(merged['dist30']*merged['dist30_ratio'].mean()).plot(label='transformed normalized ninety day distance between groups',color='red')
legend = ax.legend(loc='upper left', shadow=False,fontsize=14)




import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

get_ipython().magic('matplotlib inline')

files = ['2009-Inauguration-Records-Raw.csv','2017-Inauguration-Records-Raw.csv','2017-Womens-March-Records-Raw.csv']
cols = ['station', 'event_date', 'type', 'am_peak', 'midday', 'pm_peak', 'env', 'sum']

# Merge the 3 csv's into one
dataFrames = []
for f in files:
    # Remove commas from numeric entries
    df = pd.read_csv('./ridership-data/'+f,thousands=',')
    df.columns=cols

    # Adding 'event_name' column 
    df['event_name'] = ' '.join(f.split('-')[:-2])
    
    # Split event_date column
    df['date'] = map(lambda x: x.split(',')[0],df.event_date)
    df['day'] = map(lambda x: x.split(',')[1],df.event_date)
    
    # Drop event_date column
    df.drop(['event_date'], axis=1, inplace=True)
    
    dataFrames.append(df)
df = pd.concat(dataFrames)

# Merging station names
df.station = map(lambda x: x.replace('/', '-'),df.station)

# Renaming "2017 Womens March" to "2017 Women's March" 
df.event_name = map(lambda x: x.replace("Womens", "Women's"),df.event_name)

# Rearange column order
cols = ['event_name','date','day','station', 'type', 'am_peak', 'midday', 'pm_peak', 'env', 'sum']
df = df[cols]

print(df.info())
df.sample(10)

# Save out clean data excluding 'SYSTEMWIDE TOTAL' rows
stations = df[df.station!='SYSTEMWIDE TOTAL']
stations.to_csv('ridership-data/Inauguration-Womens-March-Records-Clean.csv',index=False)
# Sample a random station
display(stations[stations.station==stations.sample(1).iloc[0]['station']])

# Save out 'SYSTEMWIDE TOTAL' rows as its own csv
totals = df[df.station=='SYSTEMWIDE TOTAL']
totals.to_csv('ridership-data/Inauguration-Womens-March-Totals.csv',index=False)
display(totals)

# Plot system wide totals
g = sns.factorplot(x='event_name', y='sum', hue='type', data=totals, size=7, 
                   kind='bar', palette='muted');
g.despine(left=True);
g.set_ylabels('Ridership');
g.set_xlabels('');
sns.plt.title('Ridership Totals');

station_sums = pd.DataFrame(stations.groupby(['event_name','type'])['sum'].agg('sum'))
display(station_sums)
display(totals[['event_name','type','sum']])




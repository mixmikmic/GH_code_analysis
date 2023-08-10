import json
import os
import requests
import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
from ohapi import api

user = api.exchange_oauth2_member(os.environ.get('OH_ACCESS_TOKEN'))
fileurl = ''
for entry in user['data']:
    if entry['source'] == "direct-sharing-149":
        fileurl = entry['download_url']
        break

rescuetime_data = requests.get(fileurl).json()

rescuetime_data['row_headers']

date = []
time_spent_seconds = []
activity = []
category = []
productivity = []
for element in rescuetime_data['rows']:
    date.append(element[0])
    time_spent_seconds.append(element[1])
    activity.append(element[3])
    category.append(element[4])
    productivity.append(element[5])
date = [datetime.strptime(dt,"%Y-%m-%dT%H:%M:%S") for dt in date]

rt_df = pd.DataFrame(data={
    'date': date,
    'time_spent_seconds': time_spent_seconds,
    'activity': activity,
    'category': category,
    'productivity': productivity
})

rt_df = rt_df.set_index(rt_df['date'])
rt_df['day'] =  pd.to_datetime(rt_df.index.date)
daily_activity = pd.pivot_table(rt_df,values='time_spent_seconds',index='day',aggfunc=np.sum).to_frame()
daily_activity = daily_activity.reset_index()
daily_activity['hours_spent'] = daily_activity['time_spent_seconds'] / 60 / 60

daily_activity.index = pd.to_datetime(daily_activity['day'])
daily_activity['day'] = None

sns.set_style("white")
sns.set_context("paper", font_scale=1.5)
sns.set_context(rc={"figure.figsize": (15, 8)})
daily_activity.rolling('7d').mean().plot(y='hours_spent')

summed_activity_time = pd.pivot_table(rt_df,values='time_spent_seconds',index='activity',aggfunc=np.sum).to_frame()
summed_activity_time = summed_activity_time.reset_index()
summed_activity_time['time_spent_hours'] = summed_activity_time['time_spent_seconds'] / 60 / 60

plt = sns.stripplot(y='activity',
            x='time_spent_hours',
            data=summed_activity_time.sort_values(by=['time_spent_hours'],ascending=False)[:40],
            size=10)

response = requests.get("https://www.openhumans.org/api/direct-sharing/project/exchange-member/?access_token={}".format(os.environ.get('OH_ACCESS_TOKEN')))
fileurl = ''
for entry in response.json()['data']:
    if entry['source'] == "direct-sharing-102":
        fileurl = entry['download_url']
        fitbit_data = requests.get(fileurl).json()
        break
date = []
steps = []

for year in fitbit_data['tracker-steps'].keys():
    for entry in fitbit_data['tracker-steps'][year]['activities-tracker-steps']:
        date.append(entry['dateTime'])
        steps.append(entry['value'])
        
fitbit_steps = pd.DataFrame(data={
                'date':date,
                'steps': steps})
fitbit_steps['date'] = pd.to_datetime(fitbit_steps['date'])
fitbit_steps = fitbit_steps.set_index('date')

joined_data = daily_activity.join(fitbit_steps)
joined_data['steps'] = joined_data['steps'].apply(int)
joined_data = joined_data[joined_data['steps'] != 0]
joined_data['year'] = joined_data.index.year

sns.lmplot(data=joined_data,
           y='steps',
           x='hours_spent',
           row='year')
sns.plt.ylim(-0.5, None)
sns.plt.xlim(-0.5, None)

daily_activity_productivity = pd.pivot_table(rt_df,values='time_spent_seconds',index=['day'],columns=['productivity'],aggfunc=np.sum)
daily_activity_productivity = daily_activity_productivity.reset_index()

daily_activity_productivity['unproductive'] = daily_activity_productivity[-1] + daily_activity_productivity[-2]
daily_activity_productivity['productive'] = daily_activity_productivity[1] + daily_activity_productivity[2]

daily_activity_productivity = daily_activity_productivity.reset_index()
daily_activity_productivity['unproductive_hours'] = daily_activity_productivity['unproductive'] / 60 / 60
daily_activity_productivity['productive_hours'] = daily_activity_productivity['productive'] / 60 / 60

daily_activity_productivity.index = pd.to_datetime(daily_activity_productivity['day'])

joined_productive_data = daily_activity_productivity.join(fitbit_steps)
joined_productive_data['steps'] = joined_productive_data['steps'].apply(int)
joined_productive_data = joined_productive_data[joined_productive_data['steps'] != 0]
joined_productive_data['year'] = joined_productive_data.index.year
joined_productive_data_long = pd.melt(joined_productive_data,id_vars=['day','steps','year'],value_vars=['unproductive_hours','productive_hours'])

sns.lmplot(data=joined_productive_data_long,
           y='steps',
           x='value',
           col='variable',
           row='year')
sns.plt.ylim(-0.5, 60000)
sns.plt.xlim(-0.5, None)

healthkit_urls = []
for f in response.json()['data']:
    if f['source'] == "direct-sharing-14":
        healthkit_urls.append(f['download_url'])
from collections import defaultdict
healthkit_steps = defaultdict(int)
for url in healthkit_urls:
    healthkit_content = requests.get(url).content
    try:
        healthkit_json = json.loads(healthkit_content)
        for entry in healthkit_json['HKQuantityTypeIdentifierStepCount']:
            date = entry['sdate'][:10]
            steps = entry['value']
            healthkit_steps[date] += steps
    except json.JSONDecodeError: 
        next
healthkit_dates = []
healthkit_data = []
for date,steps in healthkit_steps.items():
    healthkit_dates.append(datetime.strptime(date, '%Y-%m-%d'))
    healthkit_data.append(steps)
    
hk_df = pd.DataFrame(data = {'date': healthkit_dates, 
                          'hk_steps': healthkit_data})
hk_df.sort_values(by=['date'],inplace=True)
hk_df.index = hk_df["date"]

hk_df = hk_df.groupby(hk_df.index.date).sum()
hk_df.index = pd.to_datetime(hk_df.index)

joined_productive_data_hk = daily_activity_productivity.join(hk_df)
joined_productive_data_hk = joined_productive_data_hk[joined_productive_data_hk['hk_steps'] != 0]
joined_productive_data_hk['year'] = joined_productive_data_hk.index.year
joined_productive_data_hk_long = pd.melt(joined_productive_data_hk,id_vars=['day','hk_steps','year'],value_vars=['unproductive_hours','productive_hours'])

sns.lmplot(data=joined_productive_data_hk_long,
           y='hk_steps',
           x='value',
           col='variable',
           row='year')
sns.plt.ylim(-0.5, 60000)
sns.plt.xlim(-0.5, None)


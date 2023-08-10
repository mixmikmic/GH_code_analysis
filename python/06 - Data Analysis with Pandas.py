import pandas as pd

import requests

def get_story(story_id):
    url = 'https://hacker-news.firebaseio.com/v0/item/%d.json' % story_id
    resp = requests.get(url)
    return resp.json()

def get_top_stories():
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
    resp = requests.get(url)
    all_stories = [get_story(sid) for sid in resp.json()[:50]]
    return all_stories

df = pd.read_json('../../data/hn.json')

# df = pd.DataFrame(get_top_stories())

df.head()

df = df.set_index('id')

df.head()

df.by.value_counts()

df.type.value_counts()

df.corr()

df.cov()

df.score.min()

df.score.max()

get_ipython().magic('pylab inline')

df.plot(x='time', y='score', marker='.')

df.sort_values('time').plot(x='time', y='score', marker='.')

df['time'] = pd.to_datetime(df['time'],unit='s')

df.time

df['hour'] = df['time'].map(lambda x: x.hour)

df['hour'].value_counts()

df.corr()

df.plot(x='time', y='score')

df.sort_values('hour').plot(x='hour', y='score')

df['hourly_mean'] = df.groupby('hour')['score'].transform(mean)

df.sort_values('hour').plot(x='hour', y='hourly_mean')

get_ipython().magic('load solutions/data_analysis_solution.py')






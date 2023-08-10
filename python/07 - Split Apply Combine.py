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

df['time'] = pd.to_datetime(df['time'],unit='s')

df['hour'] = df['time'].map(lambda x: x.hour)

df['day_of_week'] = df['time'].map(lambda x: x.weekday())

df.head()

df.groupby('hour')

for group, items in df.groupby('hour'):
    print(group, items)

df.groupby('hour').sum()

df.groupby('hour')['score'].sum()

get_ipython().magic('pylab inline')

df.groupby('hour')['score'].sum().plot()

df['median_hourly_score'] = df.groupby('hour')['score'].transform('median')

df.head()

get_ipython().magic('load solutions/sac_solution.py')




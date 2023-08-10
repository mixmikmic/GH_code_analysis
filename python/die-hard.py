get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.5, 6.0)
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import time
import functools
import requests
from bs4 import BeautifulSoup
from collections import OrderedDict

movies = ['Die Hard', 'Die Hard 2',
          'Die Hard with a Vengeance', 'Live Free or Die Hard', 'A Good Day to Die Hard']
url_base = 'https://en.wikipedia.org'
urls = dict([(m, url_base + '/wiki/' + m.replace(' ', '_')) for m in movies])
urls

def retrieve_cast(url):
    r = requests.get(url)
    print(r.status_code, url)
    time.sleep(3)
    soup = BeautifulSoup(r.text, 'lxml')
    cast = soup.find('span', id='Cast').parent.find_all_next('ul')[0].find_all('li')
    return dict([(li.find('a')['title'], li.find('a')['href']) for li in cast if li.find('a')])


movies_cast = dict([(m, retrieve_cast(urls[m])) for m in movies])
movies_cast

def retrieve_actor(url):
    full_url = url_base + url
    r = requests.get(full_url)
    print(r.status_code, full_url)
    time.sleep(3)
    soup = BeautifulSoup(r.text, 'lxml')
    data = {}
    bday = soup.find(class_='bday')
    if bday:
        data['bday'] = bday.string
    vcard = soup.find('table', class_='vcard')
    if vcard:
        ths = vcard.find_all('th', scope='row')
        th_rows = [th.string.replace('\xa0', ' ') for th in ths]
        th_data = [th.find_next('td').text.replace('\xa0', ' ') for th in ths]
        data.update(dict(zip(th_rows, th_data)))
    return data


cast = {}
for m in movies_cast:
    for act in movies_cast[m]:
        data = retrieve_actor(movies_cast[m][act])
        data[m] = 1
        if data and act in cast:
            cast[act].update(data)
        elif data:
            cast[act] = data

keys = [list(cast[i]) for i in cast]
flat_keys = set(functools.reduce(lambda x, y: x + y, keys))
sorted(flat_keys)

df_name = []
df_columns = {}
for k in flat_keys:
    df_columns[k] = []
for act in cast:
    df_name.append(act)
    for k in df_columns:
        if k in cast[act]:
            df_columns[k].append(cast[act][k])
        else:
            df_columns[k].append(np.nan)
df = pd.DataFrame(df_columns, index=df_name)
df.head()

df.info()

features = ['Die Hard', 'Die Hard 2', 'Die Hard with a Vengeance',
            'Live Free or Die Hard', 'A Good Day to Die Hard',
            'Alma mater', 'Born', 'Children', 'Nationality',
            'Occupation', 'Occupation(s)', 'Partner(s)', 'Spouse(s)',
            'Years active', 'bday', ]
df = df[features].copy()  # throw old `df` away
df.head()

actor = (df['Occupation'].str.contains('[Aa]ctor')
         | df['Occupation(s)'].str.contains('[Aa]ctor')
         | df.index.str.contains('[Aa]ctor'))
actor = actor.astype(np.int)
actor

df['actor'] = actor
df.head()

actress = (df['Occupation'].str.contains('[Aa]ctress')
           | df['Occupation(s)'].str.contains('[Aa]ctress')
           | df.index.str.contains('[Aa]ctress'))
actress = actress.astype(np.int)
df['actress'] = actress
df.head()

no_gender = df[(df['actor'] == 0) & (df['actress'] == 0)]
names = no_gender.index.values
no_gender

df.drop(names, inplace=True)
df.shape

df[df['Born'].isnull() & df['bday'].isnull()]

df.drop('Tom Bower (actor)', inplace=True)
df.shape

no_bday = df[df['bday'].isnull()]
no_bday

byear = no_bday['Years active'].str.extract('(\d+)', expand=False).astype(np.int) - 20
byear

bday = byear.astype(np.str) + '-01-01'
df.loc[df['bday'].isnull(), 'bday'] = bday
df.info()

features = ['Die Hard', 'Die Hard 2', 'Die Hard with a Vengeance',
            'Live Free or Die Hard', 'A Good Day to Die Hard',
            'actor', 'actress', 'bday', ]
df = df[features].copy()
df = df.fillna(0)
df['bday'] = pd.to_datetime(df['bday'])
df.head()

df['age'] = (pd.to_datetime('today') - df['bday']).dt.days // 365.25
df

dh = OrderedDict({'Total': df})
movies = ['Die Hard', 'Die Hard 2',
          'Die Hard with a Vengeance', 'Live Free or Die Hard', 'A Good Day to Die Hard']
for m in movies:
    dh[m] = df[df[m] == 1]
dh.keys()

fig, ax = plt.subplots(figsize=(18, 9))
positions = np.array([])
bar_height = []
bar_width = []
labels = []
colors = []
base_position = np.array([1, 1.10, 1.25, 1.35, 1.50, 1.60])
next_position = 0
for m, data in dh.items():
    actor = data[data['actor'] == 1]
    actress = data[data['actress'] == 1]
    positions = np.append(positions, base_position + next_position)
    bar_height += [len(data), data.age.mean() / 10,
                   data.actor.sum(), actor.age.mean() / 10,
                   data.actress.sum(), actress.age.mean() / 10]
    bar_width += [0.18, 0.11, 0.18, 0.11, 0.18, 0.11]
    labels += [m + ' - Cast', 'Mean Age', m + ' - Actors', 'Actor Age', m + ' - Actressses', 'Actress Age']
    colors += ['yellowgreen', 'mediumseagreen', 'royalblue', 'slateblue', 'crimson', 'lightcoral']
    next_position += 1
rects = ax.bar(positions, bar_height, width=bar_width, tick_label=labels, color=colors)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Die Hard - Cast Gender and Age Comparison', fontsize=16);


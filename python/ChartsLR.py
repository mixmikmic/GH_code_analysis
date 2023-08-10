import datetime as dt
from steemdata import SteemData

import pandas as pd
import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf

# helpers
from toolz import keyfilter

def keep(d, whitelist):
    return keyfilter(lambda k: k in whitelist, d)

def omit(d, blacklist):
    return keyfilter(lambda k: k not in blacklist, d)

db = SteemData().db



# time constraints
time_constraints = {
    '$gte': dt.datetime.now() - dt.timedelta(days=30),
}
conditions = {
    'timestamp': time_constraints,
    'type': {'$in': ['vote', 'comment', 'transfer']},
}
projection = {
    '_id': 0,
    'timestamp': 1,
    'account': 1,
#     'type': 1,
}
ops = list(
    db['AccountOperations'].find(conditions, projection=projection).hint([('timestamp', -1)])
)

ops2 = [{
    'account': x['account'],
    'date': x['timestamp'].date(),
} for x in ops]

from toolz import groupby
from toolz.curried import get

from funcy.colls import pluck
from funcy.seqs import distinct, rest

grouped = groupby(get('date'), ops2)
daily_users = [(k,  len(distinct(pluck('account', v)))) for k, v in grouped.items()]

df = pd.DataFrame(daily_users, columns=['date', 'users'])
df.set_index('date', inplace=True)

df.iloc[1:-1].iplot(
    title='Daily Active Users',
    colors=['blue'],
    theme='white',
    legend=False,
    filename='steemdata-30d-user-count')



# time constraints
time_constraints = {
    '$gte': dt.datetime.now() - dt.timedelta(days=7),
}
conditions = {
    'created': time_constraints,
    'net_votes': {'$gt': 3},
    'children': {'$gt': 1},
}
projection = {
    '_id': 0,
    'identifier': 1,
    'title': 1,
    'author': 1,
    'body': 1,
}
lang_posts = list(db['Posts'].find(conditions, projection=projection))

len(lang_posts)

from langdetect import detect_langs
from funcy.colls import pluck
from funcy.seqs import first, last
from toolz.functoolz import compose, thread_last
from contextlib import suppress
from collections import Counter

def detect(body):
    with suppress(Exception):
        langs = detect_langs(body)
        if langs:
            return first(langs)
    
    return []

languages = thread_last(
    filter(lambda x: len(x['body']) > 100, lang_posts),
    (pluck, 'body'),
    (map, detect),
    (filter, bool)
)

languages = [x.lang for x in languages if x and x.prob > 0.8]

c = Counter(languages)
c.most_common(10)

normalized = [{'language': first(x), 'pct_share': round(last(x) / len(languages) * 100, 3)} for x in c.most_common(10)]

df = pd.DataFrame(normalized)
df.index = range(1,len(df)+1)

df.head(5)

import plotly.plotly as py
import plotly.graph_objs as go

labels = [first(x) for x in c.most_common(7)]
values = [last(x) for x in c.most_common(7)]
colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='label', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

layout = go.Layout(
#     title='Language Domination',
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='styled_pie_chart')

## todo, create a distinct filter on author field, to count % as unique persons, not as number of posts




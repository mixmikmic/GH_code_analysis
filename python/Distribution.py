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



# app specific imports
from funcy.seqs import first
from toolz.functoolz import pipe
from steem.account import Account
from steem.utils import parse_time



from steem.converter import Converter
c = Converter()
min_vests = c.sp_to_vests(100)
max_vests = c.sp_to_vests(50000)

conditions = {
    'balances.available.VESTS': {'$gt': min_vests},
}
projection = {
    '_id': 0,
    'name': 1,
    'balances.available.VESTS': 1,
}
accounts = list(db['Accounts'].find(conditions, projection=projection))

len(accounts)

def last_op_time(username):
    history = Account(username).history_reverse(batch_size=10)
    last_item = first(history)
    if last_item:
        return parse_time(last_item['timestamp'])
    
def filter_inactive(accounts):
    limit = dt.datetime.now() - dt.timedelta(days=180)
    return list(x for x in accounts if x['timestamp'] > limit)

def filter_invalid(accounts):
    return list(x for x in accounts if x['timestamp'])

accounts = [{
    'name': account['name'],
    'timestamp': last_op_time(account['name']),
    'vests': account['balances']['available']['VESTS'],
} for account in accounts]

valid_accounts = pipe(accounts, filter_invalid, filter_inactive)



def maxval(val, _max=max_vests):
    if val > _max:
        return _max
    return val

df = pd.DataFrame(valid_accounts)
df.drop('timestamp', axis=1, inplace=True)

# ignore steemit account
df.drop(df[df.name.isin(['steemit', 'poloniex'])].index, inplace=True)

# ceil max allowable vests
df['vests'] = df['vests'].apply(maxval)

# count the vests, calc % shares
all_vests = df['vests'].sum()
df['pct_share'] = df['vests'] / all_vests * 100
df['token_share'] = df['vests'] / all_vests * 1_000_000



df_sorted = df.sort_values('vests', ascending=False)
df_sorted.head()

df_sorted[['name', 'vests', 'pct_share', 'token_share']].to_json('raw_dist.json', orient='records')

get_ipython().system("cat raw_dist.json | python -m 'json.tool' > distribution.json")



get_ipython().magic('pinfo df_sorted.iplot')

df.sort_values('token_share', ascending=False).head()

df.sort_values('token_share').iplot(
    x='name',
    y='token_share',
    kind='line',
    fill=True,
    title='Token Distribution',
    colors=['blue', 'orange'],
    theme='white',
    legend=False,
    yTitle='Tokens Awarded',
    filename='hypothetical-token-dist'
)




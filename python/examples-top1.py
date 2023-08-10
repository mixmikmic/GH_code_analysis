import pandas as pd
import numpy as np
import itertools
from faker import Faker
import importlib

import d6tjoin.top1
importlib.reload(d6tjoin.top1)
import d6tjoin.utils

# *******************************************************
# generate sample time series data with id and value
# *******************************************************
nobs = 10
f1 = Faker()
f1.seed(0)
uuid1 = [str(f1.uuid4()).split('-')[0] for _ in range(nobs)]
dates1 = pd.date_range('1/1/2010','1/1/2011')

df1 = pd.DataFrame(list(itertools.product(dates1,uuid1)),columns=['date','id'])
df1['v']=np.round(np.random.sample(df1.shape[0]),3)
df1.head()

# create mismatch
df2 = df1.copy()
df2['id'] = df1['id'].str[1:-1]
df2.head()

d6tjoin.utils.PreJoin([df1,df2],['id','date']).stats_prejoin(print_only=False)

result = d6tjoin.top1.MergeTop1(df1.head(),df2,fuzzy_left_on=['id'],fuzzy_right_on=['id'],exact_left_on=['date'],exact_right_on=['date']).merge()

result['top1']['id']

result['merged'].head()

assert not result['duplicates']

dates2 = pd.bdate_range('1/1/2010','1/1/2011') # business instead of calendar dates
df2 = pd.DataFrame(list(itertools.product(dates2,uuid1)),columns=['date','id'])
df2['v']=np.round(np.random.sample(df2.shape[0]),3)

d6tjoin.utils.PreJoin([df1,df2],['id','date']).stats_prejoin(print_only=False)

result = d6tjoin.top1.MergeTop1(df1,df2,fuzzy_left_on=['date'],fuzzy_right_on=['date'],exact_left_on=['id'],exact_right_on=['id']).merge()

result['top1']['date'].head(3)

result['top1']['date'].tail(3)

result['top1']['date']['__top1diff__'].max()

result['merged'].head()

dates2 = pd.bdate_range('1/1/2010','1/1/2011') # business instead of calendar dates
df2 = pd.DataFrame(list(itertools.product(dates2,uuid1)),columns=['date','id'])
df2['v']=np.round(np.random.sample(df2.shape[0]),3)
df2['id'] = df2['id'].str[1:-1]

d6tjoin.utils.PreJoin([df1,df2],['id','date']).stats_prejoin(print_only=False)

result = d6tjoin.top1.MergeTop1(df1,df2,['date','id'],['date','id']).merge()

result['merged'].head()

result['top1']['date'].tail()

result['top1']['id'].head()

dates2 = pd.bdate_range('1/1/2010','1/1/2011') # business instead of calendar dates
df2 = pd.DataFrame(list(itertools.product(dates2,uuid1[:-2])),columns=['date','id'])
df2['v']=np.random.sample(df2.shape[0])
df2['id'] = df2['id'].str[1:-1]

result = d6tjoin.top1.MergeTop1(df1,df2,['date','id'],['date','id']).merge()
result['top1']['id'].head()

result = d6tjoin.top1.MergeTop1(df1,df2,['date','id'],['date','id'], top_limit=[None,2]).merge()

result['top1']['id'].head()

import jellyfish
result = d6tjoin.top1.MergeTop1(df1,df2,['date','id'],['date','id'], fun_diff=[None,jellyfish.hamming_distance]).merge()

result['top1']['id'].head()




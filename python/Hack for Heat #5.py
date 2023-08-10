get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import psycopg2

pd.options.display.max_columns = 40

connection = psycopg2.connect('dbname= threeoneone user=threeoneoneadmin password=threeoneoneadmin')
cursor = connection.cursor()

cursor.execute('''SELECT createddate, closeddate, borough FROM service;''')
data = cursor.fetchall()
data = pd.DataFrame(data)

data.columns = ['createddate','closeddate','borough']

data = data.loc[data['createddate'].notnull()]
data = data.loc[data['closeddate'].notnull()]

data['timedelta'] = data['closeddate'] - data['createddate']

data['timedeltaint'] = [x.days for x in data['timedelta']]

data.head()

data.groupby(by='borough')['timedeltaint'].mean()

data.sort_values('timedeltaint').head()

data.sort_values('timedeltaint', ascending=False).head()

import datetime

today = datetime.date(2016,5,29)
janone = datetime.date(2010,1,1)

subdata = data.loc[(data['closeddate'] > janone) & (data['closeddate'] < today)]
subdata = subdata.loc[data['closeddate'] > data['createddate']]

len(subdata)

subdata.sort_values('timedeltaint').head()

subdata.sort_values('timedeltaint',ascending = False).head()

plotdata = list(subdata['timedeltaint'])

plt.figure(figsize=(12,10))
plt.hist(plotdata);

subdata.quantile([.025, .975])

quantcutdata = subdata.loc[(subdata['timedeltaint'] > 1) & (subdata['timedeltaint'] < 138) ]

len(quantcutdata)

plotdata = list(quantcutdata['timedeltaint'])

plt.figure(figsize=(12,10))
plt.hist(plotdata);

subdata.groupby(by='borough').median()

subdata.groupby(by='borough').mean()


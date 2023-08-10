ls

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('TAQ_CAT_FEB_2010_trading_.csv') as f:
    content = f.read()
    data = content.split('\n')
    columns = list(filter(lambda x:x!='',data[0].split(' '))) 

columns

import sqlite3 as lite
con = lite.connect('cat.db')
with con:
    cur = con.cursor()
    cur.execute('drop table if exists cat')
    create_table_query = 'create table cat' + '(date,hour,minute,second,price,volume)'
    cur.execute(create_table_query)

file_length = len(data)
for i in range(1, file_length):
    row = list(filter(lambda x:x!='',data[i].split(' ')))
    query = '''Insert into '''
    cur.execute('''Insert into cat values (?,?,?,?,?,?)''', row)



with con:
    cur = con.cursor()
    cur.execute('drop table if exists normal')
    cur.execute("""create table normal as select * from cat where hour||minute >= '930' OR hour||minute <= '1600'""")
    cur.execute("""select * from normal""")
    all_data = cur.fetchall()

import pandas as pd
normal_df = pd.DataFrame(all_data,columns = columns)
normal_df.columns = ['date','hour','minute','second','price','volume']
normal_df['volume'] = normal_df['volume'].astype(float)
normal_df['price'] = normal_df['price'].astype(float)
normal_df['time'] = normal_df['date']+normal_df['hour']+normal_df['minute']+normal_df['second']
normal_df['time'] = pd.to_datetime(normal_df['time'],format = '%Y%m%d%H%M%S')

normal_df.head()

first5000 = normal_df.ix[:5000,:]
last5000 = normal_df.iloc[-5000:]
first5000_filtered = first5000[first5000['volume'] < 2000]
last5000_filtered = last5000[last5000['volume'] < 2000]

fig, axs = plt.subplots(1,2)
first5000.groupby('time').sum()['volume'].plot(title = 'First 5000 Transaction Volume Volatility by second',
                                               figsize=(15,6),color = 'red',
                                              ax = axs[0])
first5000_filtered.groupby('time').sum()['volume'].plot(title = 'First 5000 Transaction Volume Volatility by second (Outlier Removed)',
                                                        figsize=(15,6),color = 'blue',
                                                       ax = axs[1])

fig, axs = plt.subplots(1,2)
last5000.groupby('time').sum()['volume'].plot(title = 'Last 5000 Transaction Volume Volatility by second',
                                               figsize=(15,6),color = 'red',
                                              ax = axs[0])
last5000_filtered.groupby('time').sum()['volume'].plot(title = 'Last 5000 Transaction Volume Volatility by second (Outlier Removed)',
                                                        figsize=(15,6),color = 'blue',
                                                       ax = axs[1])



price_change = list(normal_df['price'][:-1].values - normal_df['price'][1:].values)
print("%d times price change" % (len(price_change) - price_change.count(0)))

changed_price = list(filter(lambda x: x[1]!=0, list(enumerate(price_change))))

x = list(map(lambda x:x[0],changed_price))
y = list(map(lambda x:x[1],changed_price))
plt.figure(figsize=(16,5))
pd.DataFrame(data = y, index = x,columns = ['price change']).plot(figsize=(15,6))

plt.hist(y,bins = 500)
plt.xlim(-0.05,0.05)
plt.title('Price Change')



intervals = ['5Min','10Min','30Min','60Min']

for interval in intervals:
    grouped = normal_df.groupby(pd.TimeGrouper(key = 'time',freq=interval)).aggregate(np.sum)
    S_i = grouped['price']
    S_i_minus_1 = grouped['price'].shift(1)
    grouped['log_return'] = np.log(S_i/S_i_minus_1)
    grouped.to_csv('TAQ_JNJ_1004_1015_2010_'+interval+'trading_unit.csv')



for interval in intervals:
    grouped = normal_df.groupby(pd.TimeGrouper(key = 'time',freq=interval)).aggregate(np.mean)
    S_i = grouped['price']
    S_i_minus_1 = grouped['price'].shift(1)
    grouped['log_return'] = np.log(S_i/S_i_minus_1)
    n = len(grouped)
    x = (1/(n-1))*np.sum(np.log(S_i/S_i_minus_1)*np.log(S_i/S_i_minus_1))
    y = 1/((n-1)*n)*np.sum(np.log(S_i/S_i_minus_1))*np.sum(np.log(S_i/S_i_minus_1))
    section_vol = np.sqrt(x-y)
    vol = grouped['volume']
    x = (1/(n-1))*np.sum(vol*vol)
    y = 1/((n-1)*n)*np.sum(vol)*np.sum(vol)
    vol_section = np.sqrt(x-y)
    u_mean = np.mean(np.log(S_i/S_i_minus_1))
    skewness_0 = np.sum((np.log(S_i/S_i_minus_1)-u_mean)**3)
    skewness = n*skewness_0/((n-1)*(n-2)*section_vol**3)
    kurtosis_0 = np.sum((np.log(S_i/S_i_minus_1)-u_mean)**4)
    kurtosis = (n*skewness_0/(n*section_vol**4))-3
    print("""FOR %s TRADING UNIT:\n
    section volatility is %s \n
    volume section standard deviation is %s\n
    skewness is %s\n
    kurtosis is %s\n"""%(interval, section_vol,vol_section, skewness, kurtosis))      




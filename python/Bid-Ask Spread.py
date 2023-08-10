ls

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('TAQ_CAT_QUOTE_0104_2010.csv') as f:
    content = f.read()
    data = content.split('\n')
    columns = list(filter(lambda x:x!='',data[0].split(' '))) 

columns

import sqlite3 as lite
con = lite.connect('quote.db')
with con:
    cur = con.cursor()
    cur.execute('drop table if exists quote')
    create_table_query = 'create table quote' + '(date,hour,minute,second,bid,ofr,bidsize,ofrsize)'
    cur.execute(create_table_query)

file_length = len(data)
for i in range(1, file_length):
    row = list(filter(lambda x:x!='',data[i].split(' ')))
    query = '''Insert into '''
    cur.execute('''Insert into quote values (?,?,?,?,?,?,?,?)''', row)



with con:
    cur = con.cursor()
    cur.execute("""select * from quote""")
    all_data = cur.fetchall()

import pandas as pd
normal_df = pd.DataFrame(all_data,columns = columns)
normal_df.columns = ['date','hour','minute',
                     'second','bid','ofr','bidsize','ofrsize']
normal_df['bidsize'] = normal_df['bidsize'].astype(float)
normal_df['bid'] = normal_df['bid'].astype(float)
normal_df['ofr'] = normal_df['ofr'].astype(float)
normal_df['ofrsize'] = normal_df['ofrsize'].astype(float)
normal_df['time'] = normal_df['date']+normal_df['hour']+normal_df['minute']+normal_df['second']
normal_df['time'] = pd.to_datetime(normal_df['time'],format = '%Y%m%d%H%M%S')

normal_df['bid_offer_diff'] = normal_df['ofr'] - normal_df['bid']

normal_df.head()



intervals = ['5Min','10Min','30Min']

for interval in intervals:
    grouped = normal_df.groupby(pd.TimeGrouper(key = 'time',freq=interval)).aggregate(np.mean)
    grouped.to_csv('TAQ_CAT_QUOTE_0104_2010_'+interval+'trading_unit.csv')
    V_offer = grouped['ofrsize']
    p = V_offer/np.sum(V_offer)
    V_bid = grouped['bidsize']
    q = V_bid/np.sum(V_bid)
    p.replace({0:1})
    H_bid= -1*np.sum(q*np.log2(q))
    H_offer= -1*np.sum(p*np.log2(p))
    entropy = np.sum(p*np.log2(p/q))
    print('For %s trading unit, entropy value is %s'%(interval, entropy))




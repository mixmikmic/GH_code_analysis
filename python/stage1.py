import pandas as pd
import numpy as np

data = pd.read_csv('2018-02-01/2018-02-01_BINS_XETR08.csv')

data.columns

print(len(data.dropna()))
print(len(data))

len(data.ISIN.unique())

len(data.Mnemonic.unique())

len(data.SecurityDesc.unique())

len(data.SecurityID.unique())

from collections import defaultdict as dd
d = dd(lambda:[])
e = dd(lambda:[])
count = 0
failure = 0
for i,j in zip(data['ISIN'],data['Mnemonic']):
    if i not in d[j]:
        d[j].append(i)
    if len(d[j])>1:
        print('failure')
        failure = 1
        break
if failure==0:
    for i,j in zip(data['ISIN'],data['Mnemonic']):
        if j not in e[i]:
            e[i].append(j)
        if len(e[i])>1:
            print('failure')
            failure = 1
            break
if failure==0:
    print('all good! delete one of them!')

del data['ISIN']

e = dd(lambda:[])
f = dd(lambda:[])
count = 0
failure = 0
for i,j in zip(data['SecurityID'],data['Mnemonic']):
    if i not in e[j]:
        e[j].append(i)
    if len(e[j])>1:
        print('failure')
        failure = 1
        break
if failure==0:
    for i,j in zip(data['SecurityID'],data['Mnemonic']):
        if j not in f[i]:
            f[i].append(j)
        if len(f[i])>1:
            print('failure')
            failure = 1
            break
if failure==0:
    print('all good! delete one of them!')

del data['SecurityID']

f = dd(lambda:[])
g = dd(lambda:[])
count = 0
failure = 0
for i,j in zip(data['SecurityDesc'],data['Mnemonic']):
    if i not in f[j]:
        f[j].append(i)
    if len(f[j])>1:
        print('failure')
        failure = 1
        break
if failure==0:
    for i,j in zip(data['SecurityDesc'],data['Mnemonic']):
        if j not in g[i]:
            g[i].append(j)
        if len(g[i])>1:
            print(i, end = ': ')
            for k in g[i]:
                print(k, end = ', ')
            print(' ')

del data['SecurityDesc']

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(data.groupby('Mnemonic').count()['NumberOfTrades'])
len(data.groupby('Mnemonic').count()['NumberOfTrades'][data.groupby('Mnemonic').count().NumberOfTrades > 59])

mens = []
for i in data.Mnemonic.unique():
    if len(data[data.Mnemonic==i])>59:
        mens.append(i)

subed = data[[x in mens for x in data.Mnemonic]]

subed.SecurityType.unique()

del subed['SecurityType']

subed.Currency.unique()

del subed['Currency']

len(subed)

lht = pd.read_csv('2018-02-01/2018-02-01_BINS_XETR16.csv')
lht.Time.unique()

import pandas as pd
import numpy as np
days = ['01','02','05','06','07','08','09','12','13','14','15','16','19','20','21','22','23','26','27','28']
hours = ['08','09','10','11','12','13','14','15','16']
dfs = []
for i in days:
    for j in hours:
        fpath = '2018-02-' + i + '/2018-02-' + i + '_BINS_XETR' + j + '.csv'
        try:
            dfs.append(pd.read_csv(fpath))
        except:
            print('day ' + i + ' hour ' + j + ' is missing')
            continue
data = pd.concat(dfs)

from collections import defaultdict as dd
delfields = ['ISIN', 'SecurityID']
for k in delfields:
    d = dd(lambda:[])
    e = dd(lambda:[])
    count = 0
    failure = 0
    for i,j in zip(data[k],data['Mnemonic']):
        if i not in d[j]:
            d[j].append(i)
        if len(d[j])>1:
            failure = 1
    if failure==0 or failure==1:
        for i,j in zip(data[k],data['Mnemonic']):
            if j not in e[i]:
                e[i].append(j)
            if len(e[i])>1:
                failure = 1
    if failure==0:
        print('all good! deleting ' + k + '!')
        del data[k]
    for k,v in d.items():
        if len(v)>1:
            print(k)
            print(v)
            print(' ')
    print(' ')
    alb = True
    for k,v in e.items():
        if len(v)>1:
            alb = False
            print(k)
            print(v)
            print(' ')
    print(alb)

import pandas as pd
import numpy as np
from collections import defaultdict as dd
days = ['01','02','05','06','07','08','09','12','13','14','15','16','19','20','21','22','23','26','27','28']
hours = ['08','09','10','11','12','13','14','15','16']
dfs = []
initial = pd.read_csv('2018-02-01/2018-02-01_BINS_XETR08.csv')
mend = dd(int)
for i in days:
    for j in hours:
        fpath = '2018-02-' + i + '/2018-02-' + i + '_BINS_XETR' + j + '.csv'
        try:
            dat2 = pd.read_csv(fpath)
            mens = []
            for k in dat2.Mnemonic.unique():
                if len(dat2[dat2.Mnemonic==k])>len(dat2.Time.unique())-1:
                    mens.append(k)
            for k in mens:
                mend[k]+=1
        except:
            print('day ' + i + ' hour ' + j + ' is missing')
            continue
for k,v in mend.items():
    print(k+': '+str(v))

for k,v in mend.items():
    if v>100:
        print(k)

fmends = []
for k,v in mend.items():
    if v>100:
        fmends.append(k)
print(fmends)

data2 = data[[x in fmends for x in data.Mnemonic]]

from collections import defaultdict as dd
d = dd(lambda:[])
e = dd(lambda:[])
count = 0
failure = 0
for i,j in zip(data2['ISIN'],data2['Mnemonic']):
    if i not in d[j]:
        d[j].append(i)
    if len(d[j])>1:
        print('failure')
        failure = 1
        break
if failure==0:
    for i,j in zip(data2['ISIN'],data2['Mnemonic']):
        if j not in e[i]:
            e[i].append(j)
        if len(e[i])>1:
            print('failure')
            failure = 1
            break
if failure==0:
    print('all good! deleting ISIN!')
    del data2['ISIN']

e = dd(lambda:[])
f = dd(lambda:[])
count = 0
failure = 0
for i,j in zip(data2['SecurityID'],data2['Mnemonic']):
    if i not in e[j]:
        e[j].append(i)
    if len(e[j])>1:
        print('failure')
        failure = 1
        break
if failure==0:
    for i,j in zip(data2['SecurityID'],data2['Mnemonic']):
        if j not in f[i]:
            f[i].append(j)
        if len(f[i])>1:
            print('failure')
            failure = 1
            break
if failure==0:
    print('all good! deleting SecurityID!')
    del data2['SecurityID']

e = dd(lambda:[])
f = dd(lambda:[])
count = 0
failure = 0
for i,j in zip(data2['SecurityDesc'],data2['Mnemonic']):
    if i not in e[j]:
        e[j].append(i)
    if len(e[j])>1:
        print('failure')
        failure = 1
        break
if failure==0:
    for i,j in zip(data2['SecurityDesc'],data2['Mnemonic']):
        if j not in f[i]:
            f[i].append(j)
        if len(f[i])>1:
            print('failure')
            failure = 1
            break
if failure==0:
    print('all good! deleting Security Description!')
    del data2['SecurityDesc']

if(len(data2.SecurityType.unique()) ==1):
    print('Deleting Security Type')
    print('All securities are of type '+ data2.SecurityType.unique()[0])
    del data2['SecurityType']

del data2['SecurityType']

data2.columns

if(len(data2.Currency.unique()) ==1):
    print('Deleting Currency')
    print('All currencies are of type '+ data2.Currency.unique()[0])
    del data2['Currency']

print(fmends)
print(data2.Mnemonic.unique())

data2.head(30)

len(data2)

dts = pd.to_datetime([x+y for x,y in zip(data2['Date'], data2['Time'])], format = '%Y-%m-%d%H:%M')

dts[:10]

data2['DateTime'] = dts

data2.head()

del data2['Date']
del data2['Time']

import pickle
with open('check1.pkl', 'wb') as picklefile:
    pickle.dump(data2, picklefile)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import defaultdict as dd
days = ['01','02','05','06','07','08','09','12','13','14','15','16','19','20','21','22','23','26','27','28']
mend2 = dd(int)
for i in days:
    for j in hours:
        fpath = '2018-02-' + i + '/2018-02-' + i + '_BINS_XETR' + j + '.csv'
        try:
            dat2 = pd.read_csv(fpath)
            for k in dat2.Mnemonic:
                mend2[k]+=1
        except:
            print('day ' + i + ' hour ' + j + ' is missing')
            continue
totalT = []
for k,v in mend2.items():
    totalT.append(v)
plt.hist(totalT)

for k,v in mend2.items():
    if v>9900:
        print(k)
        print(v)


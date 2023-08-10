import pickle
import pandas as pd
import numpy as np

with open('check1.pkl','rb') as picklefile:
    data = pickle.load(picklefile)

data.head()

data.reset_index(inplace = True)
del data['index']
data.head()

data.Mnemonic.unique()

dfs = []
for i in data.Mnemonic.unique():
    dfs.append(data[data.Mnemonic==i])

dfs[0].head()

for i in dfs:
    i.reset_index(inplace=True)
    del i['index']
dfs[0].head()

for i in dfs:
    print(min(i['DateTime']))

allTimes = sorted(data.DateTime.unique())
q = 0
for i in dfs:
    for j in allTimes:
        if j in i['DateTime'] and len(i[i.DateTime==j])>1:
            print('DANGER')
            q = 1
print(q)

from copy import deepcopy
allTimes = sorted(data['DateTime'].drop_duplicates())
dfsnew = []
for i in dfs:
    df = pd.DataFrame(columns = ['Mnemonic','StartPrice','MaxPrice','MinPrice','EndPrice','TradedVolume','NumberOfTrades','DateTime'])
    j = 0
    base = i[i.DateTime==min(i['DateTime'])]
    print(base)
    base.reset_index(inplace = True)
    iTimes = sorted(i['DateTime'].drop_duplicates())
    for k in range(len(allTimes)):
        if allTimes[k]>base.DateTime[0]:
            if allTimes[k] not in iTimes:
                df.loc[j] = [base['Mnemonic'][0],base['EndPrice'][0],base['EndPrice'][0],base['EndPrice'][0],base['EndPrice'][0],0,0,allTimes[k]]
                j+=1
            else:
                base = i[i.DateTime==allTimes[k]]
                base.reset_index(inplace = True)
    i = pd.concat([i,df])
    i.sort_values('DateTime',inplace=True)
    print(len(i))
    dfsnew.append(deepcopy(i))
for i in dfsnew:
    print(len(i))

dfsnew2=[]
allTimes = sorted(data['DateTime'].drop_duplicates())
for i in dfsnew:
    df = pd.DataFrame(columns = ['Mnemonic','StartPrice','MaxPrice','MinPrice','EndPrice','TradedVolume','NumberOfTrades','DateTime'])
    j = 0
    base = i[i.DateTime==min(i['DateTime'])]
    print(base)
    base.reset_index(inplace = True)
    iTimes = sorted(i['DateTime'].drop_duplicates())
    for k in range(len(allTimes)):
        if allTimes[k]<base.DateTime[0]:
                df.loc[j] = [base['Mnemonic'][0],base['StartPrice'][0],base['StartPrice'][0],base['StartPrice'][0],base['StartPrice'][0],0,0,allTimes[k]]
                j+=1
        else:
            break
    i = pd.concat([i,df])
    i.sort_values('DateTime',inplace=True)
    print(len(i))
    dfsnew2.append(i)

mDays = ['06','09']
mHr = ['15']
mMin = ['00','01','02','03','04','05','06','07','08','09'] + [str(x) for x in range(10,60)]
MTIMES = pd.to_datetime(['201802'+x+y+z for x in mDays for y in mHr for z in mMin], format = '%Y%m%d%H%M')
print(len(MTIMES))

dfsnew3=[]
for i in dfsnew2:
    flag = 0
    print(i.head(1))
    df = pd.DataFrame(columns = ['Mnemonic','StartPrice','MaxPrice','MinPrice','EndPrice','TradedVolume','NumberOfTrades','DateTime'])
    iTimes = sorted(i['DateTime'].drop_duplicates())
    print(len(iTimes))
    if len(iTimes)!=len(i['DateTime']):
        print('Danger! Code Blue!')
        break
    j = 0
    base = i[i.DateTime==iTimes[0]]
    base.reset_index(inplace = True)
    u = 0
    while(iTimes[u] < min(MTIMES)):
        base = i[i.DateTime==iTimes[u]]
        base.reset_index(inplace = True)
        u+=1
    if(len(base)>1):
        print('Danger! Code Orange!')
        break
    for k in range(60):
        if MTIMES[k] not in iTimes:
                df.loc[j] = [base['Mnemonic'][0],base['EndPrice'][0],base['EndPrice'][0],base['EndPrice'][0],base['EndPrice'][0],0,0,MTIMES[k]]
                j+=1
        else:
            print('Danger! Code Red!')
            break
    while(iTimes[u] < MTIMES[60]):
        base = i[i.DateTime==iTimes[u]]
        base.reset_index(inplace=True)
        u+=1
    for k in range(60,120):
        if MTIMES[k] not in iTimes:
            df.loc[j] = [base['Mnemonic'][0],base['EndPrice'][0],base['EndPrice'][0],base['EndPrice'][0],base['EndPrice'][0],0,0,MTIMES[k]]
            j+=1
        else:
            print('Danger! Code Red Second!')
            break
    while(u<len(iTimes) and iTimes[u] not in MTIMES):
        u+=1
    if(u<len(iTimes)):
        print('Danger! Code Purple!')
    i = pd.concat([i,df])
    i.sort_values('DateTime',inplace=True)
    print(len(i))
    dfsnew3.append(i)    

nDays = ['01','02','05','06','07','08','09'] + [str(x) for x in range(12,17)] + [str(x) for x in range(19,24)] + ['26','27','28']
nHr = ['08','09']+ [str(x) for x in range(10,16)]
nHr2 = ['16']
nMin = ['00','01','02','03','04','05','06','07','08','09'] + [str(x) for x in range(10,60)]
nMin2 = ['0'+str(x) for x in range(10)] + [str(x) for x in range(10,31)]
NTIMES = pd.to_datetime(['201802'+x+y+z for x in nDays for y in nHr for z in nMin], format = '%Y%m%d%H%M')
NTIMES2 = pd.to_datetime(['201802'+x+y+z for x in nDays for y in nHr2 for z in nMin2], format = '%Y%m%d%H%M')
NTIMES3 = NTIMES.union(NTIMES2)
print(NTIMES3[0:10])
print(len(NTIMES3))

NTIMES3 = sorted(NTIMES3)
dfsnew4=[]
for i in dfsnew3:
    df = pd.DataFrame(columns = ['Mnemonic','StartPrice','MaxPrice','MinPrice','EndPrice','TradedVolume','NumberOfTrades','DateTime'])
    iTimes = sorted(i['DateTime'].drop_duplicates())
    print(len(iTimes))
    if len(iTimes)!=len(i['DateTime']):
        print('Danger! Code Blue!')
        break
    j = 0
    base = i[i.DateTime==iTimes[0]]
    base.reset_index(inplace = True)
    u = 0
    while u<len(NTIMES3):
        while(u<len(NTIMES3) and NTIMES3[u] in iTimes):
            if(u<len(iTimes)):
                base = i[i.DateTime==iTimes[u]]
            else:
                base = i[i.DateTime==iTimes[-1]]
            if(len(base)>1):
                print('Danger! Code Orange! Internal!')
                print(u)
            base.reset_index(inplace = True)
            u+=1
        while(u<len(NTIMES3) and NTIMES3[u] not in iTimes):
            df.loc[j] = [base['Mnemonic'][0],base['EndPrice'][0],base['EndPrice'][0],base['EndPrice'][0],base['EndPrice'][0],0,0,NTIMES3[u]]
            j+=1
            u+=1
    i = pd.concat([i,df])
    i.sort_values('DateTime',inplace=True)
    print(len(i))
    dfsnew4.append(i) 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (35,20))
i = 0
plt.plot(dfsnew4[i]['DateTime'],dfsnew4[i]['StartPrice']);

import pickle
with open('check2.pkl', 'wb') as picklefile:
    pickle.dump(dfsnew4, picklefile)




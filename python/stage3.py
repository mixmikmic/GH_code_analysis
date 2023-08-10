import pickle
import pandas as pd
import numpy as np
with open('check2.pkl','rb') as picfile:
    data = pickle.load(picfile)

for i in range(10):
    data[i].reset_index(inplace=True)

for i in range(10):
    del data[i]['index']

data[3].head()

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}

matplotlib.rc('font', **font)
fig = plt.figure(figsize = (35,20))
i = 3
plt.plot(data[i]['DateTime'],data[i]['StartPrice'])
plt.title('BMW Price Movement Chart for February 2018');
plt.xlabel('Time')
plt.ylabel('Price')

dfs = []
for i in data:
    i['logSP'] = [np.log(x) for x in i['StartPrice']]
    i['logEP'] = [np.log(x) for x in i['EndPrice']]
    i['logMax'] =[np.log(x) for x in i['MaxPrice']]
    i['logMin'] = [np.log(x) for x in i['MinPrice']]
    dfs.append(i)

dfs[0].head()

dfsnew = []
for j in data:
    j['TVPrior'] = j['TradedVolume'].shift(1)
    j['NTPrior'] = j['NumberOfTrades'].shift(1)
    j['logC'] = [x-y for x,y in zip(j['logEP'],j['logSP'])]
    j['deLogSP'] = [x-y for x,y in zip(j['logSP'], j['logSP'].shift(1))]
    j['deLogEP'] = [x-y for x,y in zip(j['logEP'], j['logEP'].shift(1))]
    j['deLogMax'] = [x-y for x,y in zip(j['logMax'], j['logMax'].shift(1))]
    j['deLogMin'] = [x-y for x,y in zip(j['logMin'], j['logMin'].shift(1))]
    dfsnew.append(j)

print(dfsnew[0].head(3))

suffixes = [str(x) for x in range(1,31)]
shifts = list(range(1,31))
dfsnew2 = []
for j in dfsnew:
    for i in range(30):
        j['TVLag'+suffixes[i]] = j['TVPrior'].shift(shifts[i])
        j['NTLag'+suffixes[i]] = j['NTPrior'].shift(shifts[i])
        j['logCL'+suffixes[i]] = j['logC'].shift(shifts[i])
        j['deLogSPL'+suffixes[i]] = j['deLogSP'].shift(shifts[i])
        j['deLogEPL'+suffixes[i]] = j['deLogEP'].shift(shifts[i])
        j['deLogMax'+suffixes[i]] = j['deLogMax'].shift(shifts[i])
        j['deLogMin'+suffixes[i]] = j['deLogMin'].shift(shifts[i])
    print(j.head(1))
    dfsnew2.append(j)

dfsnew2[0]['deLogSP'][0:10]

dfsnew3 = []
for i in dfsnew2:
    i.fillna(value = 0, inplace = True)
    dfsnew3.append(i)

dfsnew3[0]['deLogSP'][0:10]

dfsnew3[0].head()

for i in range(len(dfsnew3)):
    path = 'check3'+str(i)+'.pkl'
    with open(path, 'wb') as picklefile:
        pickle.dump(dfsnew3[i], picklefile)
    print(i)




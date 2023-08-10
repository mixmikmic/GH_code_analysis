import pickle
import pandas as pd
import numpy as np
with open('check2.pkl','rb') as picfile:
    data = pickle.load(picfile)
for i in range(10):
    data[i].reset_index(inplace=True)
for i in range(10):
    del data[i]['index']

data[0].head()

dfs = []
for j in data:
    del j['Mnemonic']
    del j['MaxPrice']
    del j['MinPrice']
    #del j['TradedVolume']
    del j['NumberOfTrades']
    del j['DateTime']
    #j['logCV'] = [np.log(x+1)-np.log(y+1) for x,y in zip(j['TradedVolume'].shift(1).fillna(value=0),j['TradedVolume'].shift(2).fillna(value=0))]
    j['logC'] = [np.log(x)-np.log(y) for x,y in zip(j['EndPrice'],j['StartPrice'])]
    del j['StartPrice']
    del j['EndPrice']
    dfs.append(j)
with open('check3x1.pkl', 'wb') as picklefile:
    pickle.dump(dfs, picklefile)

dfs[0].head()




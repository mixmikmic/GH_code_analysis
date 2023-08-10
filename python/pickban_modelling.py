import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

h_bp = pd.read_csv('hltv_csv/picksAndBans.csv').set_index('Match ID')
h_matches = pd.read_csv('hltv_csv/matchResults.csv').set_index('Match ID')[['Date', 'Team 1 ID', 'Team 2 ID', 'Map']]
h_matches.columns = ['Date', 'Team 1 ID', 'Team 2 ID', 'Map Played']
h_bp = h_bp.join(h_matches, how='left')
h_bp['Date'] = pd.to_datetime(h_bp['Date'])
h_matches['Date'] = pd.to_datetime(h_matches['Date'])
h_teams = pd.read_csv('hltv_csv/teams.csv').set_index('ID')

train = h_bp[(h_bp.Date < dt.datetime(2017,6,1)) & (h_bp.Date >= dt.datetime(2017,1,1))]
train_matches = h_matches[(h_matches.Date < dt.datetime(2017,6,1)) & (h_matches.Date >= dt.datetime(2017,1,1))]
valid = h_bp[h_bp.Date >= dt.datetime(2017,6,1)]
valid_matches = h_matches[(h_matches.Date >= dt.datetime(2017,6,1))]

filt = np.load('saved_model/eslpl/filter_teams.npy')
team_ids = np.load('saved_model/eslpl/teams.npy')
ht_filt = h_teams.loc[team_ids]
ht_filt = ht_filt[ht_filt.Name.isin(filt)]

def model_mp(train, t1, t2):
    tab = train[train['Team'].isin([t1, t2])].groupby(['Team', ' Pick Type', 'Map'])['Date'].count().unstack([' Pick Type', 'Team']).fillna(0)
    tab = (tab['picked']/tab['picked'].sum(axis=0)).mean(axis=1)# get average
    return (tab/tab.sum(axis=0)) # normalize

def model_mb(train, t1, t2):
    tab = train[train['Team'].isin([t1, t2])].groupby(['Team', ' Pick Type', 'Map'])['Date'].count().unstack([' Pick Type', 'Team']).fillna(0)
    tab = (tab['removed']/tab['removed'].sum(axis=0)).mean(axis=1)# get average
    return (tab/tab.sum(axis=0)) # normalize

def model_mix(train, t1, t2):
    tab = train[train['Team'].isin([t1, t2])].groupby(['Team', ' Pick Type', 'Map'])['Date'].count().unstack([' Pick Type', 'Team']).fillna(0)
    tab = (tab/tab.sum(axis=0)).mean(level=0,axis=1)
    tab['removed'] = (tab['removed']**-1)/(tab['removed']**-1).sum(axis=0)
    return tab.mean(axis=1)

def model_played(train, t1, t2):
    a = train[train['Team 1 ID'].isin([t1,t2])].groupby(['Team 1 ID', 'Map Played'])['Date'].count()
    b = train[train['Team 2 ID'].isin([t1,t2])].groupby(['Team 2 ID', 'Map Played'])['Date'].count()
    c = pd.DataFrame([a,b], index=['a','b']).T.fillna(0)
    c = (c['a']+c['b']).unstack(level=0).fillna(0)
    return (c/c.sum()).mean(axis=1)

err = []
for i in range(len(filt)):
    for j in range(i+1,len(filt)):
        t1 = ht_filt[ht_filt.Name == filt[i]].index[0]; t2 = ht_filt[ht_filt.Name == filt[j]].index[0]
        hup_m = valid_matches[(((valid_matches['Team 1 ID'] == t1)&(valid_matches['Team 2 ID'] == t2)) | 
                  ((valid_matches['Team 2 ID'] == t1)&(valid_matches['Team 1 ID'] == t2)))]
        if(len(hup_m) >= 8):
            predicted = model_played(train_matches, t1, t2)
            actual = hup_m.groupby('Map Played')['Date'].count()/hup_m.groupby('Map Played')['Date'].count().sum()
            df = pd.DataFrame([predicted, actual], index=['pred', 'y']).fillna(0).T.sort_values('pred', ascending=False)
            print('%s vs %s' % (filt[i], filt[j]))
            print(df)
            err.append(((df['pred']-df['y'])**2).sum())

print(np.mean(err))

err = []
for i,r in valid_matches[valid_matches['Team 1 ID'].isin(ht_filt.index) & valid_matches['Team 2 ID'].isin(ht_filt.index)].iterrows():
    t1 = r['Team 1 ID']; t2 = r['Team 2 ID']
    t1_name = ht_filt.loc[t1].Name; t2_name = ht_filt.loc[t2].Name;
    predicted = model_played(train_matches, t1, t2).sort_values().tail(1).index[0]
    #predicted = model_mp(train, t1_name, t2_name).sort_values().tail(1).index[0]
    err.append([predicted, r['Map Played']])
err = np.array(err)

from sklearn.metrics import accuracy_score
accuracy_score(err[:,1], err[:,0])

accuracy_score(err[:,1], err[:,0])

accuracy_score(err[:,1], err[:,0])

accuracy_score(err[:,1], err[:,0])




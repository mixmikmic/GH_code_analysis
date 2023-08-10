import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
from wl_model.spcl_case import *
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

import cfscrape # need nodejs
import json
scraper = cfscrape.create_scraper()
games = json.loads(scraper.get("http://thunderpick.com/api/matches").content)
games = pd.DataFrame(games['data'])
games = games[games.gameId == 6].sort_values('startTime')

bet_games = []
for i,v in games.iterrows():
    if((v['isTournament'] == False )& (v['canWager'] == True)):
        ratio = v['matchBet']['buckets'][0]['amount']/v['matchBet']['buckets'][1]['amount']
        odds = (ratio**-1+1, ratio+1)
        wr = (odds[1]/np.sum(odds)*100., odds[0]/np.sum(odds)*100.)
        det = json.loads(scraper.get('https://thunderpick.com/api/matches/'+str(v['id'])).content)['data']
        print('Date: %s  |  Event: %s  | (BO%s) %s vs. %s  |  (%.1f:%.1f) | Total Coins: %i' % 
              (v['startTime'][:10], v['championship'], det['bestOfMaps'], v['matchBet']['buckets'][0]['label'], 
               v['matchBet']['buckets'][1]['label'], wr[0], wr[1], v['matchBet']['amount']))
        bet_games.append({'1': v['matchBet']['buckets'][0]['label'], '2': v['matchBet']['buckets'][1]['label'], 'bo': det['bestOfMaps'], 'o1': odds[0], 'o2': odds[1], 'wr': wr[0]})
bet_games = pd.DataFrame(bet_games)

TEAM_SET = 'all_time'

teams = np.load('wl_model/saved_model/'+TEAM_SET+'/teams.npy')
maps = np.load('wl_model/saved_model/'+TEAM_SET+'/maps.npy')
periods = np.load('wl_model/saved_model/'+TEAM_SET+'/periods.npy')
h_teams = pd.read_csv('wl_model/hltv_csv/teams_w_ranking.csv').set_index('ID').loc[teams]
h_teams = fix_teams(h_teams)
h_teams.Name = h_teams.Name.str.lower()
h_teams_filt = h_teams.loc[teams]

rating_model = prep_pymc_time_model(len(teams), len(maps), len(periods))
trace = pm.backends.text.load('wl_model/saved_model/'+TEAM_SET+'/trace', model=rating_model)

h_bp = pd.read_csv('wl_model/hltv_csv/picksAndBans.csv').set_index('Match ID')
h_matches = pd.read_csv('wl_model/hltv_csv/matchResults.csv').set_index('Match ID')
h_matches['Date'] = pd.to_datetime(h_matches['Date'])
h_matches = h_matches[h_matches['Date'] >= dt.datetime(2017,1,1)]
h_bp = h_bp.join(h_matches[['Date']], how='left')
h_bp['Date'] = pd.to_datetime(h_bp['Date'])
h_bp = h_bp[h_bp['Date'] >= dt.datetime(2017,1,1)]

def model_mp(train, t1, t2):
    tab = train[train['Team'].isin([t1, t2])].groupby(['Team', ' Pick Type', 'Map'])['Date'].count().unstack([' Pick Type', 'Team']).fillna(0)
    return (tab/tab.sum(axis=0)).mean(level=0,axis=1)# get average

def model_played(train, t1, t2):
    a = train[train['Team 1 ID'].isin([t1,t2])].groupby(['Team 1 ID', 'Map'])['Date'].count()
    b = train[train['Team 2 ID'].isin([t1,t2])].groupby(['Team 2 ID', 'Map'])['Date'].count()
    c = pd.DataFrame([a,b], index=['a','b']).T.fillna(0)
    c = (c['a']+c['b']).unstack(level=0).fillna(0)
    return (c/c.sum()).mean(axis=1)

def predict_map(func, data, t1, t2):
    res = func(data, t1, t2)
    return res.loc[res.index != 'Default'].sort_values(ascending=False)

money = 4500.
bet_games['1'] = bet_games['1'].str.lower().replace({'ex-denial': 'denial', 'red reserve': 'japaleno'})
bet_games['2'] = bet_games['2'].str.lower().replace({'ex-denial': 'denial', 'red reserve': 'japaleno'})
matches = bet_games[bet_games['1'].isin(h_teams_filt.Name.values) & bet_games['2'].isin(h_teams_filt.Name.values)].drop_duplicates()
def sig(x):
    return 1 / (1 + np.exp(-x))
def abs_norm_interval(start,end,loc,scale):
    return (norm.cdf(end,loc,scale) - norm.cdf(start,loc,scale)) + (norm.cdf(-1*start,loc,scale) - norm.cdf(-1*end,loc,scale))


per_ind = np.where(periods == pd.Period(dt.datetime.today(), 'M')-1)[0][0]
t_rating = trace['rating_%s'%per_ind]
t_map_rating = trace['rating_%s | map' %per_ind]
t_alpha = 0.3485
for i,v in matches.iterrows():
    t1_id = h_teams_filt[h_teams_filt.Name == v['1']].index[0]; t1_ind = np.where(teams == t1_id)[0][0];
    t2_id = h_teams_filt[h_teams_filt.Name == v['2']].index[0]; t2_ind = np.where(teams == t2_id)[0][0];
    trace_1 = t_rating[:,t1_ind]; trace_2 = t_rating[:,t2_ind]
    mr_1 = trace_1.mean(); mr_2 = trace_2.mean();
    diff = trace_1-trace_2
    p_wl = 0.5*np.tanh(diff)+0.5
    wr_25 = np.percentile(p_wl, 25); wr_75 = np.percentile(p_wl, 75)
    kelly_pct_1 = ((v['o1']*np.percentile(p_wl, 47)-(1.-np.percentile(p_wl, 47)))/v['o1'])*0.10
    kelly_pct_2 = ((v['o2']*(1.-np.percentile(p_wl, 47))-(np.percentile(p_wl, 47)))/v['o2'])*0.10
    print('%s (%.3f) vs %s (%.3f) - P:%.2f%% - %.2f%% - %.2f%%  -  K: %.1f%% (%i) - %.1f%% (%i)' % 
          (v['1'], mr_1, v['2'], mr_2,  wr_25*100, v['wr'], wr_75*100, kelly_pct_1*100., 
           kelly_pct_1*money, kelly_pct_2*100., kelly_pct_2*money))

PRINT_RD_DIFF = False
for i,v in matches.iterrows():
    t1_id = h_teams_filt[h_teams_filt.Name == v['1']].index[0]; t1_ind = np.where(teams == t1_id)[0][0];
    t2_id = h_teams_filt[h_teams_filt.Name == v['2']].index[0]; t2_ind = np.where(teams == t2_id)[0][0];
    print('---------- %s vs %s ---------------------------------' % (v['1'], v['2']))
    pred_maps = predict_map(model_played, h_matches, t1_id, t2_id)
    pred_maps = pred_maps/pred_maps.sum()
    for m,s in pred_maps.iteritems():
        m_ind = np.where(maps == m)[0][0]
        trace_1 = t_map_rating[:,m_ind,t1_ind]; trace_2 = t_map_rating[:,m_ind,t2_ind]
        mr_1 = trace_1.mean(); mr_2 = trace_2.mean();
        diff = trace_1-trace_2
        p_wl = sig(diff)
        wr_25 = np.percentile(p_wl, 25); wr_75 = np.percentile(p_wl, 75)
        kappa = 32*sig(t_alpha*diff)-16.
        kelly_pct_1 = ((v['o1']*np.percentile(p_wl, 45)-(1.-np.percentile(p_wl, 45)))/v['o1'])*0.1
        kelly_pct_2 = ((v['o2']*(1.-np.percentile(p_wl, 45))-(np.percentile(p_wl, 45)))/v['o2'])*0.1
        print('    Map: %s (%.2f)  -  %s (%.3f) vs %s (%.3f) - P:%.2f%% - %.2f%% - %.2f%%  -  K: %.1f%% (%i) - %.1f%% (%i)' % 
             (m, s*100., v['1'], mr_1, v['2'], mr_2,  wr_25*100, v['wr'], wr_75*100, kelly_pct_1*100., 
               kelly_pct_1*money, kelly_pct_2*100., kelly_pct_2*money))
        
        if(PRINT_RD_DIFF):
            p_sc = [abs_norm_interval(x[0],x[1],kappa,trace['sigma'][:,m_ind]) for x in [[1.5,3.5],[3.5,5.5],[5.5,7.5],[7.5,9.5],[9.5,16]]]
            for i,sd in enumerate(['2 - 3 Rounds', '4 - 5 rounds', '6 - 7 rounds', '8 - 9 rounds', '10 rounds or more']):
                sc_25 = np.percentile(p_sc[i], 25); sc_75 = np.percentile(p_sc[i], 75)
                print('      %s : %.2f%% - %.2f%%' % (sd, sc_25*100, sc_75*100))

np.inner(np.ones((8,25)), np.ones((8,25))).shape

np.ones((8,25)).flatten().shape

np.zeros((8000, 200))

plt.ylim(0,1.2)
sns.kdeplot(trace_1, shade=True, alpha=0.65, legend=True, label=v['1'])
sns.kdeplot(trace_2, shade=True, alpha=0.65, legend=True, label=v['2'])

h_bp.groupby('Match ID').first().count()

h_bp




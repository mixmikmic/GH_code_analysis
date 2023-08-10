import sys
PROJECT_ROOT="C:\\Users\\Roger\\PycharmProjects\\sports_data"
sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
from sites.common import sql_utils
from matplotlib import pyplot as plt
pd.options.display.max_columns=999
get_ipython().magic('matplotlib inline')

matchups=pd.read_sql("matchups",sql_utils.getConnection("nba"))
matchups

FILE_ROOT=PROJECT_ROOT+"\\sites\\fivethirtyeight\\data\\nbaallelo.csv"
full_df=pd.read_csv(FILE_ROOT)
games=full_df[full_df['game_location']=='H'] #remove duplicated rows work with our elo implementation
games.head(1)

games['SEASON']=games['year_id'].apply(lambda x: "%s-%s"%(x-1,x))

import scipy.stats as st
from collections import defaultdict
import numpy as np

def silverK(MOV, elo_diff):
    K_0=20
    if MOV>0:
        multiplier=(MOV+3)**(0.8)/(7.5+0.006*(elo_diff))
    else:
        multiplier=(-MOV+3)**(0.8)/(7.5+0.006*(-elo_diff))
    return K_0*multiplier,K_0*multiplier

def silverS(home_score, away_score):
    S_home,S_away=0,0
    if home_score>away_score:
        S_home=1
    elif away_score>home_score:
        S_away=1
    else:
        S_home,S_away=.5,.5
    return S_home,S_away

def silver_elo_update(home_score, away_score, home_rating, away_rating):
    HOME_AD=100.
    home_rating+=HOME_AD
    E_home = elo_prediction(home_rating,away_rating)
    E_away=1-E_home
    elo_diff=home_rating-away_rating
    MOV=home_score-away_score
    
    S_home,S_away = silverS(home_score,away_score)
    if S_home>0:
        K_home,K_away =  silverK(MOV,elo_diff)
    else:
        K_home,K_away =  silverK(MOV,elo_diff)
        
    return K_home*(S_home-E_home),K_away*(S_away-E_away)

def elo_prediction(home_rating,away_rating):
    E_home = 1./(1 + 10 ** ((away_rating - home_rating) / (400.)))
    return E_home

def score_prediction(home_rating,away_rating):
    return (home_rating-away_rating)/28.

class HeadToHeadModel(object):
    def __init__(self, events, update_function, prediction_function=None):
        self.update_function=update_function
        self.events=events
        self.ratings={}
        self.prediction_function = prediction_function
        self.predictions = []
        self.curr_season=defaultdict(lambda: self.events[0][1]['year_id'])

    def train(self):
        for idx, event in self.events:
            new_year=event['year_id']
            label_i=event['fran_id']
            label_j=event['opp_fran']
            if self.ratings.get(label_i,False)==False:
                self.ratings[label_i]=elo_lookup(label_i,event['gameorder'])
            if self.ratings.get(label_j,False)==False:
                self.ratings[label_j]=elo_lookup(label_j,event['gameorder'])
                
            if self.curr_season[label_i]!=new_year:
                self.curr_season[label_i]=new_year
                self.ratings[label_i]=self.ratings[label_i]*.75+1505.*.25
            elif self.curr_season[label_j]!=new_year:
                self.curr_season[label_j]=new_year
                self.ratings[label_j]=self.ratings[label_j]*.75+1505.*.25
            #todo change below to just use event
            update=self.update_function(event['pts'],event['opp_pts'], self.ratings[label_i], self.ratings[label_j])
            self.ratings[label_i]+=update[0]
            self.ratings[label_j]+=update[1]
            

    def power_rankings(self):
        from operator import itemgetter
        power_rankings = sorted(self.ratings.items(), key=itemgetter(1), reverse=True)
        power = []
        for i, x in enumerate(power_rankings):
            power.append((i + 1, x))
        return power
    
STARTING_LOC=0
def elo_lookup(fran_id,gameorder):
    return full_df[(full_df['fran_id']==fran_id)&(full_df['gameorder']>=gameorder)]['elo_i'].iloc[0]
m=HeadToHeadModel(list(games[games['gameorder']>STARTING_LOC].iterrows()), silver_elo_update, elo_prediction)
m.train()
m.power_rankings()

SSE=0
my_scores=[]
nate_scores=[]
for team,rating in m.ratings.items():
    nate_final_rating=full_df[full_df['fran_id']==team]['elo_n'].iloc[-1]
    my_scores.append(rating)
    nate_scores.append(nate_final_rating)
plt.scatter(my_scores, nate_scores)
plt.ylabel("Nate Silver's Final Elo Ratings")
plt.xlabel("My Final Elo Ratings")
plt.title("Comparison of Nate Silver's Elo and my Implementation")

import statsmodels.api as sm

X=my_scores
X=sm.add_constant(X)
Y=nate_scores
model=sm.OLS(Y,X)
results=model.fit()
results.summary()

class HeadToHeadModel(object):
    def __init__(self, events, update_function, prediction_function=None):
        self.update_function=update_function
        self.events=events
        self.ratings={}
        self.prediction_function = prediction_function
        self.predictions = []
        self.curr_season=defaultdict(lambda: self.events[0][1]['year_id'])

    def train(self):
        for idx, event in self.events:
            new_year=event['year_id']
            label_i=event['fran_id']
            label_j=event['opp_fran']
            if self.ratings.get(label_i,False)==False:
                self.ratings[label_i]=elo_lookup(label_i,event['gameorder'])
            if self.ratings.get(label_j,False)==False:
                self.ratings[label_j]=elo_lookup(label_j,event['gameorder'])
            
            if self.curr_season[label_i]!=new_year:
                self.curr_season[label_i]=new_year
                self.ratings[label_i]=self.ratings[label_i]*.75+1505.*.25
            elif self.curr_season[label_j]!=new_year:
                self.curr_season[label_j]=new_year
                self.ratings[label_j]=self.ratings[label_j]*.75+1505.*.25
            self.predictions.append(elo_prediction(self.ratings[label_i]+100, self.ratings[label_j]))
            #todo change below to just use event
            update=self.update_function(event['pts'],event['opp_pts'], self.ratings[label_i], self.ratings[label_j])
            self.ratings[label_i]+=update[0]
            self.ratings[label_j]+=update[1]
            

    def power_rankings(self):
        from operator import itemgetter
        power_rankings = sorted(self.ratings.items(), key=itemgetter(1), reverse=True)
        power = []
        for i, x in enumerate(power_rankings):
            power.append((i + 1, x))
        return power
STARTING_LOC=0
m=HeadToHeadModel(list(games[games['gameorder']>STARTING_LOC].iterrows()), silver_elo_update, elo_prediction)
m.train()
m.power_rankings()

games['prediction']=m.predictions
games['predictedWinner']=games['prediction'].apply(lambda x: 1 if x>=.5 else 0)
games['winner']=games.apply(lambda x: x['pts']>=x['opp_pts'],axis=1)

from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(games['winner'],games['predictedWinner'])
conf_matrix

success_rate=np.trace(conf_matrix)/(np.sum(conf_matrix))
success_rate

class HeadToHeadModel(object):
    def __init__(self, events, update_function, prediction_function=None):
        self.update_function=update_function
        self.events=events
        self.ratings={}
        self.prediction_function = prediction_function
        self.predictions = []
        self.curr_season=defaultdict(lambda: self.events[0][1]['year_id'])

    def train(self):
        for idx, event in self.events:
            new_year=event['year_id']
            label_i=event['fran_id']
            label_j=event['opp_fran']
            if self.ratings.get(label_i,False)==False:
                self.ratings[label_i]=elo_lookup(label_i,event['gameorder'])
            if self.ratings.get(label_j,False)==False:
                self.ratings[label_j]=elo_lookup(label_j,event['gameorder'])
                
            if self.curr_season[label_i]!=new_year:
                self.curr_season[label_i]=new_year
                self.ratings[label_i]=self.ratings[label_i]*.75+1505.*.25
            elif self.curr_season[label_j]!=new_year:
                self.curr_season[label_j]=new_year
                self.ratings[label_j]=self.ratings[label_j]*.75+1505.*.25
            #todo change below to just use event
            self.predictions.append(score_prediction(self.ratings[label_i]+100, self.ratings[label_j]))
            update=self.update_function(event['pts'],event['opp_pts'], self.ratings[label_i], self.ratings[label_j])
            self.ratings[label_i]+=update[0]
            self.ratings[label_j]+=update[1]

    def power_rankings(self):
        from operator import itemgetter
        power_rankings = sorted(self.ratings.items(), key=itemgetter(1), reverse=True)
        power = []
        for i, x in enumerate(power_rankings):
            power.append((i + 1, x))
        return power

matchups['game_id']=matchups.apply(lambda x: x['data-game-date'].split(" ")[0].replace("-","")+"0"+x['home_name'],axis=1)
games_w_odds=matchups.merge(games)
games_w_odds.tail(1)

m=HeadToHeadModel(list(games_w_odds.iterrows()), silver_elo_update, elo_prediction)
m.train()
m.power_rankings()

games_w_odds['predictedHomeMOV']=m.predictions
games_w_odds['homeMOV']=games_w_odds['pts']-games_w_odds['opp_pts']
games_w_odds['homeCover']=(games_w_odds['homeMOV']+games_w_odds['data-game-odd'])>0
games_w_odds.head(1)

len(games_w_odds)

games_w_odds['SEASON'].unique()

def bettingFunction(row):
    '''
    if algo favors team more than vegas predicted score bet for, True. Else bet for the team True
    '''
    return (row['predictedHomeMOV']+row['data-game-odd'])>0
games_w_odds['bets']=games_w_odds.apply(bettingFunction,axis=1)
conf_matrix=confusion_matrix(games_w_odds['homeCover'],games_w_odds['bets'])
pd.DataFrame(conf_matrix, columns=["Bet on Away","Bet on Home"], index=["Away Covers","Home Covers"])

print(pd.DataFrame(conf_matrix, columns=["Bet on Away","Bet on Home"], index=["Away Covers","Home Covers"]).to_html())

success_rate=np.trace(conf_matrix)/(np.sum(conf_matrix))
success_rate

757/(669+757)

bets=np.random.binomial(1,.5,len(games_w_odds))
truths=np.random.binomial(1,.5,len(games_w_odds))
conf_matrix=confusion_matrix(truths,bets)
pd.DataFrame(conf_matrix, columns=["Bet on Away","Bet on Home"], index=["Away Covers","Home Covers"])

pd.DataFrame(conf_matrix, columns=["Bet on Away","Bet on Home"], index=["Away Covers","Home Covers"]).to_html()




import pandas as pd
np=pd.np
from sdd_api.api import Api
from credentials import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns=None
api = Api(username=username, password=password, client_id=client_id, client_secret=client_secret)

matchups=api.get_dataframe("matchups",season_start=2000)
matchups

matchups=matchups[(matchups['game_type'].isin(['Regular','Playoffs']))]
#drop any unscored/unplayed games
matchups=matchups.dropna(subset=["home_pts","away_pts"])
matchups.tail(2)

matchups.describe()

matchups['is_neutral']=matchups['game_location'].apply(lambda x: True if x=="N" else False)

from collections import defaultdict
def silverK(MOV, elo_diff):
    K_0=20
    multiplier=np.log(abs(MOV)+1)*(2.2/((elo_diff)*.001+2.2))
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

def silver_elo_update(home_score, away_score, home_rating, away_rating, isNeutral):
    HOME_AD=65.
    if not isNeutral:
        pass#home_rating+=HOME_AD
    E_home = elo_prediction(home_rating,away_rating)
    E_away=1-E_home
    elo_diff=home_rating-away_rating
    MOV=home_score-away_score
    
    S_home,S_away = silverS(home_score,away_score)
    if home_score>=away_score:
        elo_winner=home_rating
        elo_loser=away_rating
    else:
        elo_loser=home_rating
        elo_winner=away_rating

    K_home,K_away =  silverK(MOV,elo_winner-elo_loser)
        
    return K_home*(S_home-E_home),K_away*(S_away-E_away)

def elo_prediction(home_rating,away_rating):
    E_home = 1./(1 + 10 ** ((away_rating - home_rating) / (400.)))
    return E_home

def score_prediction(home_rating,away_rating):
    return (home_rating-away_rating)/25.
class HeadToHeadModel(object):
    def __init__(self, events, update_function, prediction_function=None):
        self.update_function=update_function
        self.events=events
        self.ratings=defaultdict(lambda: 1505)
        self.prediction_function = prediction_function
        self.predictions = []
        self.curr_season=defaultdict(lambda: self.events[0][1]['season'])

    def compute_elo_ratings(self):
        for idx, event in self.events:
            new_year=event['season']
            label_i=event['home_name']
            label_j=event['away_name']
            
            if self.curr_season[label_i]!=new_year:
                self.curr_season[label_i]=new_year
                self.ratings[label_i]=self.ratings[label_i]*1/3+1505.*.75
            elif self.curr_season[label_j]!=new_year:
                self.curr_season[label_j]=new_year
                self.ratings[label_j]=self.ratings[label_j]*.75+1505.*.25
            #todo change below to just use event
            update=self.update_function(event['home_pts'],event['away_pts'], self.ratings[label_i], self.ratings[label_j], event['is_neutral'])
            self.ratings[label_i]+=update[0]
            self.ratings[label_j]+=update[1]
            

    def power_rankings(self):
        from operator import itemgetter
        #sort dictionary by value to get ascending list of teams
        power_rankings = sorted(self.ratings.items(), key=itemgetter(1), reverse=True)
        power = []
        #Make the 0-th team 1st
        for i, x in enumerate(power_rankings):
            power.append((i + 1, x))
        return power
m=HeadToHeadModel(list(matchups.iterrows()), silver_elo_update, elo_prediction)
m.compute_elo_ratings()
m.power_rankings()

#Let's redo our elo model with the addition of elo ranking output
class HeadToHeadModel(object):
    def __init__(self, events, update_function, prediction_function=None):
        self.update_function=update_function
        self.events=events
        self.ratings=defaultdict(lambda: 1505)
        self.prediction_function = prediction_function
        self.predictions = []
        self.curr_season=defaultdict(lambda: self.events[0][1]['season'])
        self.elos=[]
    def compute_elo_ratings(self):
        for idx, event in self.events:
            new_year=event['season']
            label_i=event['home_name']
            label_j=event['away_name']
            
            if self.curr_season[label_i]!=new_year:
                self.curr_season[label_i]=new_year
                self.ratings[label_i]=self.ratings[label_i]*1/3+1505.*2/3
            elif self.curr_season[label_j]!=new_year:
                self.curr_season[label_j]=new_year
                self.ratings[label_j]=self.ratings[label_j]*.75+1505.*.25
            #todo change below to just use event
            update=self.update_function(event['home_pts'],event['away_pts'], self.ratings[label_i], self.ratings[label_j], event['is_neutral'])
            self.elos.append({
                "home_elo":self.ratings[label_i],
                "away_elo":self.ratings[label_j],
                "index": idx,
                
                             })
            self.ratings[label_i]+=update[0]
            self.ratings[label_j]+=update[1]
            

            

    def power_rankings(self):
        from operator import itemgetter
        #sort dictionary by value to get ascending list of teams
        power_rankings = sorted(self.ratings.items(), key=itemgetter(1), reverse=True)
        power = []
        #Make the 0-th team 1st
        for i, x in enumerate(power_rankings):
            power.append((i + 1, x))
        return power
m=HeadToHeadModel(list(matchups.iterrows()), silver_elo_update, elo_prediction)
m.compute_elo_ratings()
m.power_rankings()

elo=pd.DataFrame(m.elos).set_index("index").join(matchups)
elo.head(2)

elo.describe()

def predict_home_margin(row):
    #how many points the home team is expected to win bye
    return score_prediction(row['home_elo']+65, row['away_elo'])
elo['predicted_home_margin']=elo.apply(predict_home_margin,axis=1)

elo.head(1)

#1 is home winning
elo['predicted_winner']=elo.apply(lambda row: 1 if row['predicted_home_margin']>=0 else 0, axis=1)

elo['vegas_predicted_winner']=elo.apply(lambda row: 0 if row['home_line']>0 else 1, axis=1)#line assigns negative value
elo['winner']=elo.apply(lambda row: 1 if row['home_pts']>row['away_pts'] else 0, axis=1)#does not account for ties

from sklearn.metrics import confusion_matrix
def print_confusion_matrix(y_true, y_pred):
    conf_matrix=confusion_matrix(y_true, y_pred)
    success_rate=np.trace(conf_matrix)/np.sum(conf_matrix)
    print(success_rate)
    print(conf_matrix)
test_period=elo[(elo['week_num'].between(4,15,inclusive=True))&(elo['game_type']=='Regular')&(elo['season'].between(2012,2016))]

y_true='winner'
y_pred='predicted_winner'
print_confusion_matrix(test_period['winner'],test_period['predicted_winner'])

y_true='winner'
y_pred='vegas_predicted_winner'
print_confusion_matrix(test_period['winner'],test_period[y_pred])

elo.head(1)

elo['home_margin']=elo.apply(lambda x: x['home_pts']-x['away_pts'], axis=1)
elo['home_bet']=elo.apply(lambda x: (x['predicted_home_margin']+x['home_line'])<0,axis=1)
elo['home_covers']=elo.apply(lambda x: (x['home_margin']+x['home_line'])>0,axis=1)
test_period=elo[(elo['week_num'].between(4,15,inclusive=True))&(elo['game_type']=='Regular')&(elo['season'].between(2012,2016))]
y_true='home_covers'
y_pred='home_bet'
print_confusion_matrix(test_period[y_true],test_period[y_pred])

elo['predicted_winner_right']=elo.apply(lambda x: 1 if x['winner']==x['predicted_winner'] else 0, axis=1)
elo['vegas_favored_wins']=elo.apply(lambda x: 1 if x['winner']==x['vegas_predicted_winner'] else 0, axis=1)
elo['ats_right']=elo.apply(lambda x: 1 if x['home_covers']==x['home_bet'] else 0, axis=1)
test_period=elo[(elo['week_num'].between(4,15,inclusive=True))&(elo['game_type']=='Regular')&(elo['season'].between(2012,2016))]

test_period[['predicted_winner_right','vegas_favored_wins','ats_right','season']].groupby("season").agg([np.mean, np.var])

# for all years in our sample we do about just as good as vegas but no better
test_period[['predicted_winner_right','vegas_favored_wins','ats_right']].groupby(lambda x: 0).agg([np.mean, np.var])

team_season_log = api.get_dataframe("team_season_log")
team_season_log.head()

team_game_logs = api.get_dataframe("team_game_logs")
team_game_logs.head()




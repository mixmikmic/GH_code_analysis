get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3

from collections import Counter, defaultdict
from bs4 import BeautifulSoup

import numpy as np
from numpy import random

#load data

with sqlite3.connect('../data/raw/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    matches = pd.read_sql_query("SELECT * from Match", con, parse_dates=['date'])
    leagues = pd.read_sql_query("SELECT * from League", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
    players = pd.read_sql_query("SELECT * from Player", con)
    players_stats = pd.read_sql_query("SELECT * from Player_Stats", con)

selected_countries = ['England']
countries = countries[countries.name.isin(selected_countries)]
leagues = countries.merge(leagues, on='id', suffixes=('', '_y'))
matches = matches[matches.league_id.isin(leagues.id)]
matches.head()

def get_team_long_name(team_id):
    return teams.loc[teams.team_api_id == team_id, 'team_long_name'].values[0]

def get_team_short_name(team_id):
    return teams.loc[teams.team_api_id == team_id, 'team_short_name'].values[0]

def get_formation(match_api_id, team_type):
    formation_dict = Counter([matches.loc[matches.match_api_id == match_api_id]['%s_player_Y%d'%(team_type, i)].values[0] for i in range(1, 12)])
    sorted_keys = sorted(formation_dict)

    formation = ''
    for key in sorted_keys[1:-1]:
        y = formation_dict[key]
        formation += '%d-' % y
    formation += '%d' % formation_dict[sorted_keys[-1]]
    
    return formation

def decompose_season(matches):
    """
    Convert to ['2008/2009', '2009/2010'] into ([2008, 2009], [2009, 2010])
    """
    season_start = []
    season_end = []
    
    for season in matches.season:
        season_split = season.split('/')
        season_start.append(int(season_split[0]))
        season_end.append(int(season_split[1]))
    
    return (season_start, season_end)

matches['home_team_long_name'] = matches.home_team_api_id.map(get_team_long_name)
matches['home_team_short_name'] = matches.home_team_api_id.map(get_team_short_name)

matches['away_team_long_name'] = matches.away_team_api_id.map(get_team_long_name)
matches['away_team_short_name'] = matches.away_team_api_id.map(get_team_short_name)

matches['home_team_formation'] = matches.match_api_id.apply(get_formation, args=('home', ))
matches['away_team_formation'] = matches.match_api_id.apply(get_formation, args=('away',))

matches['season_start'], matches['season_end'] = decompose_season(matches)

def calculate_result(match_df):
    """
    Calculate results based on number of goals scored by home and away teams.
    """
    home_team_goals = match_df['home_team_goal']
    away_team_goals = match_df['away_team_goal']
    
    if home_team_goals > away_team_goals:
        return 'home'
    elif away_team_goals > home_team_goals:
        return 'away'
    else:
        return 'draw'

matches['result'] = matches.apply(calculate_result, axis=1)

def team_performance(matches, home_team, away_team, season_start):
    """
    Returns a tuple of performance of both the teams.
    (
        {'wins': 3, 'draw': 2, 'loss': 5}, # home team
        {'wins': 4, 'draw': 1, 'loss': 3}  # away team
    )
    """
    mask = (matches.season_start < season_start) & (matches.home_team_short_name == home_team) & (matches.away_team_short_name == away_team)
    matches_sub_df = matches[mask]
    
    wins = 0
    draws = 0
    losses = 0
    
    for key, result in matches_sub_df.result.items():
        if result == 'home':
            wins += 1
        elif result == 'draw':
            draws += 1
        else:
            losses += 1
    
    return (wins, draws, losses)

def calculate_past_performance(matches):
    prev_wins = []
    prev_draws = []
    prev_losses = []
    
    for home, away, season_start, result in zip(matches.home_team_short_name, matches.away_team_short_name, matches.season_start, matches.result):
        win, draw, loss = team_performance(matches, home, away, season_start)
        prev_wins.append(win)
        prev_draws.append(draw)
        prev_losses.append(loss)
    
    return (prev_wins, prev_draws, prev_losses)

prev_wins, prev_draws, prev_losses = calculate_past_performance(matches)

matches['prev_wins'] = prev_wins
matches['prev_draws'] = prev_draws
matches['prev_losses'] = prev_losses

def calculate_crosses_count(crosses):
    parsed_xml = BeautifulSoup(crosses, 'xml')
    num_crosses = defaultdict(int)
    
    for x in parsed_xml.find_all('value'):
        if x.crosses:
            if x.team:
                num_crosses[x.team.text] += 1
    
    return num_crosses

crosses_info = matches.cross.map(calculate_crosses_count)

def get_crosses_list(crosses_info, matches):
    home_team_crosses = []
    away_team_crosses = []
    counter = 0
    
    for team_id in matches.home_team_api_id.values:
        home_team_crosses.append(crosses_info.iloc[counter][str(team_id)])
        counter += 1
        
    counter = 0
    for team_id in matches.away_team_api_id.values:
        away_team_crosses.append(crosses_info.iloc[counter][str(team_id)])
        counter += 1
        
    return (home_team_crosses, away_team_crosses)

matches['num_home_team_crosses'], matches['num_away_team_crosses'] = get_crosses_list(crosses_info, matches)

features = ['date', 'home_team_formation', 'home_team_short_name', 'away_team_formation',             'away_team_short_name',  'stage', 'num_home_team_crosses', 'num_away_team_crosses',            'B365H','B365D','B365A','result', 'prev_wins', 'prev_daws', 'prev_losses']

from sklearn.preprocessing import LabelEncoder

for feature in ['home_team_formation', 'home_team_short_name', 'away_team_formation', 'away_team_short_name', 'result']:
    lbl = LabelEncoder()
    matches[feature] = lbl.fit_transform(matches[feature])

matches['weekday'] = matches.date.dt.weekday
matches['day'] = matches.date.dt.day

matches['B365H'] = matches.B365H.fillna(-999)
matches['B365A'] = matches.B365A.fillna(-999)
matches['B365D'] = matches.B365D.fillna(-999)

X = matches[['home_team_formation', 'home_team_short_name', 'away_team_formation',             'away_team_short_name',  'stage', 'num_home_team_crosses', 'num_away_team_crosses',             'B365H','B365D','B365A', 'weekday', 'day', 'prev_wins', 'prev_draws', 'prev_losses']]
y = matches.result

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=44)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

est = LogisticRegression(C=1., penalty='l2')
est.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

print('Accuracy score on test set %f '%(accuracy_score(y_test, est.predict(X_test))))




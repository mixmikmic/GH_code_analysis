get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import os, sys

import matplotlib.pyplot as plt
import seaborn as sns

import sqlite3

from collections import Counter, defaultdict
from bs4 import BeautifulSoup

from numpy import random

import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)

basepath = os.path.expanduser('~/Desktop/src/paul-the-octopus/')
sys.path.append(os.path.join(basepath, 'src'))

with sqlite3.connect(os.path.join(basepath,'data/raw/database.sqlite')) as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    matches = pd.read_sql_query("SELECT * from Match", con, parse_dates=['date'])
    leagues = pd.read_sql_query("SELECT * from League", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
    players = pd.read_sql_query("SELECT * from Player", con)
    players_stats = pd.read_sql_query("SELECT * from Player_Stats", con)
    

# consider only english premier league for now.
english_league_id = countries.loc[countries.name == 'England', 'id'].values[0]

epl = matches.loc[matches.country_id == english_league_id]

list(epl.columns[77:85])

get_match_statistics(epl.iloc[:1], 'card')

BeautifulSoup(epl.possession.iloc[1])

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

def get_match_statistics(match_df, feature):
    home_team_id = match_df['home_team_api_id']
    away_team_id = match_df['away_team_api_id']
        
    statistics = match_df[feature]

    home_team_counts = np.zeros(len(statistics))
    away_team_counts = np.zeros(len(statistics))

    for i, stat in enumerate(statistics):
        if stat is not None:
            parsed_xml = BeautifulSoup(stat, 'xml')

            for v in parsed_xml.find_all('value'):
                if v.find('team') is not None:
                    team_id = v.find('team').text
                    
                    if team_id == str(home_team_id.iloc[i]):
                        home_team_counts[i] += 1
                    else:
                        away_team_counts[i] += 1

    return (home_team_counts, away_team_counts)

epl['home_team_formation'] = epl.match_api_id.apply(get_formation, args=('home', ))
epl['away_team_formation'] = epl.match_api_id.apply(get_formation, args=('away',))

epl['season_start'], epl['season_end'] = decompose_season(epl)
epl['result'] = epl.apply(calculate_result, axis=1)

epl['home_team_cards'], epl['away_team_cards'] = get_match_statistics(epl, 'card')
epl['home_team_shots_on_target'], epl['away_team_shots_on_target'] = get_match_statistics(epl, 'shoton')
epl['home_team_shots_off_target'], epl['away_team_shots_off_target'] = get_match_statistics(epl, 'shotoff')
epl['home_team_crosses'], epl['away_team_crosses'] = get_match_statistics(epl, 'cross')
epl['home_team_corners'], epl['away_team_corners'] = get_match_statistics(epl, 'corner')
epl['home_team_fouls'], epl['away_team_fouls'] = get_match_statistics(epl, 'foulcommit')

features = ['home_team_api_id', 'away_team_api_id','season', 'stage', 'date',             'home_team_goal', 'away_team_goal', 'home_team_formation', 'away_team_formation',            'season_start', 'season_end', 'result', 'home_team_cards', 'away_team_cards',            'home_team_shots_on_target', 'away_team_shots_on_target', 'home_team_shots_off_target',            'away_team_shots_off_target', 'home_team_corners', 'away_team_corners', 'home_team_crosses',            'away_team_crosses', 'home_team_fouls', 'away_team_fouls'
           ]
            
processed = epl[features]

processed.head()


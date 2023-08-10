import re
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import IFrame
import matplotlib.font_manager as fm

plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')
pd.options.display.max_columns=25

shot_url = 'http://stats.nba.com/stats/leaguedashplayerptshot?LastNGames=0'    '&LeagueID=00&Month=0&OpponentTeamID=0&PORound=0&PaceAdjust=N&PerMode=Totals'    '&Period=0&PlusMinus=N&Rank=N&Season=2014-15&SeasonType=Regular+Season&TeamID=0'
resp = requests.get(shot_url)
#print resp.text
resp.json().keys()

player_info_df = pd.DataFrame(resp.json()['resultSets'][0]['rowSet'],columns=resp.json()['resultSets'][0]['headers'])
player_info_df.head(5)

print player_shot_df.columns
fields = [u'PERIOD',u'MINUTES_REMAINING', u'EVENT_TYPE',
       u'SHOT_TYPE', u'SHOT_ZONE_BASIC', u'SHOT_ZONE_AREA',
       u'SHOT_ZONE_RANGE', u'SHOT_DISTANCE']
for f in fields:
    print f, player_shot_df[f].value_counts()
player_shot_df[player_shot_df['PERIOD'].isin([1,2,3,4])]['PERIOD'].value_counts().plot(kind='pie',figsize=(8,8),autopct='%.2f')

reb_url = "http://stats.nba.com/stats/playerdashptreboundlogs?"

reb_param = {'PlayerID' : 203083,
                           'LeagueID': u'00',
                           'Season' :  u'2015-16',
                           'SeasonType' : u'Regular Season',
                           'TeamID' : 0,
                           'Outcome' : u'',
                           'Location' : u'',
                           'Month' : 0,
                           'SeasonSegment' : u'',
                           'DateFrom' :  u'',
                           'DateTo' : u'',
                           'OpponentTeamID' : 0,
                           'VsConference' : u'',
                           'VsDivision' : u'',
                           'GameSegment' : u'',
                           'Period' :  0,
                           'LastNGames' : 0
                           }
req = requests.get(reb_url, params=reb_param)
reb_df = pd.DataFrame(req.json()['resultSets'][0]['rowSet'],columns=req.json()['resultSets'][0]['headers'])
reb_df.head(5)

com_url = "http://stats.nba.com/stats/commonallplayers?"
com_param = {'IsOnlyCurrentSeason':"1",
                      'LeagueID': u'00',
                      'Season': u'2015-16'}
com_req = requests.get(com_url, params=com_param)
#req2.text
com_df = pd.DataFrame(com_req.json()['resultSets'][0]['rowSet'],columns=com_req.json()['resultSets'][0]['headers'])
com_df.head(10)

sl_url = "http://stats.nba.com/stats/playerdashptshotlog?"
sl_param = {'PlayerID' : 201939,   #cannot be 0
           'LeagueID': u'00',
           'Season' :  u'2014-15',
           'SeasonType' : u'Regular Season',
           'TeamID' : 0,
           'Outcome' : u'',
           'Location' : u'',
           'Month' : 0,
           'SeasonSegment' : u'',
           'DateFrom' :  u'',
           'DateTo' : u'',
           'OpponentTeamID' : 0,
           'VsConference' : u'',
           'VsDivision' : u'',
           'GameSegment' : u'',
           'Period' :  0,
           'LastNGames' : 0
           }
req3 = requests.get(sl_url, params=sl_param)
#req3.text
sl_df = pd.DataFrame(req3.json()['resultSets'][0]['rowSet'],columns=req3.json()['resultSets'][0]['headers'])
print sl_df.shape
sl_df.head(10)

gl_url = "http://stats.nba.com/stats/playergamelog?"
gl_param = {'PlayerID':201935,
           'SeasonType': u'Regular Season',
           'Season': u'2014-15',
                            }
gl_req = requests.get(gl_url, params=gl_param)
gl_df = pd.DataFrame(gl_req.json()['resultSets'][0]['rowSet'],columns=gl_req.json()['resultSets'][0]['headers'])
gl_df.head(10)

print gl_df.shape

gl_season_df = gl_df.groupby('SEASON_ID').sum()
gl_season_df

gl_season_df['Player_ID'] = gl_season_df['Player_ID']/81
gl_season_df['FG_PCT'] = gl_season_df['FGM']/gl_season_df['FGA']
gl_season_df['FG3_PCT'] = gl_season_df['FG3M']/gl_season_df['FG3A']
gl_season_df['FT_PCT'] = gl_season_df['FTM']/gl_season_df['FTA']
gl_season_df = gl_season_df.assign(FG2A_FREQ = 1 - (gl_season_df['FG3A']/gl_season_df['FGA']))
gl_season_df = gl_season_df.assign(FG2M = gl_season_df['FGM']-gl_season_df['FG3M'])
gl_season_df = gl_season_df.assign(FG2M = gl_season_df['FGA']-gl_season_df['FG3A'])
gl_season_df = gl_season_df.assign(FG2_PCT = gl_season_df['FG2M']/gl_season_df['FG2A'])
gl_season_df = gl_season_df.assign(EFG_PCT = (gl_season_df['FGM']+.5*gl_season_df['FG3M'])/gl_season_df['FGA'])
gl_season_df

player_info_df[player_info_df['PLAYER_ID']==201935]




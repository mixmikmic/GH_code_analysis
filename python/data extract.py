import requests
import pandas as pd
import myql #https://github.com/josuebrunel/myql
from yahoo_oauth import OAuth1
import logging
pd.options.display.max_columns=None

oauth = OAuth1(None, None, from_file='credentials.json')
logging.disable(logging.CRITICAL)
logging.disable(logging.DEBUG)
logging.disable(logging.INFO)
yql = myql.MYQL(format='json', oauth=oauth, diagnostics =False)

resp=yql.raw_query('select * from fantasysports.draftresults where team_key="238.l.627060.t.8"')
resp.json()

resp=yql.raw_query('select * from fantasysports.leagues where league_key="223.l.431"')
resp.json()

yql.raw_query("select * from fantasysports.games where league='nfl'").content

#from https://developer.yahoo.com/fantasysports/guide/
#NFL Public fantasy teams
game_id_mapping={
    #"175":"2007",
    "199": "2008",
    #"222": "2009",
    #"242": "2010",
    "257": "2011",
    #"273": "2012",
    #f"314": "2013",#had to guess and check...
    "331": "2014",
    #'348': "2015",
    '359': "2016"
    #"371": "2017"
   
}

def flattenDict(d):
    for key, val in d.items():
        if type(val)==dict:
            del d[key]
            for k,v in val.items():
                val[key+"_"+k]=v
                del val[k]
                d.update(val)
                return flattenDict(d)
    return d
league_info_rows=[]
league_standings_rows=[]
def flattenTeamStandings(json):
    league=json['query']['results']['league']
    
    league_info_row={}
    for key in league:
        if type(league[key])!=dict:
            league_info_row[key]=league[key]
        else:
            if key=="standings":
                for team in league['standings']['teams']['team']:
                    league_standings=flattenDict(team)
                    league_standings['league_id']=league['league_id']
                    league_standings_rows.append(flattenDict(team))
            else:
                print("nested", key)
    league_info_rows.append(league_info_row)
    
flattenTeamStandings(yql.raw_query("select * from fantasysports.leagues.standings where league_key='175.l.219648'").json())
pd.DataFrame(league_standings_rows)

from time import sleep
import datetime
#~11 hours per "game_id" which is really a nfl fantasy season
#FYI this is kinda crap and should be improved.
ids_to_try=range(200000,300000)
N=len(ids_to_try)
league_info_rows=[]
league_standings_rows=[]
BASE_QUERY="select * from fantasysports.leagues.standings where league_key='GAME_ID.l.LEAGUE_ID'"
bad_ids=[]
for game_id in game_id_mapping.keys():
    print(game_id, game_id_mapping[game_id])
    for idx,league_id in enumerate(ids_to_try):
        query=BASE_QUERY.replace("GAME_ID",str(game_id)).replace("LEAGUE_ID",str(league_id))
        try:
            flattenTeamStandings(yql.raw_query(query).json())
        except:
            bad_ids.append(league_id)
            if len(bad_ids)%1000==0:
                print(len(bad_ids))
        if idx%1000==5:
            print(idx,N,idx/N, datetime.datetime.now())
            oauth = OAuth1(None, None, from_file='credentials.json')
            logging.disable(logging.CRITICAL)
            logging.disable(logging.DEBUG)
            logging.disable(logging.INFO)
            yql = myql.MYQL(format='json', oauth=oauth, diagnostics =False)
            pd.DataFrame(league_info_rows).to_csv("leage_info_rows-in-progressv5.csv",index=False)
            pd.DataFrame(league_standings_rows).to_csv("league_standings_rows-in-progressv5.csv",index=False)




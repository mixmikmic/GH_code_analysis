import os
import time
import datetime
import urllib2
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs

def injury_url_2_df(link):
    test = urllib2.urlopen(link).read()
    final = bs(test, "lxml")

    holder = list()
    for i in final.find_all("table", class_="datatable center"):
        holder.append(i.text.split("\n"))

    columns = holder[0][:7]
    data = holder[0][9:]

    i=0
    new_list=[]
    while i<len(data):
        new_list.append(data[i:i+7])
        i+=7

    df = pd.DataFrame(new_list)[[0,1,2,3,4]]
    df.columns = ["date", "team", 'acquired', 'relinquished', 'notes']
    return df

mth = datetime.datetime.today().month
mth = 12
mth

link = "http://www.prosportstransactions.com/basketball/Search/SearchResults.php?Player=&Team=&BeginDate=2017-{}-01&EndDate=&ILChkBx=yes&InjuriesChkBx=yes&PersonalChkBx=yes&DisciplinaryChkBx=yes&Submit=Search&start=".format(mth)

# base = pd.DataFrame()
# Shifting to have a base, and scraping just that month's data
base = pd.read_csv("injury_list_2017.csv")
try:
    for i in range(0, 1000, 25):
        print("On results " + str(i))
        base = base.append(injury_url_2_df(link + str(i)))
        time.sleep(1)
except:
    print("Hit end of page! We are done!")

base = base.drop_duplicates()
base.shape

base.to_csv("injury_list_2017.csv", index=False, encoding='utf-8')

base = pd.read_csv("injury_list_2017.csv")
base = base.reset_index(drop=True)
base['acquired'] = [i[5:] for i in base['acquired']]
base['relinquished'] = [i[5:] for i in base['relinquished']]
base['date'] = pd.to_datetime(base['date'], format="%Y-%m-%d")

injury = base[base['relinquished']!=""][["date", 'team', 'relinquished', "notes"]]
injury.rename(columns={'relinquished':'player'}, inplace=True)
injury['status'] = "injuried"

recovered = base[base['acquired']!=""][["date", 'team', 'acquired', "notes"]]
recovered.rename(columns={'acquired':'player'}, inplace=True)
recovered['status'] = "recovered"

c_injury_list = injury.append(recovered).sort_values(["player", "date"])
c_injury_list = c_injury_list[~c_injury_list.notes.str.contains("fined")]

testing_date = (datetime.datetime.today() - datetime.timedelta(days=2))
date = str(testing_date.date())
date

c_injury_list = c_injury_list[c_injury_list['date'] <= date]
c_injury_list = c_injury_list.groupby(['player']).last().reset_index()

# Only keep those that are still injuried. If their last status is recovered, I can remove them!
final = c_injury_list[c_injury_list.status=='injuried']

final.shape

final.tail()

final.date.max()

import glob
glob.glob("*.csv")

players = pd.read_csv('main_players_2017.csv')
games = pd.read_csv('game_date_2017.csv')

test_date = (testing_date).strftime("%A, %B %-d")
test_date

games_to_predict = pd.read_csv("2017_2018_nba_schedule.csv")
games = games_to_predict[games_to_predict.date==test_date]
games["away"] = [i.split(" ")[-1] for i in games['away']]
games["home"] = [i.split(" ")[-1] for i in games['home']]
games['away'].replace("Blazers", "Trail Blazers", inplace=True)
games['home'].replace("Blazers", "Trail Blazers", inplace=True)
teams_to_play = games['away'].tolist() + games['home'].tolist()

teams = pd.read_csv("team_2017.csv")
teams_to_play_ids = teams[teams.TEAM_NAME.isin(teams_to_play)].TEAM_ID.unique()

players_to_play = players[players.GAME_ID.isin(players[players.TEAM_ID.isin(teams_to_play_ids)].groupby(["TEAM_ID"]).last().reset_index().GAME_ID.tolist())]
players_to_play.shape

players_to_play.head(2)

player_season_stats = pd.read_csv("player_season_statistics.csv")
pstats16 = player_season_stats[player_season_stats.season==2016]
pstats16[pstats16.PLAYER_NAME.isin(test_team_players)].head(2)

tmp = pd.merge(players_to_play, pstats16, on=['PLAYER_ID'], how='left')

tmp[tmp.TEAM_ID==1610612744]

tmp.groupby(["TEAM_ID", "TEAM_ABBREVIATION"])[
    'defenders', 'facilitator', 'game_winners', 'inside_gamers', 'pure_scorers'].sum().reset_index()




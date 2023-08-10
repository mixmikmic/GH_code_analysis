import PbPMethods2 as pm2
import GetPbP

team_game_cf = {}
team_game_ca = {}
team_game = {}
for season in range(2015, 2017):
    for team in pm2.get_teams(season):
        if team not in team_game:
            team_game_cf[team] = []
            team_game_ca[team] = []
            team_game[team] = []
        cur_game = 0
        for line in pm2.read_team_pbp(team, season, strengths=['3v3'], 
                                      types=['GOAL', 'SHOT', 'MISS', 'BLOCK']):
            if pm2.get_event_period(line) == 4:
                g = pm2.get_game(line)
                if not g == cur_game:
                    cur_game = g
                    team_game[team].append('{0:d}0{1:d}'.format(season, g))
                    team_game_cf[team].append(0)
                    team_game_ca[team].append(0)
                if pm2.get_acting_team(line) == team:
                    team_game_cf[team][-1] += 1
                else:
                    team_game_ca[team][-1] += 1
    print('Done with', season)

w = open('./3v3 cf data.csv', 'w')
w.write('Team,Game,CF,CA')
for team in team_game:
    for i in range(len(team_game[team])):
        w.write('\n{0:s},{1:s},{2:d},{3:d}'.format(team, team_game[team][i], 
                                                 team_game_cf[team][i],
                                                 team_game_ca[team][i]))
w.close()

import pandas as pd

df = pd.read_csv('./3v3 cf data.csv')
df.head()

df[df.Team == 'WSH']

df.tail()

team_sec_cf = {}
team_sec_ca = {}
for season in range(2015, 2017):
    for team in pm2.get_teams(season):
        if team not in team_sec_cf:
            team_sec_cf[team] = [0 for i in range(301)] #5 minutes
            team_sec_ca[team] = [0 for i in range(301)] #5 minutes
        for line in pm2.read_team_pbp(team, season, strengths=['3v3'], 
                                      types=['GOAL', 'SHOT', 'MISS', 'BLOCK']):
            if pm2.get_event_period(line) == 4:
                time = pm2.convert_time(pm2.get_event_time(line), 4) - 3600
                if pm2.get_acting_team(line) == team:
                    team_sec_cf[team][time] += 1
                else:
                    team_sec_ca[team][time] += 1
    print('Done with', season)

w = open('./3v3 cf data by sec.csv', 'w')
w.write('Team,Sec,CF,CA,CumCF,CumCA,CumCF%')
for team in team_sec_cf:
    cfsum = 0
    casum = 0
    for i, (cf, ca) in enumerate(zip(team_sec_cf[team], team_sec_ca[team])):
        cfsum += cf
        casum += ca
        if cfsum + casum == 0:
            cfpct = 0
        else:
            cfpct = cfsum*100/(cfsum+casum)
        w.write('\n{0:s},{1:d},{2:d},{3:d},{4:d},{5:d},{6:.1f}'.format(team, i, cf, ca,
                                                                    cfsum, casum, cfpct))
w.close()

df = pd.read_csv('./3v3 cf data by sec.csv')
df[df.Sec == 300].head()

df[df.Team == 'WSH'].head()

from pylab import *
get_ipython().magic('matplotlib inline')
df[df.Team == 'WSH']['CumCF%'].plot()




import pandas as pd
import numpy as np
import scipy.stats as scipy
import random

df = pd.read_csv("./Data/E0_13.csv")
df_14 = pd.read_csv("./Data/E0_14.csv")

df.columns

res_13 = df.ix[:,:23]
res_13 = res_13.drop(['Div','Date','Referee'],axis=1)
res_14 = df_14.ix[:,:23]
res_14 = res_14.drop(['Div','Date','Referee'],axis=1)
bet_13 = df.ix[:,23:]

res_13.head()

#Team, Home Goals Score, Away Goals Score, Attack Strength, Home Goals Conceded, Away Goals Conceded, Defensive Strength
table_13 = pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS'))

avg_home_scored_13 = res_13.FTHG.sum() / 380.0
avg_away_scored_13 = res_13.FTAG.sum() / 380.0
avg_home_conceded_13 = avg_away_scored_13
avg_away_conceded_13 = avg_home_scored_13
print "Average number of goals at home",avg_home_scored_13
print "Average number of goals away", avg_away_scored_13
print "Average number of goals conceded at home",avg_away_conceded_13
print "Average number of goals conceded away",avg_home_conceded_13

res_home = res_13.groupby('HomeTeam')
res_away = res_13.groupby('AwayTeam')

table_13.Team = res_home.HomeTeam.all().values
table_13.HGS = res_home.FTHG.sum().values
table_13.HGC = res_home.FTAG.sum().values
table_13.AGS = res_away.FTAG.sum().values
table_13.AGC = res_away.FTHG.sum().values
table_13.head()

table_13.HAS = (table_13.HGS / 19.0) / avg_home_scored_13
table_13.AAS = (table_13.AGS / 19.0) / avg_away_scored_13
table_13.HDS = (table_13.HGC / 19.0) / avg_home_conceded_13
table_13.ADS = (table_13.AGC / 19.0) / avg_away_conceded_13
table_13.head()

#Expected number of goals based on the average poisson probability
def exp_goals(mean):
    max_pmf = 0;
    for i in xrange(7):
        pmf = scipy.distributions.poisson.pmf(i,mean) * 100 
        if pmf > max_pmf:
            max_pmf = pmf
            goals = i
    return goals

test_13 = res_13.ix[:,0:5]
test_13.head()
test_14 = res_14.ix[:,0:5]
test_14.head()

table_13[table_13['Team'] == 'Arsenal']
test_14['ER'] = ''
test_14 = test_14.drop(test_14.index[380],axis=0)

results = []
for index, row in test_13.iterrows():

    home_team = table_13[table_13['Team'] == row['HomeTeam']]
    away_team = table_13[table_13['Team'] == row['AwayTeam']]
    #print "Home : ", home_team.HAS.values, "Away: ", away_team.AAS.
    if row.HomeTeam not in ['Leicester', 'QPR', 'Burnley'] and row.AwayTeam not in ['Leicester', 'QPR', 'Burnley']:
        EH = home_team.HAS.values * away_team.ADS.values * avg_home_scored_13
        EA = home_team.HDS.values * away_team.AAS.values * avg_home_conceded_13
        #print row.HomeTeam, row.AwayTeam
        if EH[0] > EA[0]:
            results.append('H')
        else:
            results.append('A')
    else:
        results.append('D')

len(results)

test_13['ER'] = results

accuracy_score(test_13['ER'],test_13['FTR'])

team_1 = 'Man United'
team_2 = 'Cardiff'

home_team = table_13[table_13['Team'] == team_1]
away_team = table_13[table_13['Team'] == team_2]
EH = home_team.HAS.values * away_team.ADS.values * avg_home_scored_13
EA = home_team.HDS.values * away_team.AAS.values * avg_home_conceded_13

home_team.Team.values[0]

def exp_goals_prob(mean):
    max_pmf = 0;
    prob = []
    for i in xrange(0,6):
        pmf = scipy.distributions.poisson.pmf(i,mean) * 100 
        prob.append(pmf[0])
    return prob

prob_goals = pd.DataFrame(columns=['Team','0','1','2','3','4','5'])
home_team_prob = exp_goals_prob(EH)
away_team_prob = exp_goals_prob(EA)

prob_goals.loc[0,1:] = home_team_prob
prob_goals.loc[1,1:] = away_team_prob
prob_goals.iloc[0,0] = team_1
prob_goals.iloc[1,0] = team_2

prob_goals


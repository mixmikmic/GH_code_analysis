import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from pylab import *
get_ipython().magic('matplotlib inline')
import numpy as np
from sklearn import datasets, linear_model
from sklearn import cross_validation

#read serie A data
df_coll = []
'''
#serie A 
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/I1.csv"))
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1415/I1.csv"))
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1314/I1.csv"))
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1213/I1.csv"))
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1112/I1.csv"))
#df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1011/I1.csv"))
#df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/0910/I1.csv"))
#df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/0809/I1.csv"))
'''

#Bundes 
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/D1.csv"))
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1415/D1.csv"))
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1314/D1.csv"))
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1213/D1.csv"))
df_coll.append(pd.read_csv("http://www.football-data.co.uk/mmz4281/1112/D1.csv"))



'''
#read EPL data
#df_cal   = pd.read_csv("../data/EPL-1516.csv")
df_cal_2 = pd.read_csv("../data/EPL-1415.csv")
df_cal_3 = pd.read_csv("../data/EPL-1314.csv")
df_cal_4 = pd.read_csv("../data/EPL-1213.csv")
df_cal_5 = pd.read_csv("../data/EPL-1112.csv")

#read La Liga data
df_cal   = pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/SP1.csv")
df_cal_2 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1415/SP1.csv")
df_cal_3 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1314/SP1.csv")
df_cal_4 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1213/SP1.csv")
df_cal_5 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1112/SP1.csv")
'''


df_coll[-1].head(10)
df_coll[-1].keys()

def get_teams(df):
    
    teams = []
    
    for i,cal in enumerate(df.iterrows()):
        teams.append(cal[1]["HomeTeam"])
        teams.append(cal[1]["AwayTeam"])

        if i == 9: 
            return teams

teams_a  = get_teams(df_coll[0])
teams_coll = [get_teams(df) for df in df_coll]

print teams_coll[4]

def build_features(df, teams):

    team_features = []
    
    for team in teams:
        #print team_a
        team_home = df[df['HomeTeam']==team]
        team_away = df[df['AwayTeam']==team]
        
        #shots made
        team_s    = team_away["AS"].sum()  + team_home["HS"].sum()
        #shots on-target made
        team_st   = team_away["AST"].sum() + team_home["HST"].sum()
        #shots conceded
        team_sc    = team_away["HS"].sum()  + team_home["AS"].sum()
        #shots on-target conceded
        team_stc   = team_away["HST"].sum() + team_home["AST"].sum()
        #corners awarded
        team_c    = team_away["AC"].sum()  + team_home["HC"].sum()
        #corners conceded
        team_cc    = team_away["HC"].sum()  + team_home["AC"].sum()
        #woodwork hit
        #team_w    = team_away["AHW"].sum()  + team_home["HHW"].sum()
        #woodwork conceded
        #team_wc = team_away["HHW"].sum()  + team_home["AHW"].sum()

        team_features.append([team_s,team_sc,team_st,team_stc,team_c,team_cc])

    return team_features
    

def build_target(df, teams):

    team_target = []
    
    for team in teams:
        #print team_a
        t      = df[(df['HomeTeam']==team) | (df['AwayTeam']==team)]
        team_home = df[df['HomeTeam']==team]
        team_away = df[df['AwayTeam']==team]

        team_h_win = len(team_home[team_home['FTHG']>team_home['FTAG']])
        team_a_win = len(team_away[team_away['FTAG']>team_away['FTHG']])
        team_draw = len(t[t['FTAG']==t['FTHG']])

        team_points = 3*team_a_win + 3*team_h_win + team_draw
        team_target.append(team_points)

    return team_target

feat_test = build_features(df_coll[0],teams_a)
targ_test = build_target(df_coll[0],teams_a)

print len(feat_test)
print len(targ_test)

print feat_test
print targ_test

features = []

for seas,teams in zip(df_coll,teams_coll): 
    features += build_features(seas,teams) 

print len(features)

target = []

for seas,teams in zip(df_coll,teams_coll): 
    target += build_target(seas,teams)

print len(target)

df = pd.DataFrame(features, columns=["shots","shots_ontarget","shots_conceded","shots_conceded_ontarget","corners","corners_conceded"])

df["points"] = target

df.to_csv("../data/proc_data_de.csv")

#print features
#print target

# Plot outputs
figure(1)
figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(np.array(features)[:,0], target,  color='b',s=180,alpha=0.5)

xlabel('Shots Made')
ylabel('Points')

plt.show()

# Plot outputs
figure(1)
figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(np.array(features)[:,1], target,  color='r',s=180,alpha=0.5)

xlabel('Shots Conceded')
ylabel('Points')


# Plot outputs
figure(1)
figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(np.array(features)[:,2], target,  color='b',s=180,alpha=0.5)

xlabel('Shots On Target')
ylabel('Points')

plt.show()

# Plot outputs
figure(1)
figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(np.array(features)[:,3], target,  color='r',s=180,alpha=0.5)

xlabel('Shots On Target Conceded')
ylabel('Points')


# Plot outputs
figure(1)
figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(np.array(features)[:,4], target,  color='b',s=180,alpha=0.5)

xlabel('Corners Awarded')
ylabel('Points')


# Plot outputs
figure(1)
figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(np.array(features)[:,5], target,  color='r',s=180,alpha=0.5)

xlabel('Corners Conceded')
ylabel('Points')

df = pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/D1.csv")

df




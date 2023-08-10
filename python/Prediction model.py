import pandas as pd
import numpy as np
import seaborn as sns

rb1 = pd.read_csv('RB_Week1')
rb2 = pd.read_csv('RB_Week2')
rb3 = pd.read_csv('RB_Week3')
rb4 = pd.read_csv('RB_Week4')
rb5 = pd.read_csv('RB_Week5')
rb6 = pd.read_csv('RB_Week6')
rb7 = pd.read_csv('RB_Week7')
rb8 = pd.read_csv('RB_Week8')
rb9 = pd.read_csv('RB_Week9') ###
current = pd.read_csv('Current_week')

current

'''''
Ari = 1
Atl = 2
Bal = 3
Buf = 4
Car = 5 
Chi = 6
Cin = 7
Cle = 8
Dal = 9
Den = 10 
Det = 11
GB = 12
Hou = 13
Ind = 14
Jax = 15
KC = 16
Lac = 17
Lar =18
Mia =19
Min =20
NE = 21
NO =22
NYG = 23
NYJ =24
Oak =25
Phi =26
Pit =27
SEA = 28
SF = 29
TB = 30
Ten = 31
Wsh = 32
'''''
def team(row):
    if row['Team'] == 'Ari':
        return 1
    elif row['Team'] == 'Atl':
        return 2
    elif row['Team'] == 'Bal':
        return 3
    elif row['Team'] == 'Buf': 
        return 4
    elif row['Team'] == 'Car':
        return 5 
    elif row['Team'] == 'Chi':
        return 6
    elif row['Team'] == 'Cin':
        return 7
    elif row['Team'] == 'Cle':
        return 8
    elif row['Team'] == 'Dal':
        return 9
    elif row['Team'] == 'Den':
        return 10
    elif row['Team'] == 'Det': 
        return 11
    elif row['Team'] == 'GB':
        return 12
    elif row['Team'] == 'Hou':
        return 13
    elif row['Team'] == 'Ind': 
        return 14
    elif row['Team'] =='Jax': 
        return 15
    elif row['Team'] =='KC': 
        return 16
    elif row['Team'] =='LAC':
        return 17
    elif row['Team'] =='LAR':
        return 18
    elif row['Team'] =='Mia':
        return 19
    elif row['Team'] =='Min':
        return 20
    elif row['Team'] =='NE':
        return 21
    elif row['Team'] =='NO':
        return 22
    elif row['Team'] =='NYG': 
        return 23
    elif row['Team'] =='NYJ':
        return 24
    elif row['Team'] =='Oak':
        return 25
    elif row['Team'] =='Phi':
        return 26
    elif row['Team'] =='Pit':
        return 27
    elif row['Team'] =='Sea':
        return 28
    elif row['Team'] =='SF':
        return 29
    elif row['Team'] =='TB':
        return 30
    elif row['Team'] =='Ten': 
        return 31
    elif row['Team'] =='Wsh':
        return 32

def Opp(row):
    if row['Opp'] == 'Ari' or row['Opp'] == '@Ari' :
        return 1
    elif row['Opp'] == 'Atl' or row['Opp'] == '@Atl':
        return 2
    elif row['Opp'] == 'Bal' or row['Opp'] == '@Bal':
        return 3
    elif row['Opp'] == 'Buf' or row['Opp'] == '@Buf': 
        return 4
    elif row['Opp'] == 'Car' or row['Opp'] == '@Car':
        return 5 
    elif row['Opp'] == 'Chi' or row['Opp'] == '@Chi':
        return 6
    elif row['Opp'] == 'Cin' or row['Opp'] == '@Cin':
        return 7
    elif row['Opp'] == 'Cle' or row['Opp'] == '@Cle':
        return 8
    elif row['Opp'] == 'Dal' or row['Opp'] == '@Dal':
        return 9
    elif row['Opp'] == 'Den' or row['Opp'] == '@Den':
        return 10
    elif row['Opp'] == 'Det' or row['Opp'] == '@Det': 
        return 11
    elif row['Opp'] == 'GB' or row['Opp'] == '@GB':
        return 12
    elif row['Opp'] == 'Hou' or row['Opp'] == '@Hou':
        return 13
    elif row['Opp'] == 'Ind' or row['Opp'] == '@Ind': 
        return 14
    elif row['Opp'] =='Jax' or row['Opp'] == '@Jax': 
        return 15
    elif row['Opp'] =='KC' or row['Opp'] == '@KC': 
        return 16
    elif row['Opp'] =='LAC' or row['Opp'] == '@LAC':
        return 17
    elif row['Opp'] =='LAR' or row['Opp'] == '@LAR':
        return 18
    elif row['Opp'] =='Mia' or row['Opp'] == '@Mia':
        return 19
    elif row['Opp'] =='Min' or row['Opp'] == '@Min':
        return 20
    elif row['Opp'] =='NE' or row['Opp'] == '@NE':
        return 21
    elif row['Opp'] =='NO' or row['Opp'] == '@NO':
        return 22
    elif row['Opp'] =='NYG' or row['Opp'] == '@NYG': 
        return 23
    elif row['Opp'] =='NYJ' or row['Opp'] == '@NYJ':
        return 24
    elif row['Opp'] =='Oak' or row['Opp'] == '@Oak':
        return 25
    elif row['Opp'] =='Phi' or row['Opp'] == '@Phi':
        return 26
    elif row['Opp'] =='Pit' or row['Opp'] == '@Pit':
        return 27
    elif row['Opp'] =='Sea' or row['Opp'] == '@Sea':
        return 28
    elif row['Opp'] =='SF' or row['Opp'] == '@SF':
        return 29
    elif row['Opp'] =='TB' or row['Opp'] == '@TB':
        return 30
    elif row['Opp'] =='Ten' or row['Opp'] == '@Ten': 
        return 31
    elif row['Opp'] =='Wsh' or row['Opp'] == '@Wsh':
        return 32
    else:
        return 'bye'

rb1['Team'] = rb1.apply(team, axis = 1)
rb2['Team'] = rb2.apply(team, axis = 1)
rb3['Team'] = rb3.apply(team, axis = 1)
rb4['Team'] = rb4.apply(team, axis = 1)
rb5['Team'] = rb5.apply(team, axis = 1)
rb6['Team'] = rb6.apply(team, axis = 1)
rb7['Team'] = rb7.apply(team, axis = 1)
rb8['Team'] = rb8.apply(team, axis = 1)
rb9['Team'] = rb9.apply(team, axis = 1) ####

def clean_df(df):
    df.drop('Unnamed: 0', axis = 1, inplace = True)
    return df
rb1 = clean_df(rb1)
rb2 = clean_df(rb2)
rb3 = clean_df(rb3)
rb4 = clean_df(rb4)
rb5 = clean_df(rb5)
rb6 = clean_df(rb6)
rb7 = clean_df(rb7)
rb8 = clean_df(rb8)
rb9 = clean_df(rb9)  #####
current = clean_df(current)

current.head()

rb1['Week'] = np.ones(150, dtype ='int')
rb2['Week'] = np.ones(150, dtype ='int') * 2
rb3['Week'] = np.ones(150, dtype ='int') * 3
rb4['Week'] = np.ones(150, dtype ='int') * 4
rb5['Week'] = np.ones(150, dtype ='int') * 5
rb6['Week'] = np.ones(150, dtype ='int') * 6
rb7['Week'] = np.ones(150, dtype ='int') * 7
rb8['Week'] = np.ones(150, dtype ='int') * 8
rb9['Week'] = np.ones(150, dtype ='int') * 9 ###
current['Week'] = np.ones(len(current), dtype ='int') * 10 ###
rb1.set_index(['Week','Player'], inplace = True)
rb2.set_index(['Week','Player'], inplace = True)
rb3.set_index(['Week','Player'], inplace = True)
rb4.set_index(['Week','Player'], inplace = True)
rb5.set_index(['Week','Player'], inplace = True)
rb6.set_index(['Week','Player'], inplace = True)
rb7.set_index(['Week','Player'], inplace = True)
rb8.set_index(['Week','Player'], inplace = True)
rb9.set_index(['Week','Player'], inplace = True) ###
current.set_index(['Week','Player'], inplace = True)
rb1.sort_index(inplace = True)
rb2.sort_index(inplace = True)
rb3.sort_index(inplace = True)
rb4.sort_index(inplace = True)
rb5.sort_index(inplace = True)
rb6.sort_index(inplace = True)
rb7.sort_index(inplace = True)
rb8.sort_index(inplace = True)
rb9.sort_index(inplace = True) ###
current.sort_index(inplace = True)

#calculates the scores to train on
weekly_stats = [rb2,rb3,rb4,rb5,rb6,rb7,rb8, rb9] ####

def calculate_score(row):
    score = (row['Rushing YDS'] / 10) + (row['Rush TD'] * 6) + (row['Rec YDS'] / 10) + (row['2PC'] * 2) +(row['Rec TD'] * 6) - (row['Fuml'] * 3)
    return score
score = pd.DataFrame()
score['FP'] = rb1.apply(calculate_score, axis = 1)

for week in weekly_stats:
    nextweek_score = pd.DataFrame()
    nextweek_score['FP'] = week.apply(calculate_score, axis = 1)
    score = score.append(nextweek_score)

score

#dataframe of all stats
stats = rb1
stats = stats.append(rb2)
stats = stats.append(rb3)
stats = stats.append(rb4)
stats = stats.append(rb5)
stats = stats.append(rb6)
stats = stats.append(rb7)
stats = stats.append(rb8)
stats = stats.append(rb9) ###

A = '@Ari'
A = list(A)
"".join(A[1:])

def Away(row):
    away = list(row['Opp'])
    if away[0] == '@':
        return 'Away'
    else:
        return 'Home'
stats['Setting'] = stats.apply(Away, axis =1)

# def Opp(row):
#     away = list(row['Opp'])
#     if away[0] == '@':
#         return "".join(away[1:])
#     else:
#         return "".join(away)
stats['Opp'] = stats.apply(Opp, axis =1)

stats

current

current = current[current.Opp != 'bye']
current['Team'] = current.apply(team, axis = 1)
test['Opp'] = test.apply(Opp, axis = 1)
#current['Opp'] = current.apply(Opp, axis =1)

current

list(rb1.apply(Opp, axis =1))

#initilizes the test dataframe
#gives averages going into the game
test = pd.DataFrame(index = rb1.index)
test['Team'] = list(rb1['Team'])
test['Opp'] = list(rb1.apply(Opp, axis =1))
test['Score'] = np.zeros(150)
test['Attempts'] = np.zeros(150)
test['Rushing YDS'] = np.zeros(150)
test['Rush TD'] = np.zeros(150)
test['Receptions'] = np.zeros(150)
test['Rec YDS'] = np.zeros(150)
test['Rec TD'] = np.zeros(150)
test['Targets'] = np.zeros(150)
test['2PC'] = np.zeros(150)
test['Fuml'] = np.zeros(150)

stats = pd.DataFrame()

def team_score(row):
    return int(row['Score'].split()[1].split('-')[0])

weekly_stats = [rb1,rb2,rb3,rb4,rb5,rb6,rb7,rb8,rb9,current] ###

for week in range(len(weekly_stats)-1):
    current_week = week + 1
    next_week = week + 2
    df = weekly_stats[current_week]
    to_add = pd.DataFrame(index = df.index)
    to_add['Team'] = list(df['Team'])
    to_add['Opp'] = list(df.apply(Opp, axis = 1))
    current_players = list(df.reset_index(level = 0).index)
    player_averages = pd.DataFrame(index = df.loc[next_week].index)
    stats = stats.append(weekly_stats[week])
    stats['score'] = stats.apply(team_score, axis = 1)
    player_averages['Score'] = stats.groupby('Player').sum()['score'].filter(items = current_players) / (week+1)
    player_averages['Attempts'] = stats.groupby('Player').sum()['Attempts'].filter(items = current_players) / (week+1)
    player_averages['Rushing YDS'] = stats.groupby('Player').sum()['Rushing YDS'].filter(items = current_players) / (week+1)
    player_averages['Rush TD'] = stats.groupby('Player').sum()['Rush TD'].filter(items = current_players) / (week +1)
    player_averages['Receptions'] = stats.groupby('Player').sum()['Receptions'].filter(items = current_players) /(week+1)
    player_averages['Rec YDS'] = stats.groupby('Player').sum()['Rec YDS'].filter(items = current_players) / (week+1)
    player_averages['Rec TD'] = stats.groupby('Player').sum()['Rec TD'].filter(items = current_players) / (week+1)
    player_averages['Targets'] = stats.groupby('Player').sum()['Targets'].filter(items = current_players) / (week+1)
    player_averages['2PC'] = stats.groupby('Player').sum()['2PC'].filter(items = current_players) / (week+1)
    player_averages['Fuml'] = stats.groupby('Player').sum()['Fuml'].filter(items = current_players) / (week+1)
    player_averages['Week'] = np.ones(len(df), dtype = 'int')*next_week
    player_averages.reset_index(inplace = True)
    player_averages.set_index(['Week','Player'],inplace = True)
    
    player_averages.fillna(value= 0, inplace = True)
    to_add = to_add.join(player_averages)
    test = test.append(to_add)

test

to_add

def Score(row):
    if row['Score'] < 10:
        return 0
    elif row['Score'] >= 10 and row['Score'] < 20:
        return 1
    elif row['Score'] >= 20 and row['Score'] < 30:
        return 2
    elif  row['Score'] >= 30 and row['Score'] < 40:
        return 3
    else:
        return 4
test['Score'] = test.apply(Score, axis =1)

def Score(row):
    if row['Score'] < 10:
        return 0
    elif row['Score'] >= 10 and row['Score'] < 20:
        return 1
    elif row['Score'] >= 20 and row['Score'] < 30:
        return 2
    elif  row['Score'] >= 30 and row['Score'] < 40:
        return 3
    else:
        return 4
test['Score'] = test.apply(Score, axis =1) 

def Attempts(row):
    if row['Attempts'] < 5:
        return 0
    elif row['Attempts'] >= 5 and row['Attempts']  < 10:
        return 1
    elif row['Attempts'] >= 10 and row['Attempts']  < 20:
        return 2
    elif row['Attempts'] >= 20:
        return 3
test['Attempts'] = test.apply(Attempts, axis =1)

def Rush_yds(row):
    if row['Rushing YDS'] < 10:
        return 0
    elif row['Rushing YDS'] >= 10 and row['Rushing YDS'] < 30:
        return 1
    elif row['Rushing YDS'] >= 30 and row['Rushing YDS'] < 50:
        return 2
    elif row['Rushing YDS'] >= 50 and row['Rushing YDS'] < 70:
        return 3
    elif row['Rushing YDS'] >= 70 and row['Rushing YDS'] < 90:
        return 4
    elif row['Rushing YDS'] >= 90 and row['Rushing YDS'] < 110:
        return 5
    else:
        return 6
test['Rushing YDS'] = test.apply(Rush_yds, axis =1)

def Rush_TD(row):
    if row['Rush TD'] < .1:
        return 0
    elif row['Rush TD'] >= .1 and row['Rush TD'] < .2:
        return 1
    elif row['Rush TD'] >= .2 and row['Rush TD'] < .3:
        return 2
    elif row['Rush TD'] >= .3 and row['Rush TD'] < .4:
        return 3
    elif row['Rush TD'] >= .4 and row['Rush TD'] < .5:
        return 4
    elif row['Rush TD'] >= .5 and row['Rush TD'] < .6:
        return 5
    elif row['Rush TD'] >= .6 and row['Rush TD'] < 1:
        return 6
    else:
        return 7
test['Rush TD'] = test.apply(Rush_TD, axis =1)

def Receptions(row):
    if row['Receptions'] < 1:
        return 0
    elif row['Receptions'] >= 1 and row['Receptions'] < 2:
        return 1
    elif row['Receptions'] >= 2 and row['Receptions'] < 3:
        return 2
    elif row['Receptions'] >= 3 and row['Receptions'] < 4:
        return 3
    elif row['Receptions'] >= 4 and row['Receptions'] < 5:
        return 4
    elif row['Receptions'] >= 5 and row['Receptions'] < 6:
        return 5
    else:
        return 6
test['Receptions'] = test.apply(Receptions, axis =1) 

def Rec_yds(row):
    if row['Rec YDS'] < 5:
        return 0
    elif row['Rec YDS'] >= 5 and row['Rec YDS'] < 10:
        return 1
    elif row['Rec YDS'] >= 10 and row['Rec YDS'] < 20:
        return 2
    elif row['Rec YDS'] >= 20 and row['Rec YDS'] < 30:
        return 3
    elif row['Rec YDS'] >= 30 and row['Rec YDS'] < 40:
        return 4
    elif row['Rec YDS'] >= 40 and row['Rec YDS'] < 50:
        return 5
    elif row['Rec YDS'] >= 50 and row['Rec YDS'] < 60:
        return 6
    elif row['Rec YDS'] >= 60 and row['Rec YDS'] < 70:
        return 7
    else:
        return 8
test['Rec YDS'] = test.apply(Rec_yds, axis =1)

def Rec_TD(row):
    if row['Rec TD'] < .1:
        return 0
    elif row['Rec TD'] >= .1 and row['Rec TD'] < .2:
        return 1
    elif row['Rec TD'] >= .2 and row['Rec TD'] < .3:
        return 2
    elif row['Rec TD'] >= .3 and row['Rec TD'] < .4:
        return 3
    elif row['Rec TD'] >= .4 and row['Rec TD'] < .5:
        return 4
    elif row['Rec TD'] >= .5 and row['Rec TD'] < .6:
        return 5
    elif row['Rec TD'] >= .6 and row['Rec TD'] < 1:
        return 6
    else:
        return 7
test['Rec TD'] = test.apply(Rec_TD, axis =1)

def Targets(row):
    if row['Targets'] < 5:
        return 0
    elif row['Targets'] >= 5 and row['Targets'] < 10:
        return 1
    elif row['Targets'] >= 10 and row['Targets'] < 15:
        return 2
    elif row['Targets'] >= 15 and row['Targets'] < 20:
        return 3
    else:
        return 4
test['Targets'] = test.apply(Targets, axis =1)

def PC(row):
    if row['2PC'] > 0:
        return 1
    else:
        return 0
test['2PC'] = test.apply(PC, axis =1)

#fumbles greater than .5 fumble on more than 10 carries
def Fuml(row):
    if row['Attempts'] >= 2:
        if row['Fuml'] > .5:
            return 1
        else: 
            return 2
    else:
        return 0
test['Fuml'] = test.apply(Fuml, axis =1)

def FP(row):
    if row['FP'] < 5:
        return 0
    if row['FP'] >= 5 and row['FP'] < 10:
        return 1
    if row['FP'] >= 10 and row['FP'] < 15:
        return 2
    if row['FP'] >= 15 and row['FP'] < 20:
        return 3
    if row['FP'] >= 20 and row['FP'] < 25:
        return 4
    else:
        return 5
score['FP'] = score.apply(FP, axis =1)

#to_plot = test
#to_plot['FP'] = score['FP']
#%matplotlib inline
#sns.lmplot('Attempts', 'FP',data = to_plot)

train = test.drop(10) #previous weeks
X_test = test.loc[10] #current week

test.loc[10]

X_test

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators= 100)
X = train
Y = score
forest = forest.fit(X,Y)
predictions = pd.DataFrame({'Predicted Points':forest.predict(X_test)}, index = current.index)

def unpack(row):
    if row['Predicted Points'] == 0:
        return '0-5'
    elif row['Predicted Points'] == 1:
        return '5-10'
    elif row['Predicted Points'] == 2:
        return '10-15'
    elif row['Predicted Points'] == 3:
        return '15-20'
    elif row['Predicted Points'] == 4:
        return '20-25'
    else:
        return '>25'
    
predictions['Predicted Points'] = predictions.apply(unpack, axis = 1)

predictions

predictions.to_csv('week 10 RB predictions')

stats.to_csv('all RB stats')


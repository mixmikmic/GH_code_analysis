import pandas as pd
import numpy as np
import scipy.stats as scipy
import random

df = pd.read_csv("./Data/England/E0_13.csv")
df_14 = pd.read_csv("./Data/England/E0_14.csv")

df.columns

res_13 = df.ix[:,:23]
res_13 = res_13.drop(['Div','Date','Referee'],axis=1)
res_14 = df_14.ix[:,:23]
res_14 = res_14.drop(['Div','Date','Referee'],axis=1)
table_features = df.ix[:,:7]
table_features = table_features.drop(['FTHG','FTAG','Div','Date'],axis=1)
bet_13 = df.ix[:,23:]

from math import log

entropy = -((0.32 * log(0.32,3)) + (0.20 * log(0.20,3)) + (0.47 * log(0.47,3)))

entropy

res_13.head()
feature_table = df.ix[:,:23]

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

feature_table = feature_table[['HomeTeam','AwayTeam','FTR']]
f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for index,row in feature_table.iterrows():
    f_HAS.append(table_13[table_13['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table_13[table_13['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table_13[table_13['Team'] == row['HomeTeam']]['AAS'].values[0])
    f_ADS.append(table_13[table_13['Team'] == row['HomeTeam']]['ADS'].values[0])
    
feature_table['HAS'] = f_HAS
feature_table['HDS'] = f_HDS
feature_table['AAS'] = f_AAS
feature_table['ADS'] = f_ADS

feature_table.head()

def transformResult(row):
    if(row.FTR == 'H'):
        return 1
    elif(row.FTR == 'A'):
        return -1
    else:
        return 0

feature_table["Result"] = feature_table.apply(lambda row: transformResult(row),axis=1)

feature_table.head()

X_train = feature_table[['HAS','HDS','AAS','ADS']]
y_train = feature_table['Result']

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

outcome_list = [-1,0,1]
y_pred = []
for i in xrange(1,381):
    y_pred.append(random.choice(outcome_list))
    

accuracy_score(y_train,y_pred)

clf1 = DecisionTreeClassifier()
clf2 = XGBClassifier()
clf3 = KNeighborsClassifier(n_neighbors=15)
clf3.fit(X_train,y_train)

y_pred = clf3.predict(X_train)
accuracy_score(y_pred,y_train)






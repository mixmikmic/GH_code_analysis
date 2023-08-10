import pandas as pd
import numpy as np
import scipy.stats as scipy
from __future__ import division

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

res_13.head()
feature_table = df.ix[:,:23]

#Team, Home Goals Score, Away Goals Score, Attack Strength, Home Goals Conceded, Away Goals Conceded, Defensive Strength
table_13 = pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS',))

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

# list(res_home)[0]

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

feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST']]
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

X_train = feature_table[['HST','AST','HAS','HDS','AAS','ADS',]]
y_train = feature_table['Result']

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

clf = DecisionTreeClassifier()
clf2 = MultinomialNB()
clf3 = XGBClassifier()
clf3.fit(X_train,y_train)

y_pred = clf3.predict(X_train)
accuracy_score(y_pred,y_train)
scores = cross_val_score(clf2, X_train, y_train, cv=10)

print scores
print scores.mean()

ht = res_13.loc[40].HomeTeam
at = res_13.loc[40].AwayTeam

feat_table = res_13.sort_index(ascending=False)
feat_table = feat_table[['HomeTeam','AwayTeam','FTR','FTHG','FTAG','HS','AS','HC','AC']]
feat_table["pastHS"] = 0.0
feat_table["pastHC"] = 0.0
feat_table["pastAS"] = 0.0
feat_table["pastAC"] = 0.0
feat_table["pastHG"] = 0.0
feat_table["pastAG"] = 0.0


feat_table.head()

k = 3
for i in range(379,-1,-1):
    row = feat_table.loc[i]
    ht = row.HomeTeam
    at = row.AwayTeam
    ht_stats = feat_table.loc[i-1:-1][(feat_table.HomeTeam == ht) | (feat_table.AwayTeam == ht)].head(k)
    at_stats = feat_table.loc[i-1:-1][(feat_table.HomeTeam == at) | (feat_table.AwayTeam == at)].head(k)

    feat_table.set_value(i, 'pastHC', (ht_stats[ht_stats["AwayTeam"] == ht].sum().HC + ht_stats[ht_stats["HomeTeam"] == ht].sum().HC)/k)
    feat_table.set_value(i, 'pastAC', (at_stats[at_stats["AwayTeam"] == at].sum().HC + at_stats[at_stats["HomeTeam"] == at].sum().HC)/k)
    feat_table.set_value(i, 'pastHS', (ht_stats[ht_stats["AwayTeam"] == ht].sum().HS + ht_stats[ht_stats["HomeTeam"] == ht].sum().AS)/k)
    feat_table.set_value(i, 'pastAS', (at_stats[at_stats["AwayTeam"] == at].sum().HS + at_stats[at_stats["HomeTeam"] == at].sum().AS)/k)
    feat_table.set_value(i, 'pastHG', (ht_stats[ht_stats["AwayTeam"] == ht].sum().FTAG + ht_stats[ht_stats["HomeTeam"] == ht].sum().FTHG)/k)
    feat_table.set_value(i, 'pastAG', (at_stats[at_stats["AwayTeam"] == at].sum().FTAG + at_stats[at_stats["HomeTeam"] == at].sum().FTHG)/k)

f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for index,row in feature_table.iterrows():
    f_HAS.append(table_13[table_13['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table_13[table_13['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table_13[table_13['Team'] == row['HomeTeam']]['AAS'].values[0])
    f_ADS.append(table_13[table_13['Team'] == row['HomeTeam']]['ADS'].values[0])
    
feat_table['HAS'] = f_HAS
feat_table['HDS'] = f_HDS
feat_table['AAS'] = f_AAS
feat_table['ADS'] = f_ADS

test_table = feat_table.drop(['FTHG','FTAG','HS','AS','HC','AC'],axis=1)

test_table["Result"] = test_table.apply(lambda row: transformResult(row),axis=1)
test_table.sort_index().head()



X_train = test_table[['pastHS','pastHC','pastAS','pastAC','pastHG','pastAG','HAS','HDS','AAS','ADS']]
y_train = test_table['Result']



clf1 = DecisionTreeClassifier()
clf2 = MultinomialNB(alpha=3)
clf3 = XGBClassifier()
clf4 = RandomForestClassifier()
clf5 = KNeighborsClassifier(n_neighbors=5)
clf5.fit(X_train,y_train)

accuracy_score(y_train,clf5.predict(X_train))

scores = cross_val_score(clf2, X_train, y_train, cv=10)
print scores, scores.mean()

for n in xrange(1,30):
    clf5 = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(clf5, X_train, y_train, cv=10)
    print scores.mean()






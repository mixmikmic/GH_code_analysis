import pandas as pd
import numpy as np
df1=pd.read_csv('E0.csv',usecols=['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HS','AS','HST','AST'
,'B365H','B365D','B365A'])
df1.head()

import matplotlib.pyplot as plt
plt.figure(figsize=(6,8))
plt.pie(df1['FTR'].value_counts(),labels=['Home','Away','Draw'], autopct='%1.1f%%',shadow=True, startangle=0)
plt.axis('equal')
plt.title('Win Percentage', size=18)
plt.show()

from statsmodels.stats import proportion
conf=proportion.proportion_confint((df1['FTR']=='H').sum(), df1['FTR'].count(), alpha=0.05, method='normal')
print('The chance of home team to win with %95 confidence interval falls in :{}'.format(conf))

##add new data to data frame
dfSq=pd.read_csv('dataE0.csv',index_col='Team').dropna(axis=0,how='any')
##Hde: Home Defense    Hatt: Home Attack    Hpo: Home possession    Htot : Home total power  
##Ade: Away defense   Aatt: away attack    Apo : Away possession :  Atot: Away total power
dfSq.head()
dff=df1.join(dfSq[['Hde','Hatt','Hpo','Htot']],on='HomeTeam')
df=dff.join(dfSq[['Ade','Aatt','Apo','Atot']],on='AwayTeam')

def make_data(df):
    ##add points for away and home team : win 3 points, draw 1 point, loss 0 point
    df['HP']=np.select([df['FTR']=='H',df['FTR']=='D',df['FTR']=='A'],[3,1,0])
    df['AP']=np.select([df['FTR']=='H',df['FTR']=='D',df['FTR']=='A'],[0,1,3])
    ## add difference in goals for home and away team
    df['HDG']=df['FTHG']-df['FTAG']
    df['ADG']=-df['FTHG']+df['FTAG']
    ##add momentum to data 
    cols=['Team','Points','Goal','Shoot','TargetShoot','DiffG']
    df1=df[['HomeTeam','AwayTeam','HP','AP','FTHG','FTAG','HS','AS','HST','AST','HDG','ADG']]
    df1.columns=[np.repeat(cols,2),['Home','Away']*len(cols)]
    d1=df1.stack()
    ##find momentum of previous five games for each team
    mom5 = d1.groupby('Team').apply(lambda x: x.shift().rolling(5, 4).mean())
    mom=d1.groupby('Team').apply(lambda x: pd.expanding_mean(x.shift()))
    ##add the found momentum to the dataframe
    df2=d1.assign(MP=mom5['Points'],MG=mom5['Goal'],MS=mom5['Shoot'],MST=mom5['TargetShoot'],MDG=mom5['DiffG'],AP=mom['Points'],AG=mom['Goal'],AS=mom['Shoot'],AST=mom['TargetShoot'],ADG=mom['DiffG']).unstack()
    df2=df2.drop(['Points','Goal','Shoot','TargetShoot','DiffG'],axis=1)
    df_final=pd.merge(df[['HomeTeam','AwayTeam','FTR','B365H','B365D','B365A','Ade','Aatt','Apo','Atot','Hde','Hatt','Hpo','Htot']],df2,left_on=['HomeTeam','AwayTeam'],right_on=[df2['Team']['Home'],df2['Team']['Away']])
    df_final=df_final.dropna(axis=0,how='any')
    ##Full time results ('FTR') : Home=0,Draw=1,Away=2
    Y_all=df_final['FTR']
    ##Full time results ('FTR') : Home=0,Draw=1,Away=2
    ##Prediction of betting company (bet365)=Y_Bet
    Y_Bet=df_final[['B365H','B365D','B365A']].apply(lambda x:1/x)
    ## winner based on bet365 data
    Y_Bet_FTR=np.select([Y_Bet.idxmax(axis=1)=='B365H',Y_Bet.idxmax(axis=1)=='B365D',Y_Bet.idxmax(axis=1)=='B365A'],['H','D','A'])
    ##scale data
    df_X=df_final.drop([('Team', 'Home'),('Team', 'Away'),'FTR','HomeTeam','AwayTeam','B365H','B365D','B365A'],axis=1)
    return df_X, Y_all,Y_Bet,Y_Bet_FTR
df_X, Y_all,Y_Bet,Y_Bet_FTR=make_data(df)

from sklearn.preprocessing import scale
X_all=scale(df_X)

from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import classification_report
def data_split(X_all, Y_all,Y_Bet_FTR,Y_Bet):
    X_train, X_test, y_train, y_test,y_train_bet_FTR,y_test_bet_FTR,y_train_bet,y_test_bet = train_test_split(X_all, Y_all,Y_Bet_FTR,Y_Bet, test_size=0.3, random_state=42) 
    return X_train, X_test, y_train, y_test,y_train_bet_FTR,y_test_bet_FTR,y_train_bet,y_test_bet  
def predict_labels(clf,X_test):
    y_pred=clf.predict(X_test)
    return y_pred   
def report_score(clf,X_test,y_test,y_pred,X_train,y_train):
    target_names = ['H', 'D', 'A']
    print(classification_report(y_test, y_pred, target_names=target_names))
    print ('{}....Test accuracy:{} Train accuracy:{}'.format(clf.__class__.__name__,clf.score(X_test,y_test),clf.score(X_train,y_train)))
def report_score_bet365(y_test,y_pred):
    target_names = ['H', 'D', 'A']
    print(classification_report(y_test, y_pred, target_names=target_names))
    print ('BET365 accuracy:{} '.format((y_test==y_pred).sum()/len(y_test)))
def train_classifier(clf,parameters,X_train,y_train):
    grid_class = GridSearchCV(clf,scoring='accuracy',param_grid=parameters)
    grid_class = grid_class.fit(X_train,y_train)
    clf = grid_class.best_estimator_
    return clf
clf_logistic= linear_model.LogisticRegression(multi_class = "ovr", solver = 'newton-cg', class_weight = 'balanced')
clf_svc = SVC(kernel="linear",probability=True)
clfs=[clf_logistic,clf_svc]
X_train, X_test, y_train, y_test,y_train_bet_FTR,y_test_bet_FTR,y_train_bet,y_test_bet=data_split(X_all, Y_all,Y_Bet_FTR,Y_Bet)
parameter_logistic = {'C': np.logspace(-5,5,100)}
parameter_SVC = {'C': np.arange(0.1,3,0.01)}
parameters={clfs[0]:parameter_logistic,clfs[1]:parameter_SVC}

for clf in clfs:
    clf=train_classifier(clf,parameters[clf],X_train,y_train)
    y_pred=predict_labels(clf,X_test)
    report_score(clf,X_test,y_test,y_pred,X_train,y_train)

report_score_bet365(y_test,y_test_bet_FTR)




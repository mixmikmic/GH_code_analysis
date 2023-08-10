import sys,os
PROJECT_ROOT=os.environ['HOME']+"\\PycharmProjects\\sports_data"
sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import numpy as np
from sites.common import sql_utils
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
import matplotlib.pyplot as plt
import re
get_ipython().magic('matplotlib inline')

scored_df=pd.read_csv("first_year_fantasy.csv")
scored_df.head(5)

DRAFT_KINGS_POSITIONS=["QB","RB","WR","TE"]#FB are boring

scored_df.columns

scored_df.groupby("position").describe()

scored_df.head(1)

names=['Sammy Watkins','Hakeem Nicks','Rueben Randle', 'Dwayne Bowe','Jordy Nelson']
cols=['name','height','weight','10_yd','forty_yd','vertical','broad_jump','shuttle']
scored_df['10_yd']='UNK'
scored_df[scored_df['name'].isin(names)][cols]#.set_index("name")#.ix[names]#[['height','weight','forty_yd','vertical','broad_jump','shuttle']]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(scored_df['height'], scored_df['weight'], zs=scored_df['dk_score'])

features=['position',
           'bench_reps','broad_jump','cone',
           'shuttle','forty_yd','vertical','weight','height']
target=["dk_score"]

from sklearn.preprocessing import scale
scored_df['scaled_forty']=scale(scored_df['forty_yd'])*scale(scored_df['draft_round'])
scored_df['pick_scaled_forty']=scale(scored_df['forty_yd'])*scale(scored_df['draft_pick'])

scored_df['density']=scored_df['height']/scored_df['weight']
scored_df['strength']=scored_df['bench_reps']/scored_df['weight']

from pandas.stats.api import ols
df=scored_df.copy()
#df=df.loc[(df>0).all(axis=1)]
col_subset=['weight',"height",'density','scaled_forty','pick_scaled_forty','forty_yd']
i=0

f, axes = plt.subplots(len(DRAFT_KINGS_POSITIONS), len(col_subset), sharey='row',figsize=(40,7*len(DRAFT_KINGS_POSITIONS)))
for position,pos_df in df.groupby("position"):
    if len(pos_df)<5:
        continue

    for idx,feature in enumerate(col_subset):        
        if "scaled" in feature:
            pos_df=pos_df[pos_df[feature]<10]#remove outliers
        axes[i][idx].scatter(x=pos_df[pos_df[feature]>0][feature],y=pos_df[pos_df[feature]>0]['dk_score'])
        axes[i][idx].set_title(position+" "+feature)
        
    axes[i][0].set_ylabel("Draft Kings Score")
    i+=1
f.suptitle("Rookie Fantasy Results vs Draft Information", fontsize=16)
f.show()

import seaborn as sns
i=0
for position,pos_df in scored_df.groupby("position"):
    if len(pos_df)<5:
        continue
    plt.figure(i)
    i+=1
    plt.title(position+ " Correlation Matrix")
    sns.heatmap(pos_df.ix[:, pos_df.columns!="position_label"].corr(), annot=True)

import xgboost as xgb

filtered_df=scored_df[(scored_df['dk_score']>0)&(scored_df['draft_round']>0)]
filtered_df.head(3)

from sklearn import preprocessing
from sklearn import metrics
def scoreXGB(data,params={'max_depth':3},new_features=[]):
    param = {'max_depth':3}#, "objective":"reg:logistic"}#change target to greater than the position average [0, 1] if using logistic
    num_rounds=50
    features=['position_label','weight','forty_yd',"height",'bench_reps','broad_jump','cone','shuttle','vertical']
    features+=new_features
    target="dk_score"
    le = preprocessing.LabelEncoder()
    le.fit(data['position'])
    data['position_label']=le.transform(data['position'])
    train_percent=.5
    msk = np.random.rand(len(data)) < train_percent
    train_df=data[msk]
    test_df=data[~msk]
    dtrain=xgb.DMatrix(train_df[features], label=train_df[target])
    dtest=xgb.DMatrix(test_df[features], label=test_df[target])
    bst = xgb.train(param, dtrain, num_rounds)
    #print(bst.get_score())
    trained_predictions=bst.predict(dtrain)
    test_predictions=bst.predict(dtest)
    train_accuracy=metrics.r2_score(dtrain.get_label(), trained_predictions)
    test_accuracy=metrics.r2_score(dtest.get_label(), test_predictions)
    return train_accuracy,test_accuracy
sample_runs=100
training_results=[]
for i in range(sample_runs):
    training_results.append(scoreXGB(filtered_df)+scoreXGB(filtered_df,new_features=['draft_pick','draft_round','density']))
training_results_df=pd.DataFrame(training_results,columns=["Combine Train","Combine Test", "With Draft Train","With Draft Test"])

training_results_df.describe()

import statsmodels.api as sm
for pos, data in filtered_df.groupby("position"):
    if len(data)<5:
        continue
    X=np.asarray(data[['weight','forty_yd',"height",'bench_reps','broad_jump','cone','shuttle','vertical','draft_pick','draft_round']])
    y=np.asarray(data['dk_score'])
    model = sm.OLS(y,X)
    results = model.fit()
    print(pos)
    #print(results.summary())
    print("R-squarerd", results.rsquared)
    plt.scatter(results.predict(),data['dk_score'])
    plt.title(pos)
    plt.xlabel("Predictions")
    plt.ylabel("Actaul")
    plt.show()




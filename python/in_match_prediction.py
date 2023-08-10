import numpy as np
import math
import time
import pandas as pd
from helper_functions import cross_validate
from sklearn import linear_model, ensemble
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

df_pred = pd.read_csv('../my_data/feature_df_pbp3_9_6_alphas.csv')

scores = ['sets_0','sets_1','games_0','games_1','points_0','points_1']
break_feats = [u'up_break_point', u'down_break_point', u'break_adv']
point_rates = [u'sv_points_pct_0', u'sv_points_pct_1']
cols = ['in_lead','elo_diff','s_elo_diff']+scores+break_feats+point_rates
c = int(max(df_pred['match_id'])*.8)
val_df,test_df = df_pred[df_pred['match_id']<=c],df_pred[df_pred['match_id']>c]

# for MLP, I would consider grid search over alpha, batch_size, and hidden_layer_sizes...
models = [linear_model.LogisticRegression(fit_intercept = True)]
model_hparams = {'LogisticRegression':{'C':[100]}}
baseline = ['lead_margin','elo_diff','s_elo_diff']
column_lists = [baseline,scores,baseline+scores,cols]

for columns in column_lists:
    print 'cols: ',columns
    Xtrain, ytrain = val_df[columns],val_df['winner']
    Xtest, ytest = test_df[columns],test_df['winner']
    for clf in models:
        model_name = clf.__str__().split('(')[0]
        print model_name
        best_hyper_p = cross_validate(val_df,clf,columns,'winner',model_hparams[model_name],n_splits=5)
        for key in best_hyper_p.keys():
            setattr(clf,key,best_hyper_p[key])
        clf.fit(Xtrain,ytrain)

        probs_train,probs_test = clf.predict_proba(Xtrain),clf.predict_proba(Xtest)
        train_loss, test_loss = log_loss(ytrain,probs_train,labels=[0,1]),log_loss(ytest,probs_test,labels=[0,1])
        train_accuracy = clf.score(Xtrain, ytrain)
        test_accuracy = clf.score(Xtest, ytest)  
        print train_loss, test_loss
        print train_accuracy, test_accuracy
    print '\n'

# graphically, try to compare different probabilities of different models
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

probs = np.concatenate((probs_train[:,1],probs_test[:,1]))
df_pred['prob4'] = probs

match1 = df_pred[df_pred['match_id']==1]
m1_probs = match1['prob4']
set_lengths = [len(a) for a in list(match1['score'])[-1].replace(';','').replace('/','').split('.')]
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(len(m1_probs)),m1_probs,'b-',linewidth=2.4)
fig.suptitle('Richard Gasquet d. Julian Reister 6-7, 6-3, 6-3')
ax.set_ylabel('Reister win probability')
ax.set_xlabel('# points played')
ax.text(2, 0.65, r'Reister wins first set 7-6')
ax.text(80, 0.25, r'Gasquet wins')
ax.text(80, 0.18, r'second set 6-3')
ax.text(133, 0.45, r'Gasquet wins')
ax.text(133, 0.38, r'match 6-7, 6-3, 6-3')
ax.axvline(set_lengths[0]); ax.axvline(sum(set_lengths[:2]))
plt.savefig('my_data/gasquet_reister_9_6_all_features')
fig.show()

probs = np.concatenate((probs_train[:,1],probs_test[:,1]))
df_pred['prob4'] = probs

match1 = df_pred[df_pred['match_id']==6310]
m1_probs = match1['prob4']
set_lengths = [len(a) for a in list(match1['score'])[-1].replace(';','').replace('/','').split('.')]
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(len(m1_probs)),m1_probs,'b-',linewidth=2.4)
fig.suptitle('Teymuraz Gabashvili d. Giles Simon 4-6, 6-4, 6-4')
ax.set_ylabel('Simon win probability')
ax.set_xlabel('# points played')
ax.axvline(set_lengths[0]); ax.axvline(sum(set_lengths[:2]))
#plt.savefig('simon_gabashvili_9_6_all_features')
fig.show()


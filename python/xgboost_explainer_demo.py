import xgboost as xgb
import numpy as np
import sys

import xgboost_explainer as xgb_exp

def sigmoid(x):
    return 1/(1+np.exp(-x))

dtrain = xgb.DMatrix('./train.libsvm')
dtest = xgb.DMatrix('./test.libsvm')
lmda = 1.0

params = {"objective":"binary:logistic", 'silent': 1, 'eval_metric': 'auc', 'base_score':0.5, "lambda":lmda}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
best_iteration = 42
bst = xgb.train(params, dtrain, best_iteration, watchlist)

feature_map = ["satisfaction_level","last_evaluation","number_project",
               "average_montly_hours","time_spend_company","Work_accident",
               "promotion_last_5years","sales","salary"]

tree_lst = xgb_exp.model2table(bst, lmda=lmda)
sample = dtest.slice([802])
print(bst.predict(sample))

leaf_lst = bst.predict(sample, pred_leaf=True)
dist = xgb_exp.logit_contribution(tree_lst, leaf_lst[0])
sum_logit = 0.0
for k in dist:
    sum_logit += dist[k]
    fn = feature_map[int(k[1:])] if k != "intercept" else k
    print(fn + ":", dist[k])
# print(sigmoid(sum_logit))

leaf_lsts = bst.predict(dtest, pred_leaf=True)
satisfaction_level_logit = []
last_evaluation_logit = []
for i,leaf_lst in enumerate(leaf_lsts):
    dist = xgb_exp.logit_contribution(tree_lst, leaf_lst)
    sum_logit = 0.0
    satisfaction_level_logit.append(dist['f0'])
    last_evaluation_logit.append(dist['f1'])

fp = open('./test.libsvm')
satisfaction_level_value = []
last_evaluation_value = []
for line in fp.readlines():
    arr = line.split()
    p = arr[1].split(':')
    assert p[0]=='0'
    satisfaction_level_value.append(float(p[1]))
    p = arr[2].split(':')
    assert p[0]=='1'
    last_evaluation_value.append(float(p[1]))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = (12, 8)

plt.scatter(satisfaction_level_value, satisfaction_level_logit)

plt.scatter(last_evaluation_value, last_evaluation_logit)




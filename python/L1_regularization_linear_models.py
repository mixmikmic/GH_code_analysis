import numpy as np
import matplotlib.pyplot as plt
import utls
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
utls.reset_plots()

d = pd.read_csv('../Data/prostate.csv', delimiter = '\t',index_col=0).reset_index(drop=True)
d.head()

d.train.unique()

features = list(d.columns)[:-2]

train = d[d['train']=='T']
test = d[d['train']=='F']

xtrain = utls.standardize(train[features]).as_matrix()
ytrain = utls.standardize(train['lpsa']).as_matrix()

xtest = utls.standardize(test[features]).as_matrix()
ytest = utls.standardize(test['lpsa']).as_matrix()

len(xtrain), len(xtest)

from sklearn.model_selection import KFold

regularizer_max = np.linalg.norm(np.dot(xtrain.T,ytrain),ord=np.inf) # find the maximum value of the regularizer (Eq 13.57 Murphy)
regularizer_max

alpha_space = np.logspace(np.log10(0.001),np.log10(regularizer_max))

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=1) # split the data into k consecutive folds

loss_alpha_arr = np.zeros((len(alpha_space),n_folds))
weight_counts = []
for i, alpha_val in enumerate(alpha_space):
    j=0
    for train_index, test_index in kf.split(xtrain):
        xtrain_k, x_test_k = xtrain[train_index], xtrain[test_index]
        ytrain_k, y_test_k = ytrain[train_index], ytrain[test_index]
        clf = Lasso(alpha=alpha_val)
        clf.fit(xtrain,ytrain)
        loss_ik = (clf.predict(x_test_k) - y_test_k)**2
        loss_alpha_arr[i,j] = loss_ik.sum()        
        weight_counts.append(clf.sparse_coef_.count_nonzero())
        j += 1
weight_counts = np.array(weight_counts)
    

mean_loss = loss_alpha_arr.mean(axis=1)
se_loss = loss_alpha_arr.std(axis=1,ddof=1)/np.sqrt(n_folds)

best_model_index = np.argmin(mean_loss)

best_model_ub_risk = mean_loss[best_model_index] + se_loss[best_model_index]

for i in range(len(alpha_space))[::-1]:
    if mean_loss[i] < best_model_ub_risk:
        optimal_model = i
        print(i)
        break

weight_counts[optimal_model]

mean_loss[optimal_model]

mean_loss[optimal_model - 1], best_model_ub_risk, mean_loss[optimal_model + 1]

coef_alpha = []

for alpha_val in alpha_space:
    clf = Lasso(alpha=alpha_val)
    clf.fit(xtrain,ytrain)
    coef_alpha.append(clf.coef_)
coef_alpha = np.array(coef_alpha)

coef_alpha[optimal_model]

fig, ax = plt.subplots(1,1)
for i, feat_name in enumerate(features):
    ax.plot(np.log10(alpha_space), coef_alpha[:,i],'.-',label=feat_name)

ax.plot(np.log10(alpha_space[optimal_model])*np.ones(50),np.linspace(-0.25,0.6),'-r', label = '5-fold CV, 1SE')
ax.legend(prop={'size':10})
ax.set_xlabel('Log regularizer, $\log_{10} \lambda$')
ax.set_ylabel('Feature weight');


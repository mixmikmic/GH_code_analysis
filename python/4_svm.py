get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

x = pd.read_csv('data/x_train.csv.gz', delimiter=';')
y = np.ravel(pd.read_csv('data/y_train.csv.gz', names=['target']))
test = pd.read_csv('data/x_test.csv.gz', delimiter=';')

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

clf = SVC(probability=True)
cross_val_score(clf, x_scaled, y, cv=5, scoring='neg_log_loss')

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],'cache_size': [2000], 'probability': [True], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
search = GridSearchCV(SVC(),param_grid=param_grid, scoring='neg_log_loss', cv=StratifiedKFold())

search.fit(x_scaled, y)

print(search.best_params_)
print(search.best_score_)

pred = search.predict_proba(test)[:,1]

submission = pd.DataFrame()
submission['target'] = pred
submission.to_csv("submissions/4_svm.csv", index=False, header=False)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (8,6)
plt.style.use('ggplot')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

loan_data = pd.read_csv("data/loan_data_cleaned.csv")

loan_data.head()

X = loan_data.iloc[:7500,1:]
y = loan_data.iloc[:7500,0]

model = RandomForestClassifier(n_estimators=100)
model = model.fit(X, y)

X_test = loan_data.iloc[7500:,1:]
y_test = loan_data.iloc[7500:,0]

from sklearn.metrics import accuracy_score
print accuracy_score(model.predict(X_test),y_test)

importances = model.feature_importances_
#Calculate the standard deviation of variable importance
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
length = X.shape[1]
labels = []
for i in range(length):
    labels.append(X.columns[indices[i]])
# Plot the feature importances of the forest
plt.figure(figsize=(16, 6))
plt.title("Feature importances")
plt.bar(range(length), importances[indices], yerr=std[indices], align="center")
plt.xticks(range(length), labels)
plt.xlim([-1, length])
plt.show()

import eli5

y_test.iloc[5]

eli5.show_prediction(model, X_test.iloc[5,:], show_feature_values=True)

eli5.show_prediction(model, X_test.iloc[16,:], show_feature_values=True)

import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X.as_matrix(), feature_names=('amount', 'grade', 'years', 'ownership', 'income', 'age'))

exp = explainer.explain_instance(X_test.iloc[5,:].as_matrix(), model.predict_proba)
exp.show_in_notebook(show_table=True, show_all=True)

y_test.iloc[5]

exp = explainer.explain_instance(X_test.iloc[16,:].as_matrix(), model.predict_proba)
exp.show_in_notebook(show_table=True, show_all=True)

y_test.iloc[16]


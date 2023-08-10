import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# load and explore the data

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# The data set is presented in a dictionary form
cancer.keys()

print(cancer['DESCR'])

cancer['feature_names']

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()

df_feat.head()

df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
df_target.info()

df_target.head()

# now have 2 dataframes-df_feat and df_target

cancer['target']

X=df_feat
y=cancer['target']

# train the data and fit the model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

# model evaluation

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# tune the perameter for better result

from sklearn.model_selection import GridSearchCV
# let sklearn choose the best value for me.
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]} 



grid = GridSearchCV(SVC(),param_grid)

# May take awhile!
grid.fit(X_train,y_train)


grid.best_params_

grid.best_estimator_

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))


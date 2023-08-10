import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("winequality-red.csv")
df.head()

df.shape

df.describe()

from collections import Counter
Counter(df["quality"])

df["quality_bin"] = np.zeros(df.shape[0])

df["quality_bin"] = df["quality_bin"].where(df["quality"]>=6, 1)
#1 means good quality and 0 means bad quality

df.head()

Counter(df["quality_bin"])

#No missing data

#feature scaling

from sklearn.preprocessing import StandardScaler

X_data = df.iloc[:,:11].values
y_data = df.iloc[:,12].values

scaler = StandardScaler()

X_data = scaler.fit_transform(X_data)

#train test splitting

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

print("y_train: ",Counter(y_train),"\n",
      "y_test: ",Counter(y_test))

#balancing the data

from imblearn.over_sampling import SMOTE

#resampling need to be done on training dataset only
X_train_res, y_train_res = SMOTE().fit_sample(X_train, y_train)

Counter(y_train_res)

#model selection 
from sklearn.linear_model import SGDClassifier
sg = SGDClassifier(random_state=42)

sg.fit(X_train_res,y_train_res)

pred = sg.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test, pred))

accuracy_score(y_test, pred)

#parameter tuning 
from sklearn.model_selection import GridSearchCV

model = SGDClassifier(random_state=42)
params = {'loss': ["hinge", "log", "perceptron"],
          'alpha':[0.001, 0.0001, 0.00001]}

clf = GridSearchCV(model, params)

clf.fit(X_train_res, y_train_res)

clf.best_score_

clf.best_estimator_

clf.best_estimator_.loss

clf.best_estimator_.alpha

#final model by taking suitable parameters

clf = SGDClassifier(random_state=42, loss="hinge", alpha=0.001)

clf.fit(X_train_res, y_train_res)

pred = clf.predict(X_test)

print(classification_report(y_test, pred))

accuracy_score(y_test, pred)

#saving model into a pickle file for later use

from sklearn.externals import joblib
joblib.dump(clf, "wine_quality_clf.pkl")

###
clf1 = joblib.load("wine_quality_clf.pkl")

X_test[0]

clf1.predict([X_test[0]])




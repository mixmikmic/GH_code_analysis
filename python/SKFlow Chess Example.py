import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics
import pandas as pd

df = pd.read_csv("data/krkopt.data.csv")
df.head()



# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
feature_keys = ['kwc', 'kwr', 'rwc', 'rwr', 'kbc', 'kbr']

features = list()
for key in feature_keys:
    features.append(pd.get_dummies(df[key], prefix=key))

features_df = pd.concat(features, axis=1)

features_df.head()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['result'])
y = le.transform(df['result'])



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.33, random_state=42)

myClassifier = skflow.TensorFlowLinearClassifier(18)
myClassifier.fit(X_train, y_train)

score = metrics.accuracy_score(myClassifier.predict(X_test), y_test)
print("Accuracy: %f" % score)

import skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()












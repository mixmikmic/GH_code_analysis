import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import pickle

# get the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
df = pd.read_csv(url, names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
df.head()

X = df.values[:, 0:8]
Y = df.values[:, 8]
X

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=6)

model = LogisticRegression()
model.fit(X_train, Y_train)

pickle.dump(model, open('saved_model.pickle', 'wb'))

trained_model = pickle.load(open('saved_model.pickle', 'rb'))
result = trained_model.score(X_test, Y_test)
print(result)


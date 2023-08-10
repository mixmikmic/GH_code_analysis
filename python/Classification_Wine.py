import os, sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

datasource = "datasets/winequality-red.csv"
print(os.path.exists(datasource))

df = pd.read_csv(datasource).sample(frac = 1).reset_index(drop = True)

df.head()

del df["Unnamed: 0"]

df.head()

print(df.shape)

X = np.array(df.iloc[:, :-1])[:, [1, 2, 6, 9, 10]]
# remove the quality column
# only grab the following columns:
# volatile acidity, citric acid, total sulfur dioxide
# sulphates, alcohol

y = np.array(df["quality"])

print(X.shape)

print(y.shape)

print("Label Distribution", {i: np.sum(y == i) for i in np.unique(df["quality"])})

df.describe()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
m = GaussianNB()
m.fit(X_train, y_train)
m.score(X_test, y_test)

actual = y[20:50]
actual

pred = m.predict(X[20:50])
pred

pred = np.round(m.predict(X_test)).astype("i4")

# remember the labels from before:
# Label Distribution {3: 10, 4: 53, 5: 681, 6: 638, 7: 199, 8: 18}

labels = [3, 4, 5, 6, 7, 8]

cm = confusion_matrix(y_test, pred, labels)
cm = pd.DataFrame(cm).reset_index(drop = True)
cm.columns = labels
cm.index = labels
cm

f1_score(y_test, pred, average = "micro")




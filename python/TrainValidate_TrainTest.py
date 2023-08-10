import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from collections import OrderedDict
from sklearn.externals import joblib

raw = load_iris()

rawData = raw["data"]

rawData[0:5]

X = rawData[:, 0:2]
# all the rows and first 2 columns

X[:5]

rawTarget = raw["target"]

rawTarget[0:5]

y = rawTarget

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

m = GaussianNB()

scores = cross_val_score(m, X_train, y_train, cv = 5)
print(scores)

fitted = m.fit(X_train, y_train)

labels = m.predict(X_test)

labels

colors = ["red", "blue", "green"]

# iterate over the labels and assign a color to each point
for i in range(0, len(X_test)):
    col = colors[labels[i]]
    plt.plot(X_test[:, 0][i], X_test[:, 1][i], color = col, marker = "o", 
                markersize = 5, label = "Class %i" % labels[i])

# cut out duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc = "upper right")
plt.show()

from sklearn.externals import joblib

joblib.dump(m, "GaussianIris.pkl")

loaded_model = joblib.load("GaussianIris.pkl")

labels = loaded_model.predict(X_test)

labels

colors = ["red", "blue", "green"]

for i in range(0, len(X_test)):
    col = colors[labels[i]]
    plt.plot(X_test[:,0][i], X_test[:, 1][i], color = col, marker = "o", 
                    markersize = 5, label = "Class %i" % labels[i])
    
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc = "upper right")
plt.show()


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

dataset = datasets.load_breast_cancer()
dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state = 0)

clf = KNeighborsClassifier()
clf.fit(X_train, Y_train)

clf.score(X_test, Y_test)

for i in range(1, 26, 2):
    clf = KNeighborsClassifier(n_neighbors = i)
    score = cross_val_score(clf, X_train, Y_train)
    print(i, score.mean())

x_axis = []
y_axis = []
for i in range(1, 26, 2):
    clf = KNeighborsClassifier(n_neighbors = i)
    score = cross_val_score(clf, X_train, Y_train)
    x_axis.append(i)
    y_axis.append(score.mean())
    

import matplotlib.pyplot as plt
plt.plot(x_axis, y_axis)
plt.show()




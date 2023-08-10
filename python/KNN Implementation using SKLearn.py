from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

data = datasets.load_breast_cancer()

x_train,x_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2,random_state=1) 

x_axis =[]
y_axis =[]
for i in range(1,28,2):
    clf = KNeighborsClassifier(n_neighbors=i)
    scr = cross_val_score(clf,data.data,data.target)
    x_axis.append(i)
    y_axis.append(scr.mean())

import matplotlib.pyplot as plt
plt.plot(x_axis,y_axis)
plt.show()

clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(x_train,y_train)

clf.score(x_test,y_test)

cross_val_score(clf,data.data,data.target,cv=KFold(5,True,random_state=0))


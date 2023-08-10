from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from collections import Counter

data = datasets.load_breast_cancer()

x_train,x_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2,random_state=1) 

clf = KNeighborsClassifier(n_neighbors=10)
cross_val_score(clf,data.data,data.target)

clf.fit(x_train,y_train)

clf.score(x_test,y_test)

def predict_one(x_train,y_train,x_test,k):
    distances = []
    for i in range(len(x_train)):
        distance = ((x_train[i,:]-x_test)**2).sum()
        distances.append([distance,i])
    distances = sorted(distances)
    target = []
    for i in range(k):
        index_of_training = y_train[distances[i][1]]
        target.append(index_of_training)
    res = Counter(target).most_common(1)[0][0]
    return res
def predict(x_train,y_train,x_test_data,k):
    predictions =[]
    for x_test in x_test_data:
        predictions.append(predict_one(x_train,y_train,x_test,k=10))
    return predictions

pred = predict(x_train,y_train,x_test,10)
accuracy_score(y_test,pred)


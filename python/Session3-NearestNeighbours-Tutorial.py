import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from collections import Counter

get_ipython().run_line_magic('matplotlib', 'inline')

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data.txt', header=None, names=names)
df.head()

fig, ax = plt.subplots()
colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}
ax.scatter(df['sepal_length'], df['sepal_width'], c=df['class'].apply(lambda x: colors[x]))

plt.show()

_, bx = plt.subplots()
bx.scatter(df['petal_length'], df['petal_width'], c=df['class'].apply(lambda x: colors[x]))
plt.show()

# create design matrix X and target vector y
X = np.array(df.iloc[:, 0:4]) 	# end index is exclusive
y = np.array(df['class']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#print(X)

#print(y)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
acc = accuracy_score(y_test, pred) * 100
print('\nAccuracy of scikit-learn knn classifier for k = 3 is %d%%' % acc)

neighbors = list(range(1,50,2))
print(neighbors)

# empty list that will hold cross-validation scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)

# plot misclassification error vs k 
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

def train(X_train, y_train):
    # does nothing, its a dummy function 
    return

def predict(X_train, y_train, x_test, k):
    # create 2 Lists2 store distances and targets
    # Targets are the associated labels of the k-Nearest Neighbours
    distances = []
    targets = []

    for i in range(len(X_train)):
        # first we compute the Euclidean Distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # Perform a majority vote using a Counter. Return most common target
    return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
    # check if k larger than n
    if k > len(X_train):
        raise ValueError

    # train on the input data
    train(X_train, y_train)

    # predict for each testing observation
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))

## Running our code on the testing set

# making our predictions 
predictions = []
try:
    kNearestNeighbor(X_train, y_train, X_test, predictions, 7)
    predictions = np.asarray(predictions)

    # evaluating accuracy
    accuracy = accuracy_score(y_test, predictions) * 100
    print('Accuracy Score: %d%%' % accuracy)

except ValueError:
    print('Can\'t have more neighbors than training samples!!')




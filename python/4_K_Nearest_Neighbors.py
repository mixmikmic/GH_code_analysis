import numpy as np
import operator

class KNearestNeighbors():

    def __init__(self, k, model_type='regression', weights='uniform'):

        # model_type can be either 'classification' or 'regression'
        # weights = 'uniform', the K nearest neighbors are equally weighted
        # weights = 'distance', the K nearest entries are weighted by inverse of the distance
        self.model_type = model_type
        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def _dist(self, example1, example2):

        # calculate euclidean distance between two examples
        if len(example1) != len(example2):
            print "Inconsistent Dimension!"
            return

        return np.sqrt(sum(np.power(np.array(example1) - np.array(example2), 2)))

    def _find_neighbors(self, test_instance):

        # find K nearest neighbors for a test instance
        # this function return a list of K nearest neighbors for this test instance,
        # each element of the list is another list of distance and target
        m, n = self.X_train.shape
        neighbors = [[self._dist(self.X_train[i, :], test_instance), self.y_train[i]]
                     for i in range(m)]
        neighbors.sort(key=lambda x: x[0])
        return neighbors[:self.k]

    def fit(self, X, y):

        # no parameters learning in model fitting process for KNN
        # just to store all the training instances
        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):

        # predict using KNN algorithm
        X = np.array(X)

        # if only have one test example to predict
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        m = X.shape[0]
        y_predict = np.zeros((m, 1))

        # for regression problems, depending on the weights ('uniform' or 'distance'),
        # it will perform average or weighted average based on inverse of distance
        if self.model_type == 'regression':
            for i in range(m):
                distance_mat = np.array(self._find_neighbors(X[i, :]))
                if self.weights == 'distance':
                    y_predict[i] = np.average(distance_mat[:, 1], weights=1.0/distance_mat[:, 0])
                else:
                    y_predict[i] = np.average(distance_mat[:, 1])

        # for classification, we will apply majority vote for prediction
        # it still offer two options in terms of the weights
        else:
            for i in range(m):
                votes = {}
                distance_mat = np.array(self._find_neighbors(X[i, :]))
                for j in range(self.k):
                    if self.weights == 'distance':
                        votes[distance_mat[j, 1]] = votes.get(distance_mat[j, 1], 0)                                                     + 1.0 / distance_mat[j, 0]
                    else:
                        votes[distance_mat[j, 1]] = votes.get(distance_mat[j, 1], 0) + 1.0
                sorted_votes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
                y_predict[i] = sorted_votes[0][0]

            y_predict = y_predict.astype(int)

        return y_predict.ravel()

from sklearn.datasets import load_iris
iris = load_iris()

X = iris['data']
y = iris['target']

print X.shape
print y.shape
print "Number of Classes: {}".format(len(np.unique(y)))

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

knn = KNearestNeighbors(k=3, model_type='classification', weights='uniform')
knn = knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print "True Values:      {}".format(y_test)
print "Predicted Values: {}".format(y_predict)
print "Prediction Accuracy: {:.2%}".format(np.mean((y_predict == y_test).astype(float)))

from sklearn.datasets import load_boston

boston = load_boston()

X = boston['data']
y = boston['target']

print X.shape
print y.shape

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.025, random_state=26)

knn = KNearestNeighbors(k=20, model_type='regression', weights='uniform')
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print "True Values:\n{}".format([round(elem, 1) for elem in y_test])
print "Predicted Values:\n{}".format([round(elem, 1) for elem in y_predict])
print "RMSE is {:.4}".format(np.sqrt(np.mean((y_test.reshape((len(y_test), 1)) - y_predict) ** 2)))


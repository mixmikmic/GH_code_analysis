get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition, datasets

# load data
digits = datasets.load_digits()
X = digits.data
y = digits.target

# load PCA
pca = decomposition.PCA()

# fit it! remember that this is an unsupervised technique so we're not passing in y
pca.fit(X)

# plot it
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()

# let's use those components to fit our logistic regression
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train the dimensionality reduction for 20 output features
new_pca = decomposition.PCA(n_components=20)

# the dimensionality reduction can be trained on the entire X dataset
# just be careful never to use y_test in a train setting
new_pca.fit(X)
new_X_train = new_pca.transform(X_train)
new_X_test = new_pca.transform(X_test)

# let's train a logistic regressor
logistic = linear_model.LogisticRegression()
logistic.fit(new_X_train, y_train)

predicted = logistic.predict(new_X_test)

from sklearn import metrics
print("Classification report for classifier %s:\n%s\n"
      % (logistic, metrics.classification_report(y_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))




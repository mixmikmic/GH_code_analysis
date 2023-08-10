from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

for i in ['full', 'tied', 'diag', 'spherical']:
    clf = GaussianMixture(covariance_type= i)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print('For covariance_type = ', i, 'the test accuracy = ' + str(accuracy_score(y_test, y_predict)))

clf = GaussianMixture(n_components=3, covariance_type='full')  

clf.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                for i in range(3)])
# fit
clf.fit(X_train, y_train)
# predict
pred_train = clf.predict(X_train)
pred = clf.predict(X_test)
#evaluate
print ('Train accuracy = ' + str(accuracy_score(y_train, pred_train)))
print ('Test accuracy = ' + str(accuracy_score(y_test, pred)))

colors = ['navy', 'turquoise', 'darkorange']

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

estimator = GaussianMixture(n_components=3,
                   covariance_type='spherical', max_iter=20, random_state=0)

estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                for i in range(3)])

# Fit
estimator.fit(X_train, y_train)

# Plot
plt.figure(figsize=(10,10))
plt.ylim([-1,3])
plt.xlim([11,15])
plt.xlabel('Alcohol', fontsize=15)
plt.ylabel('Hue', fontsize=15)
h = plt.subplot()
make_ellipses(estimator, h)

# Plot train data with dots
for n, color in enumerate(colors):
    train_data = X_train[y_train == n]
    plt.scatter(train_data['Alcohol'], train_data['Hue'], s=10, color=color)

# Plot the test data with crosses
for n, color in enumerate(colors):
    test_data = X_test[y_test == n]
    plt.scatter(test_data['Alcohol'], test_data['Hue'], marker='x', color=color)

plt.title('Gaussian Mixture Model', fontsize=15)

plt.show()




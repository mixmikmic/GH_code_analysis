import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1./(1.+np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.figure(figsize=(8, 6))
plt.plot(z, phi_z, c='orange') # sigmoid line
plt.axvline(0., color='k') # vertical line at 0
plt.axhspan(0., 1., alpha=1, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.show()

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
np.unique(y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0 )
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression( C=1000.0, random_state=0) # C is regularization parameter
lr.fit(X_train_std, y_train)

from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

X_combined_std = np.vstack( (X_train_std, X_test_std) )
y_combined = np.hstack( (y_train, y_test) )
plot_decision_regions( X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105,150) )
plt.xlabel('petal length [std.]')
plt.ylabel('petal width [std.]')
plt.legend(loc='upper left')
plt.show()

lr.predict_proba(X_test_std[0,:])

weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit( X_train_std, y_train)
    weights.append( lr.coef_[1] )
    params.append(10.**c)
weights = np.array(weights)

plt.figure(figsize=(8, 6))
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], label='petal width', linestyle='--')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()




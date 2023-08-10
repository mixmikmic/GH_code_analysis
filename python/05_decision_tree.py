import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().magic('matplotlib inline')

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');

from myfun import visualize_tree

from sklearn.tree import DecisionTreeClassifier
max_depth = 15
clf = DecisionTreeClassifier(max_depth=max_depth)

plt.figure()
visualize_tree(clf, X[:200], y[:200], boundaries=True)

from sklearn.tree import DecisionTreeClassifier
max_depth = 4
clf = DecisionTreeClassifier(max_depth=max_depth)
clf.fit(X,y)

plt.figure()
visualize_tree(clf,X[:200], y[:200])
plt.title("max_depth="+str(max_depth))

clf.feature_importances_

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=2.0)
max_depth = 15
clf = DecisionTreeClassifier(max_depth=max_depth)

visualize_tree(clf, X[:250], y[:250], boundaries=True)
plt.title("max_depth="+str(max_depth))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

get_ipython().magic('pinfo RandomForestClassifier')

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth= 8,random_state=0)
visualize_tree(clf, X, y, boundaries=False);

from sklearn.ensemble import RandomForestRegressor

x = 10 * np.random.rand(100)

def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * np.random.randn(len(x))

    return slow_oscillation + fast_oscillation + noise

y = model(x)
plt.errorbar(x, y, 0.3, fmt='o');

xfit = np.linspace(0, 10, 1000)
yfit = RandomForestRegressor(100).fit(x[:, None], y).predict(xfit[:, None])
ytrue = model(xfit, 0)

plt.errorbar(x, y, 0.3, fmt='o')
plt.plot(xfit, yfit, '-r');
plt.plot(xfit, ytrue, '-k', alpha=0.5);


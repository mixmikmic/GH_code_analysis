import numpy as np
rng = np.random.RandomState(42)

X = np.sort(5 * rng.rand(100, 1), axis=0)
y = np.sin(X).ravel()

y[::2] += 0.5 * (0.5 - rng.rand(50))

from sklearn import tree

regr1 = tree.DecisionTreeRegressor(max_depth=2, random_state=42)
regr1.fit(X, y)

regr2 = tree.DecisionTreeRegressor(max_depth=5, random_state=42)
regr2.fit(X, y)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

y_1 = regr1.predict(X_test)
y_2 = regr2.predict(X_test)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='k', s=50, label='data')
plt.plot(X_test, y_1, label="max_depth=2", linewidth=5)
plt.plot(X_test, y_2, label="max_depth=5", linewidth=3)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()


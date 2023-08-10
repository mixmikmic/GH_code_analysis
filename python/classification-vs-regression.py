get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

df = pd.DataFrame({
    'cooking time (h)': [0.2, 0.5, 0.6, 0.7, 1.0, 1.2, 1.3, 1.5, 1.9, 2.0, 2.5, 3.0, 3.7, 4.5],
    'satisfied eater':  [  0,   0,   0,   1,   0,   1,   0,   0,   1,   1,   1,   1,   1,   1]})
df

model = LogisticRegression()
data = df['cooking time (h)'].values[:, np.newaxis]
y = df['satisfied eater']
cross_val_score(model, data, y)

model.fit(data, y)
model.predict([[0.3], [2.6]]), model.predict_proba([[0.3], [2.6]])

xfit = np.linspace(0, 5, 100)
yfit = model.predict_proba(xfit[:, np.newaxis])
fig, ax = plt.subplots(1, figsize=(14, 6))
ax.plot(xfit, yfit[:, 0], color='crimson')
ax.plot(xfit, yfit[:, 1], color='steelblue')
ax.plot(data, y, 'o', color='black');

weight = np.linspace(0, 10, 10) + np.random.rand(10)
size = np.linspace(0, 10, 10) + np.random.rand(10)
w, s = np.meshgrid(weight, size)
lift_foundation = w*3 + s*2 + 17 + 7*np.random.rand(10)
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w, s, lift_foundation)
data = np.dstack([w, s]).reshape(100, 2)

model = KNeighborsRegressor(n_neighbors=3)
# R2 scoring is particularly bad for KNN regression
grid = GridSearchCV(model, {'n_neighbors': [3, 4, 5, 6, 7]}, scoring='explained_variance')
grid.fit(data, lift_foundation.reshape(100))
print(grid.best_score_)
grid.best_estimator_

xmesh, ymesh = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
xfit = np.dstack([xmesh, ymesh]).reshape(10000, 2)
model = grid.best_estimator_
yfit = model.fit(data, lift_foundation.reshape(100))
yfit = model.predict(xfit)
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xmesh, ymesh, yfit.reshape(100, 100));


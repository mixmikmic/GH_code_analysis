get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ignore this code, it is just to draw the figure below
fig, ax = plt.subplots(figsize=(9, 6))
ax.axis('off')
ax.axis('equal')
ax.vlines(range(4), ymin=0, ymax=6, lw=1)
ax.hlines(range(7), xmin=0, xmax=3, lw=1)
ax.vlines(range(4, 6), ymin=0, ymax=6, lw=1)
ax.hlines(range(7), xmin=4, xmax=5, lw=1)
ax.text(1.5, 6.5, "input / data / X", size=14, ha='center')
ax.text(4.6, 6.5, "target / y", size=14, ha='center')
ax.text(0, -0.3, r'features $\longrightarrow$')
samples = r'$\longleftarrow$ samples'
ax.text(0, 6, samples, rotation=90, va='top', ha='right')
ax.text(5.1, 6, samples, rotation=90, va='top', ha='left')
ax.set_ylim(-1, 7);

class MyKNN(object):
    def __init__(self, k=1):
        self.k_ = k

    def fit(self, data, y):
        self.data = data
        self.y = y

    def predict(self, new_data):
        sq_dist_dim = (new_data[:, np.newaxis, :] - self.data[np.newaxis, :, :])**2
        sq_dist = sq_dist_dim.sum(axis=-1)
        self.nearest_ = np.argpartition(sq_dist, self.k_, axis=1)
        new_y = self.y[self.nearest_[:, :self.k_]]
        new_y = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=new_y)
        return new_y.argmax(axis=1)  # not really correct for even k, hides ties

random_state = 27
rng = np.random.RandomState(random_state)
data, y = make_blobs(30, 2, centers=2, cluster_std=0.2, center_box=(0, 1), random_state=random_state)
new_data = rng.rand(6, 2)
knn_left = MyKNN(k=1)
knn_left.fit(data, y)
new_yl = knn_left.predict(new_data)
knn_right = MyKNN(k=5)
knn_right.fit(data, y)
new_yr = knn_right.predict(new_data)
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].set_title('k =  1')
ax[0, 0].scatter(data[:, 0], data[:, 1], s=50, c=y, cmap='viridis')
ax[0, 0].scatter(new_data[:, 0], new_data[:, 1], s=200)
ax[1, 0].scatter(data[:, 0], data[:, 1], s=50, c=y, cmap='viridis')
ax[1, 0].scatter(new_data[:, 0], new_data[:, 1], s=200, c=new_yl, cmap='viridis')
ax[0, 1].set_title('k =  5')
ax[0, 1].scatter(data[:, 0], data[:, 1], s=50, c=y, cmap='viridis')
ax[0, 1].scatter(new_data[:, 0], new_data[:, 1], s=200)
ax[1, 1].scatter(data[:, 0], data[:, 1], s=50, c=y, cmap='viridis')
ax[1, 1].scatter(new_data[:, 0], new_data[:, 1], s=200, c=new_yr, cmap='viridis')
for axi in ax.flat:
    axi.axis('equal')
new_yl, new_yr

knn = MyKNN(k=1)
knn.fit(data, y)
new_y = knn.predict(data)
error = (y - new_y)**2
error.mean()

knn = MyKNN(k=5)
knn.fit(data, y)
new_y = knn.predict(data)
error = (y - new_y)**2
error.mean()

data1, data2, y1, y2 = train_test_split(data, y, random_state=random_state, train_size=0.7, test_size=0.3)
accuracy = []
f1_scores = []
k_vals = list(range(1, 16))
for k in k_vals:
    knn = MyKNN(k=k)
    knn.fit(data1, y1)
    new_y = knn.predict(data2)
    accuracy.append(accuracy_score(y2, new_y))
    f1_scores.append(f1_score(y2, new_y))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(k_vals, accuracy, color='crimson')
ax.plot(k_vals, f1_scores, color='steelblue');


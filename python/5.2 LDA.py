import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# load wine data
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
X, y = df_wine.iloc[:, 1:14].values, df_wine.iloc[:, 0].values

# Normalize input
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# 1.calculate mean vectors
np.set_printoptions(precision=4)
mean_vecs = []
labels = np.unique(y_train)
for label in labels:
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))

# print mean vector
for i in range(len(mean_vecs)):
    print "MeanVector[%d] ---> \n" %(i), mean_vecs[i]

# 2.Use mean vectors and compute within-class scatter matrix (S_W)
# num_features
d = X_train_std.shape[1]

S_W = np.zeros((d,d))
for label, mv in zip(labels, mean_vecs):
    class_scatter = np.zeros((d,d))
    for row in X[y== label]:
        row = row.reshape(d,1)
        mv = mv.reshape(d,1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter
print S_W.shape

print "Class Distribution:", np.bincount(y_train)[1:]

# num_features
d = X_train_std.shape[1]

S_W = np.zeros((d,d))

for label, mv in zip(labels, mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print S_W.shape

mean_overall = np.mean(X_train_std, axis=0).reshape(d,1)
S_B = np.zeros((d,d))
for label, mv in zip(labels, mean_vecs):
    n = X[y==label].shape[0]
    mv = mv.reshape(d,1)
    S_B += n* (mv - mean_overall).dot((mv - mean_overall).T)

print S_B.shape

# 1. Compute eigen vector
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W1).dot(S_B1))

# 2. Sort and print the eigenvalues in descending order
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[i]) for i in range(0, len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key= lambda k: k[0], reverse=True)
for row in eigen_pairs:
    print row[0]

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

# Let's now stack 2 most discriminative eigenvector columns to create weight matrix
# or transformation matrix
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
X_train_lda = X_train_std.dot(w)

# If we plot X_train_lda, we can make out that its now linearly separable.

from sklearn.lda import LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

# plot X_train_lda to verify that its now linearly separable
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l,c,m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1], c=c, label=l, marker=m)

plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend(loc="upper right")
plt.title("Plt transformed training data")
plt.show()

# fit X_train_lda with logisticRegression
from sklearn.linear_model import LogisticRegression
get_ipython().magic("run 'plot_decision_regions.ipynb'")
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.title("Training Data classified with LDA")
plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.title("Test data classified with LDA")
plt.show()


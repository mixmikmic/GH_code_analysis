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

print X_train_std.shape

# calculate covariance
covariance_matrix = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)

# calculate variance_explained_ratio (is equal to eigen_val[i]/sum_of_eigen_values)
tot = np.sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse = True)]

# calculate cumulative sum of var_exp
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label = "individual explained variance")
plt.step(range(1,14), cum_var_exp, label='cumulative explained variance')
plt.legend(loc='best')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('This graph is pretty similar to feature importance graph that we plotted in randomforest.\n It mainly tells us that first principal component alone accounts for 40 percent of the variance.\n Also, we can see that the first two principal components combined explain almost 60 percent of the variance in the data')
plt.show()

# calculate weight matrix using eigen vals:
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print 'Projection Weight Matrix [W] =\n', w

X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m) 

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.title('we can intuitively see that a linear classifier will likely be able to separate the classes well')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

get_ipython().magic("run 'plot_decision_regions.ipynb'")

# apply pca and logReg
pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)

# plot training data
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc = 'best')
plt.title('training data classified with logistic regression and PCA')
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc = 'best')
plt.title('testing data classified with logistic regression and PCA')
plt.show()

# If we are interested in the explained variance ratios of the different principal components,
# we can simply initialize the PCA class with the n_components parameter set to None, 
# so all principal components are kept and the explained variance ratio can then be accessed
# via the explained_variance_ratio_ attribute:

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('precision', '3')
np.set_printoptions(precision=3)
import pylab as pl

x1 = np.arange(0,10)
y1 = np.arange(10,0,-1)

plt.plot(x1,y1)

np.cov([x1,y1])

x2 = np.arange(0,10)
y2 = np.array([2]*10)
plt.plot(x2,y2)

cov_mat = np.cov([x2,y2])
cov_mat

x3 = np.array([2]*10)
y3 = np.arange(0,10)
plt.plot(x3,y3)

np.cov([x3,y3])

iris = load_iris()

iris_df = pd.DataFrame(iris.data,columns=[iris.feature_names])
iris_df.head()

X = iris.data

X.shape

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
print X_std[0:5]
print "The shape of Feature Matrix is -",X_std.shape

X_covariance_matrix = np.cov(X_std.T)

X_covariance_matrix

eig_vals, eig_vecs = np.linalg.eig(X_covariance_matrix)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print "Variance captured by each component is \n",var_exp
print(40 * '-')
print "Cumulative variance captured as we travel each component \n",cum_var_exp

print "All Eigen Values along with Eigen Vectors"
pprint.pprint(eig_pairs)
print(40 * '-')
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print 'Matrix W:\n', matrix_w

Y = X_std.dot(matrix_w)
print Y[0:5]

pl.figure()
target_names = iris.target_names
y = iris.target
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    pl.scatter(Y[y==i,0], Y[y==i,1], c=c, label=target_name)
pl.xlabel('Principal Component 1')
pl.ylabel('Principal Component 2')
pl.legend()
pl.title('PCA of IRIS dataset')
pl.show()


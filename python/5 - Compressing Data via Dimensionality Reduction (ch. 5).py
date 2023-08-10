## Wine dataset
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', 
                      header=None)

## Split into train/test sets, 70/30 split respectively
### and standardize it to unit variance
#Step 1
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test =              train_test_split(X, y,
             test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

## Constructing the covariance matrix
# step 2
import numpy as np
cov_mat = np.cov(X_test_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print '\nEigenvalues \n%s' % eigen_vals

## sum of explained variance
## using cumsum (cumulative sum)
### then plot
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
       label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',
        label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

## sort eigenpairs by decreasing order of the eigenvalues
eigen_pairs = [(np.abs(eigen_vals[i], eigen_vecs[:,i])
               for i in range(len(eigen_vals)))]
eigen_pairs.sort(reverse=True)

## here we will choose top two eigenvectors(for illustration),
## that correspond to the larges values to capture, ~60% of 
## the variance in this dataset
####
# IN PRACTICE: the number of Pricipal Components has to be 
# determined from a trade-off between computational efficiency 
# and the performance of the classifier

# w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
#               eigen_pairs[1][1][:,np.newaxis]))
# print 'Matrix W:\n', w

w= np.hstack((eigen_pairs[0][1][:, np.newaxis],
              eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n',w)




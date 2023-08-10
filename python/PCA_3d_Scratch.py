import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ensures that random dummy data we are creating is same irrespective of how much times we run
np.random.seed(2343243)

mean_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1 = np.random.multivariate_normal(mean_vec1, cov_mat1, 100)

mean_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2 = np.random.multivariate_normal(mean_vec2, cov_mat2, 100)

from mpl_toolkits.mplot3d import Axes3D, proj3d

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection = '3d')      ## 111 means 1 row, 1 column in grid and we are filling 1st position

ax.plot(class1[:,0], class1[:,1], class1[:,2], 'o')
ax.plot(class2[:,0], class2[:,1], class2[:,2], '^')
plt.show()

all_data = np.concatenate((class1, class2))
all_data

## Transforming 3d data into 2d data
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(all_data)
transformed_data

pca.components_

plt.scatter(transformed_data[:,0], transformed_data[:,1])
plt.show()

plt.plot(transformed_data[0:100,0], transformed_data[0:100,1], "o")
plt.plot(transformed_data[100:200,0], transformed_data[100:200,1], "^")
plt.show()

## In case of conversion from 2d to 1d, approxed data from 1d back to 2d lied in a straight line
## Here, in conversion of data from 3d to 2d, approx data from 2d back to 3d will lie in a single plane

X_approx = pca.inverse_transform(transformed_data)
X_approx

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection = '3d')

ax.plot(X_approx[:,0], X_approx[:,1], X_approx[:,2], 'o')
plt.show()

# This graph actually lies on a plane whose equation is ax + by + cz = 1 where a,b,c are (as calculated)-

a = -0.409689
b = 7.2827
c = -7.1008

# To prove lets put x,y,z from approxed data and put back in the equation (ax + by + cz)

for i in range(10):
    y = a * X_approx[i][0] + b * X_approx[i][1] + c * X_approx[i][2]
    print(y)
    
# We can observe that y values is approximately 1.

# Calculating co-variance matrix
all_data_transpose = all_data.T
cov = np.cov(all_data_transpose)
cov

# Calculating Eigen Vectors and Eigen Values from numpy Linear agebra inbuilt function.
eig_values, eig_vectors = np.linalg.eig(cov)

# Returns two array: first is Eigen values and second is Eigen Vectors

eig_values

# Note: Eigen vectors are vertically downwards/column-wise ie. first column is first eigen vector, 
# second column is second eigen vector and so on...
eig_vectors

#Sort the eigen values in decreasing order and in each line the eigen value is followed by corresponding eigen vectors

eig_values_vector_pair = []
for i in range(len(eig_values)):
    eig_vec = eig_vectors[:, i]
    eig_values_vector_pair.append((eig_values[i], eig_vec))

eig_values_vector_pair.sort(reverse = True)

eig_values_vector_pair

# Same as top two eigen vectors calculated by us
pca.components_

#Same as top two eigen values calculated by us
pca.explained_variance_


import numpy as np

# Element-wise multiplication without linear algebra
x = [1,2,3]
y = [2,3,4]
product = []
for i in range(len(x)):
 product.append(x[i]*y[i])

# Element-wise multiplication utilizing linear algebra
x = np.array([1,2,3])
y = np.array([2,3,4])
x * y

import numpy as np

## Scalar
my_scalar = 5
my_scalar = 5.098

## Vector
my_vector = np.array([1,4,5])

## Matrix
my_matrix = np.array([[1,2,3], [4,5,6]])

## Tensor

my_tensor = [[[1,2,3,4]],[[2,5,6,3]],[[7,6,3,4]]] 

my_tensor_two = np.arange(27).reshape((3, 3, 3))

## Matrix Math

a = np.array([[1,2],[3,4]])
b = np.array([[1,2],[3,4]])

a + b

## Taking the Dot Product (Scalar Output)

y = np.array([1,2,3])
x = np.array([2,3,4])
np.dot(y,x)

# Taking the Hadamard Product (Vector Output)

y = np.array([1,2,3])
x = np.array([2,3,4])
y * x 

from scipy.stats.kde import gaussian_kde
from numpy import linspace

testData = np.random.randn(1000)  ## Create Random Data; this numpy function will create a random normal distribution for us

gaussKDE = gaussian_kde(testData)

dist_space = linspace(min(testData), max(testData), 100)

plt.plot(dist_space, gaussKDE(dist_space))

num_bins = 50

## Set the Number of bins to create the PMF. 
counts, bins = np.histogram(testData, bins=num_bins)
bins = bins[:-1] + (bins[1] - bins[0])/2

prob = counts/float(counts.sum())

## Create the chart
plt.bar(bins, prob, 1.0/num_bins)
plt.show()

p_diseasePos = 0.8 ## Chance of having the disease given a positive result
p_diseaseNeg = 0.2 ## Chance of having the disease given a negative result

p_noPos = 0.096
p_noNeg = 0.904

p_FalsePos = (.80 * .01) + (.096 * .99)

p_disease_given_pos = (.80 * .01) / p_FalsePos

print(p_disease_given_pos)

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

from sklearn import preprocessing

labels = df['species']
features = df.iloc[:,0:3]

le = preprocessing.LabelEncoder()
labelsEnc = le.fit_transform(labels)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, labelsEnc, test_size = 0.25, random_state = 50)

rf_classifier = RandomForestClassifier(n_estimators=1000) 
rf_classifier.fit(x_train, y_train)

preds = iris.target_names[rf_classifier.predict(x_test)]
preds

pd.crosstab(iris.target_names[y_test], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

pcaData = pd.read_csv('/users/patricksmith/desktop/demographics.csv')

demData = pcaData[['health','income','stress']]
demData = (demData - demData.mean()) / demData.std()

demData_corr = np.corrcoef(demData.values.T)
demData.corr()

eig_vals, eig_vecs = np.linalg.eig(demData_corr)
print(eig_vals)
print(eig_vecs)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)

value_vector_pairs = [[eig_vals[i], eig_vecs[:,i]] for i in range(len(eig_vals))]
value_vector_pairs.sort(reverse=True)

weight_2d_projection = np.hstack((value_vector_pairs[0][1].reshape(eig_vecs.shape[1],1),
                                  value_vector_pairs[1][1].reshape(eig_vecs.shape[1],1)))

print('Weight data 2d PCA projection matrix:\n', weight_2d_projection)

Z = demo_noage.dot(weight_2d_projection)

fig = plt.figure(figsize=(9,7))

ax = fig.gca()
ax = sns.regplot(Z.iloc[:,0], Z.iloc[:,1],
                 fit_reg=False, scatter_kws={'s':70}, ax=ax)

ax.set_xlabel('principal component 1', fontsize=16)
ax.set_ylabel('principal component 2', fontsize=16)


for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12) 
    
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12) 
    
ax.set_title('PC1 vs PC2\n', fontsize=20)

plt.show()

fig = plt.figure(figsize=(9,7))
ax = fig.gca()
ax = sns.regplot(Z.iloc[:,0], pcaData.age.values,
                 fit_reg=True, scatter_kws={'s':70}, ax=ax)

ax.set_xlabel('principal component 1', fontsize=16)
ax.set_ylabel('age', fontsize=16)


for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12) 
    
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12) 
    
ax.set_title('PC1 vs age\n', fontsize=20)

plt.show()




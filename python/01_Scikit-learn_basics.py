import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')
get_ipython().magic('matplotlib inline')

# load the data
iris = datasets.load_iris()

print(iris.keys())

print(iris.feature_names)
# only print the first 10 samples
print(iris.data[:10])
print('We have %d data samples with %d features'%(iris.data.shape[0], iris.data.shape[1]))

print(iris.target_names)
print(set(iris.target))

print(iris.DESCR)

digits = datasets.load_digits()
print('We have %d samples'%len(digits.target))

print(digits.data)
print('The targets are:')
print(digits.target_names)

print(digits.data.shape)

## plot the first 64 samples, and get a sense of the data
fig = plt.figure(figsize = (8,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))

boston = datasets.load_boston()
print(boston.DESCR)

boston.feature_names

# let's just plot the average number of rooms per dwelling with the price
plt.figure(figsize = (10,8))
plt.plot(boston.data[:,5], boston.target, 'o')
plt.xlabel('Number of rooms')
plt.ylabel('Price (thousands)')

from sklearn.linear_model import LinearRegression

# you can check the parameters as
get_ipython().magic('pinfo LinearRegression')

# let's change one parameter
model = LinearRegression(normalize=True)
print(model.normalize)

print(model)

x = np.arange(10)
y = 2 * x + 1

plt.figure(figsize = (10,8))
plt.plot(x,y,'o')

# generate noise between -1 to 1
# this seed is just to make sure your results are the same as mine
np.random.seed(42)
noise = 2 * np.random.rand(10) - 1

# add noise to the data
y_noise = y + noise

plt.figure(figsize = (10,8))
plt.plot(x,y_noise,'o')

# The input data for sklearn is 2D: (samples == 10 x features == 1)
X = x[:, np.newaxis]
print(X)
print(y_noise)

# model fitting is via the fit function
model.fit(X, y_noise)

# underscore at the end indicates a fit parameter
print(model.coef_)
print(model.intercept_)

# then we can use the fitted model to predict new data
predicted = model.predict(X)

plt.figure(figsize = (10,8))
plt.plot(x,y_noise,'o')
plt.plot(x,predicted, label = 'Prediction')
plt.legend()


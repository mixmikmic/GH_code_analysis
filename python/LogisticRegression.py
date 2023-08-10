# Import Dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Sigmoid Function
# sigmoid(x) = 1 / (1 + exp(-x))

def sigmoid(z):
    return float(1.0 / float(1.0 + np.exp(-1.0 * z)))

# Hypothesis Function
# y_hat = m*X
# Hypothesis = sigmoid(y_hat) => sigmoid(m*X)

def hypothesis(m, X):
    z = 0
    for i in range(len(m)):
        z += X[i] * m[i]
    return sigmoid(z)

# Cost Function
# J = (-1/n) [y log(sigmoid(mX + b)) + (1 - y) log(1 - sigmoid(mX + b))]

def costFunction(X,y,m):
    errorSum = 0
    error = 0
    n = len(y)
    for i in range(n):
        hy = hypothesis(m,X[i])
        if y[i] == 1:
            error = y[i] * np.log(hy)
        elif y[i] == 0:
            error = (1-y[i]) * np.log(1 - hy)
        errorSum += error
    J = (-1/n) * errorSum
    print('Cost: ', J)
    return J

# Gradient Descent
# X,y: Features,Labels
# m: Slope
# lr: Learning Rate

def gradientDescend(X,y,m,lr):
    new_m = []
    n = len(y)
    const = lr/n
    for j in range(len(m)):
        errorSum = 0
        for i in range(n):
            Xi = X[i]
            Xij = Xi[j]
            hi = hypothesis(m, X[i])
            error = (hi - y[i]) * Xij
            errorSum += error
        n = len(y)
        const = float(lr) / float(n)
        J = const * errorSum
        updated_m = m[j] - J
        new_m.append(updated_m)
    return new_m

# Runner Function
# X: Features
# y: Labels
# lr: Learning Rate
# m: Slope
# iters: Number of Iterations

def runner(X,y,lr,m,iters):
    n = len(y)
    a = 0
    hist = []
    print('Starting Gradient Descend...\n')
    for x in range(iters):
        new_m = gradientDescend(X,y,m,lr)
        m = new_m
        a = costFunction(X,y,m)
        hist.append(a)
        
        # Print the information at every 100th step 
        if x % 100 == 0:
            costFunction(X,y,m)
            print('m: ', m)
            print('Cost ', costFunction(X,y,m))
            print('\n')
    return [m,hist]

# Load the Dataset

df = pd.read_csv('dataset/haberman-data.csv')

# Let's have a look at it

df.head()

# Let's give names to the Columns

df.columns = ['age','year_operation','pos_auxillary_nodes','survival_status']

# Let's check the data again
df.head()

# Describe the data

df.describe()

# Correlation

df.corr()

# Features:

X = np.array(df[['age','year_operation','pos_auxillary_nodes']])
print('Features: ',X)

# Preprocessing data to normalize data points and bring to same scale
# using MinMaxScala brings all data point to range between -1 and 1

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
X = min_max_scaler.fit_transform(X)
print('Preprocessed Features: ',X)

# Labels

y = np.array(df['survival_status'])
print('Labels: ',y)

# Training Parameters

# Initial Slopes [m1,m2,m3]
initial_m = [0,0,0]

# Learning Rate
learning_Rate = 0.01

# Number of Iterations
iterations = 2000

# Initial Cost

print('Initial Cost with m1 = {0}, m2 = {1} and m3 = {2} is Cost = {3}'.format(initial_m[0],initial_m[1],initial_m[2],costFunction(X,y,initial_m)))

# Run the Classifier

[m,hist] = runner(X,y,learning_Rate,initial_m,iterations)

# Final Cost

print('Value of m after {0} iterations is m1 = {1}, m2 = {2}, m3 = {3}'.format(iterations,m[0],m[1],m[2]))

# Plot Cost Function Decay

fig,ax = plt.subplots(figsize=(10,8))
ax.plot(hist)
ax.set_title('Cost Decay')


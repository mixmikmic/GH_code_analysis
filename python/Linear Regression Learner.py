def scaleData(z):
   """
   z-scores some array
   """
   mu = z.mean(axis=0)
   sigma = z.std(axis=0)
   z = (z - mu)/sigma 
   return z, mu, sigma

import numpy as np

#My sample data set to train our regression with.   
#This is an array with features price, bedrooms, and square footage
#For example a 100,000 dollar house has 2 bedrooms and 1600 square feet.

trainData = np.array([[100000,2,1600],
                     [200000,4,2500],
                     [250000,4,3000],
                     [150000,3,2000]])
#We'll scale the data as noted above.
trainData, mu, sigma = scaleData(trainData)

y = np.matrix(trainData[:,0]) #slice the first column, house price
y = y.T #we just prefer y to be a column vector instead of a row vector.
X = np.matrix(trainData[:,1:]) #slice the rest of the column into matrix X

#get the number of training samples in X
m = y.size
#Add a column of ones, size m, to X (interception data)
it = np.ones(shape=(m, 1))
X = np.append(it,X,1)

#Evaluate the linear regression
def compute_cost(X, y, theta):
    m = y.size
    y_hat = X.dot(theta)
    J = (1.0/2*m)* (y_hat - y).T.dot((y_hat - y))  
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
   
    m = y.size
    J_history = np.zeros(shape=(num_iters, 1)) # a column vec to hold our previous Js
 
    for i in range(num_iters):
        gradient = (1.0/2*m) * (( y - X.dot(theta)).T.dot(X)).T
        theta =  theta + (alpha * gradient)
        J_history[i, 0] = compute_cost(X, y, theta)
    return theta, J_history

alpha = .01
iterations = 1000
#np.shape[1] is the number of features, so we need an equal # of thetas
theta = np.zeros((X.shape[1],1))

theta, J_history = gradient_descent(X,y,theta,alpha, iterations)
print(theta)

get_ipython().magic('matplotlib inline')

from pylab import *
def plot_grad_descent(alpha, iterNum):
    theta = np.zeros((X.shape[1],1))#reinitialize theta
    theta, J_history = gradient_descent(X, y,theta, alpha, iterNum)
    plot(J_history)
    title("alpha = " +str(alpha)+ "; iterations = " +str(iterNum))
    show()

plot_grad_descent(.01,10)

plot_grad_descent(.01,100)

plot_grad_descent(.01,1000)

plot_grad_descent(.1295,13)

def scaleTestData(z, mu, sigma):
   """
   z-scores some array
   """
   z = (z - mu[1:])/sigma[1:] 
   return z

alpha = .01
iterations = 100
#np.shape[1] is the number of features, so we need an equal # of thetas
theta = np.zeros((X.shape[1],1))
theta, J_history = gradient_descent(X,y,theta,alpha, iterations)

testData = np.array([[2,2000],
                     [3,2200],
                     [5,4000]])



testData = scaleTestData(testData, mu, sigma)


#Add a column of ones to X (intercept term)
m = testData.shape[0]
it = np.ones(shape=(m, 1))
testData = np.append(it,testData,1)
                     

print(testData)
print(theta)

#predicted house prices, unscaled.
print (testData.dot(theta)*sigma[0]+mu[0])


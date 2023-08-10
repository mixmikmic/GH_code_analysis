import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt

def f(x): return 0.1*x**3 + 0.5*x**2 + 10.2*x - 21.

def sim_data(seed=0):
    np.random.seed(seed)
    min_x, max_x, stepsize = -10, 10, 0.1
    #independent var (does not have to be uniformly space in general):
    X = arr([np.arange(min_x, max_x, stepsize)]).T
    #(N,_) = X.shape
    Y = f(X) + np.random.normal(0,5,size=np.shape(X)) #dependent var
    return X,Y

X,Y = sim_data()
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X,Y,marker='x')
plt.show()

(N,D) = X.shape
X1 = np.hstack((np.ones((N,1)), X))
print('X1 (first 10 rows):\n',X1[:10,:])
print(type(X1))

mX1 = np.asmatrix(X1)
print(mX1.shape)
print(np.transpose(mX1).shape)
XX = np.matmul(np.transpose(mX1),mX1)
XX2 = np.matmul(XX.I,np.transpose(mX1))
b = np.matmul(XX2,Y)
print(b)
#XX2= np.transpose(XX2)
print("Coefficient on X is " ,b.item(1))
XXi =XX.I
i = np.matmul(XX,XXi)
print(i)

M = np.linalg.pinv(X1)
b2 = np.matmul(M,Y)
print("Manual inversion of matrix ", b2)
#print(str(b2[0]),str(b2[1]))
print("Numpy package inversion ", b)

Y_hat = np.dot(X1, b2)
print(Y[:5,:])
Y_hat[:5,:]

#now plot the original data and the linear regression fit:
plt.scatter(X1[:,1],Y,marker='x')
plt.plot(X1[:,1],Y_hat)

residuals = (Y - Y_hat)
SR = residuals**2
print('Residual sum of squares', SR.sum())
print('Mean squared error of residuals', SR.sum()/(N-D-1))
print("The sum of residuals is","{0:.4f}".format(residuals.sum()))

total_var = Y.var()
explained_var = total_var - (residuals**2).sum()/float(N)
r2 = explained_var / total_var
print('total_var, explained_var, r-squared:\n', total_var, explained_var, r2)

plt.hist(residuals)
plt.xlabel('error')
plt.ylabel('frequency')

import statsmodels.api as sm
import statsmodels.formula.api as smf
model = sm.OLS(Y,X1).fit()
# Print out the statistics
print(model.params)

print(model.summary())
print("The Mean squared error of residuals is "+ str(model.mse_resid))


dsig=sum(np.square(Y-np.matmul(mX1,b)))/(mX1.shape[0]-mX1.shape[1])
print(dsig)
vcv=np.dot(np.asscalar(dsig),np.linalg.inv(np.matmul(np.transpose(mX1),mX1)))
print(np.sqrt(np.diag(vcv)))











plt.scatter(X,residuals)
plt.xlabel('x')
plt.ylabel('residual error')




import numpy as np
import pyGPs

X = np.arange(-5, 5, 0.2)
s = 1e-9
n = X.size
m = 1/4*np.square(X)
a = np.repeat(X, n).reshape(n, n)
k = np.exp(-0.5*(a - a.transpose())) + s*np.identity(n)   
Y = np.random.multivariate_normal(m, k, 1)
Y = Y.reshape(n)                                          # Converting y from a matrix to an array

model = pyGPs.GPR()
model.setData(X,Y)
# model.setPrior(mean=pyGPs.mean.Zero(), kernel=pyGPs.cov.RBF()) -- This step is redundant
model.optimize(X,Y)                                            
model.predict(np.array([5,6,7,8,9,10]))
model.plot()




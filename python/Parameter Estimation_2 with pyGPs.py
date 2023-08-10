import numpy as np
import pyGPs

# Linear functional L = phi*u(x) was chosen. Desired result: phi = 12.0.

# Generating data
x_u = np.linspace(0, 2*np.pi, 15)
y_u = np.sqrt(x_u)               # Keeping it as simple as possible, with sin instead of sqrt the optimizer can't 
                                 # calculate the optimal hyperparameters, independent of the method
x_f = np.linspace(0, 2*np.pi, 15)
y_f = 12.0*np.sqrt(x_f)          # You can vary the factor, and that very factor should be the output of this program

# The function u is assumed to be a Gaussian Process. 
# After a linear transformation, f has to be a Gaussian Process as well.

model_u = pyGPs.GPR()
model_u.setData(x_u, y_u)
model_u.optimize(x_u, y_u)

model_f = pyGPs.GPR()
model_f.setData(x_f, y_f)
model_f.optimize(x_f, y_f)

# Note that in hyp only the logarithm of the hyperparameter is stored!
# Characteristic length-scale is equal to np.exp(hyp[0]) (Default: 1)
# Signal variance is equal to np.exp(hyp[1]) (Default: 1)

print(np.exp(model_f.covfunc.hyp[1])/np.exp(model_u.covfunc.hyp[1]))	# This should give 12 as output

# My Output: 12.0486915165


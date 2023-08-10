# Remember to use the inline command for plottin
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


x = np.arange(0.001,1.0,.001) # can't start at sero because of singularity
y = np.exp(-1/x)/x
plt.plot(x, y)

# Interesting stuff seems to happen before x = 0.1
x = np.arange(0.001,.1,.0001)
y = np.exp(-1/x)/x
plt.plot(x, y)

# Let's define a function that does a numerical integration

def lhs(eps):
    "returns the left hand side, which we are approximating"
    result = integrate.quad(lambda t: np.exp(-1/t)/t, 0, eps)
    return np.exp(1/eps)*result[0]/eps

# We can test out a few values 
lhs(1)

# Note our first order expansion expects 1-.1 = .9
lhs(.1)

# Note our first order expansion expects 1-.01 = .99
lhs(.01)

lhs(.004)

# Now lets compare the numerical integration with the 1st order approx
x = np.arange(0.005,1.0,.001) # can't start at sero because of singularity
y = np.zeros(len(x))
z = np.zeros(len(x))
for k in range(len(x)):
    y[k] = lhs(x[k])
    z[k] = 1 - x[k]

plt.plot(x, y,x,z)

# Now lets compare the numerical integration with the 2nd order approx
x = np.arange(0.005,.5,.001) # can't start at sero because of singularity
y = np.zeros(len(x))
z = np.zeros(len(x))
for k in range(len(x)):
    y[k] = lhs(x[k])
    z[k] = 1 - x[k] + 2*x[k]*x[k]

plt.plot(x, y,x,z)

# Now lets compare the numerical integration with the 3rd order approx
x = np.arange(0.005,.5,.001) # can't start at sero because of singularity
y = np.zeros(len(x))
z = np.zeros(len(x))
for k in range(len(x)):
    y[k] = lhs(x[k])
    z[k] = 1 - x[k] + 2*x[k]*x[k] - 6*x[k]*x[k]*x[k]

plt.plot(x, y,x,z)

# Now lets compare the numerical integration with the 3rd order approx
x = np.arange(0.005,.5,.001) # can't start at sero because of singularity
y = np.zeros(len(x))
z = np.zeros(len(x))
for k in range(len(x)):
    y[k] = lhs(x[k])
    z[k] = 1 - x[k] + 2*x[k]*x[k]  - 6*x[k]*x[k]*x[k] # + 24*x[k]*x[k]*x[k]*x[k]

plt.plot(x, np.abs(y-z)/(x*x*x))

# slope of log-log plot tells us the order of accuracy
plt.plot(np.log(x), np.log(np.abs(y-z)))

16/4.5

# Now lets compare the numerical integration with the 3rd order approx
x = np.arange(0.005,.5,.001) # can't start at sero because of singularity
y = np.zeros(len(x))
z = np.zeros(len(x))
for k in range(len(x)):
    y[k] = lhs(x[k])
    z[k] = 1 - x[k] + 2*x[k]*x[k]  - 6*x[k]*x[k]*x[k]  + 24*x[k]*x[k]*x[k]*x[k]

# log log plot tells us the order of accuracy
plt.plot(np.log(x), np.log(np.abs(y-z)))




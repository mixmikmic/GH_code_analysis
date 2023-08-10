import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

mainDir = "/home/gf/src/Python/Python-in-the-lab/Bk"
# Today we use the same file of the shapes
filename = "F64ac_0.02_time_V_T.dat"
filename = os.path.join(mainDir, filename)
data = np.loadtxt(filename, comments="#")
time = data[:,0]
with open(filename) as f:
    header = f.readline()
sizes = [float(size) for size in header.split()[1:]]
shapes = dict()
for i, size in enumerate(sizes):
    shapes[size] = data[:,i+1]

# Get the average of the 8 curves
#average = np.zeros_like(shapes[size]) # Ahah, this is not required!
average = 0
for size in shapes:
    shape = shapes[size]
    average += shape/integrate.trapz(shape,time)
average /= len(shapes)
type(average) == type(shapes[size])

# Replot for comparison with the average
for size in sorted(shapes):
    lb = "{0:.2e}".format(size)
    shape = shapes[size]
    norm = integrate.trapz(shapes[size], time)
    plt.plot(time, shapes[size]/norm, label=lb)
plt.legend(ncol=2,loc=(0.15,.05))
plt.plot(time, average, 'k', lw=3);

# Introduction to functions and fitting function
parameters = ["gamma", "A1", "A2"]
def fitShape(x, gamma, a1, a2):
    """
    fitting function for the average shape
    
    Parameters:
    ===========
    a1: float
        amplitude
    a2: float
        constant of the exponential
    gamma: float
        exponent of the shape
    """
    return a1*(x*(1-x))**(gamma-1) * np.exp(-a2*x)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(fitShape, time, average)
for p, diag, parameter in zip(popt,pcov.diagonal(),parameters):
    print("Parameter {0} = {1:.3f} +/- {2:.3f}".format(parameter, p, diag**0.5))
#pcov.diagonal()**0.5 # These are the errors of the fitting parameters at 1 sigma

integrate.trapz(average, time)

# Introduction to functions and fitting function
parameters = ["gamma", "A"]
def fitShape(x, gamma, a):
    """
    fitting function for the average shape
    
    Parameters:
    ===========
    a: float
        constant of the exponential
    gamma: float
        exponent of the shape
    """
    f = (x*(1-x))**(gamma-1) * np.exp(-a * x)
    norm = integrate.trapz(f, x)
    return f/norm

from scipy.optimize import curve_fit
popt, pcov = curve_fit(fitShape, time, average)
for p, diag, parameter in zip(popt,pcov.diagonal(),parameters):
    print("Parameter {0} = {1:.3f} +/- {2:.3f}".format(parameter, p, diag**0.5))
#pcov.diagonal()**0.5 # These are the errors of the fitting parameters at 1 sigma

plt.plot(time, average, 'bo')
plt.plot(time, fitShape(time, *popt), '-r', lw=2) # Note the use of *popt
plt.plot(time, fitShape(time, popt[0], 0), '--r', lw=1) # What did I do?
plt.xlabel("time", size=16)
plt.ylabel("average shape (normalized)", size=14)

from IPython.display import Image
imageDir = "/home/gf/src/Python/Python-in-the-lab/images"
Image(filename=os.path.join(imageDir,'curve_fit1.png'))

Image(filename=os.path.join(imageDir,'curve_fit2.png'))

Image(filename=os.path.join(imageDir,'curve_fit3.png'))

# The solution goes here...
rows, cols = data.shape
variances = np.array([np.var(row) for row in data[:,1:]])
err_average = (variances / (cols - 1))**0.5
popt1, pcov1 = curve_fit(fitShape, time, average, p0=(1.6,0.5), sigma=err_average)
for p, diag, parameter in zip(popt1,pcov1.diagonal(),parameters):
    print("Parameter {0} = {1:.4f} +/- {2:.4f}".format(parameter, p, diag**0.5))
plt.plot(time, average, 'bo')
plt.plot(time, fitShape(time, *popt), '-b', lw=2) # Note the use of *popt
popt2, pcov2 = curve_fit(fitShape, time, average, p0=(1.6,0.5), sigma=err_average, absolute_sigma=True)
for p, diag, parameter in zip(popt2,pcov2.diagonal(),parameters):
    print("Parameter {0} = {1:.4f} +/- {2:.4f}".format(parameter, p, diag**0.5))

def my_int(y, t=time):
    return integrate.trapz(y, t)

fig = plt.figure(figsize=(12,8))
norms = np.apply_along_axis(my_int, 0, data[:,1:])
data_norm = data[:,1:] / norms
average1 = np.apply_along_axis(np.mean, 1, data_norm)

plt.plot(time, average, 'b') # The old calculus
plt.plot(time, average1, 'ro') # The new one

# Can we calculate the sigmas (error bars) in the same way?
rows, cols = data_norm.shape
sigmas = (np.apply_along_axis(np.var, 1, data_norm)/cols)**0.5
plt.errorbar(time, average1, sigmas, fmt="", ecolor='r')

print("No weights")
popt, pcov = curve_fit(fitShape, time, average)
for p, diag, parameter in zip(popt,pcov.diagonal(),parameters):
    print("Parameter {0} = {1:.4f} +/- {2:.4f}".format(parameter, p, diag**0.5))
print(35*"*")
print("Just weights")
popt, pcov = curve_fit(fitShape, time, average, sigma=err_average)
for p, diag, parameter in zip(popt,pcov.diagonal(),parameters):
    print("Parameter {0} = {1:.4f} +/- {2:.4f}".format(parameter, p, diag**0.5))
print(35*"*")
print("Weights and error bars")
popt, pcov = curve_fit(fitShape, time, average, sigma=err_average, absolute_sigma=True)
for p, diag, parameter in zip(popt,pcov.diagonal(),parameters):
    print("Parameter {0} = {1:.4f} +/- {2:.4f}".format(parameter, p, diag**0.5))

popt1, pcov1 = curve_fit(fitShape, time, average, p0=(1.6,0.5), sigma=sigmas) # See the problem?

popt1

# Explore the function
print(fitShape.__doc__)

# Can we make a running code out of the notebook? Let's do it!

fitShape.__name__




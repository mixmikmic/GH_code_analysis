import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy import odr, sqrt
get_ipython().magic('matplotlib inline')

def OLSfit(x, y, dy=None):
    """Find the best fitting parameters of a linear fit to the data through the 
    method of ordinary least squares estimation. (i.e. find m and b for 
    y = m*x + b)
    
    Args:
        x: Numpy array of independent variable data
        y: Numpy array of dependent variable data. Must have same size as x.
        dy: Numpy array of dependent variable standard deviations. Must be same 
            size as y.

    Returns: A list with four floating point values. [m, dm, b, db]
    """
    if dy is None:
        #if no error bars, weight every point the same
        dy = np.ones(x.size)
    denom = np.sum(1 / dy**2) * np.sum((x / dy)**2) - (np.sum(x / dy**2))**2
    m = (np.sum(1 / dy**2) * np.sum(x * y / dy**2) - 
         np.sum(x / dy**2) * np.sum(y / dy**2)) / denom
    b = (np.sum(x**2 / dy**2) * np.sum(y / dy**2) - 
         np.sum(x / dy**2) * np.sum(x * y / dy**2)) / denom
    dm = np.sqrt(np.sum(1 / dy**2) / denom)
    db = np.sqrt(np.sum(x / dy**2) / denom)
    return([m, dm, b, db])

# Use the SciPy ODR tool to fit a line to the data, I wrote a function that takes as input your x and y values,
# their uncertainties, and initial guesses for the slope and intercept of the best fit line. It returns the best fit
# slope and intercept and their uncertainties in a list with [slope,slope uncert.,intercept,intercept uncert.]

def ODRfit(x, y, dx, dy, m_init, b_init):
    linear_model = odr.Model(linearEqn)
    data = odr.RealData(x, y, sx=dx, sy=dy)
    myodr = odr.ODR(data, linear_model, beta0=[m_init, b_init])
    output = myodr.run()
    return[output.beta[0], output.sd_beta[0], 
           output.beta[1], output.sd_beta[1]]

def linearEqn(p, x):
    return p[0]*x + p[1] 

xs = np.abs(np.random.randint(0,20,100))
dxs = np.abs(np.random.randn(100))
ys = ((2+np.random.randn(100))*xs)+3*np.random.randn(100)
dys = 3*np.abs(np.random.randn(100))

olsfit = OLSfit(xs, ys, dy=dys)
print('Slope=',olsfit[0],'+/-',olsfit[1])
print('Intercept=',olsfit[2],'+/-',olsfit[3])

odrfit = ODRfit(xs,ys,dxs,dys,olsfit[0],olsfit[2])
print('Slope=',odrfit[0],'+/-',odrfit[1])
print('Intercept=',odrfit[2],'+/-',odrfit[3])

plt.errorbar(xs,ys,xerr=dxs,yerr=dys,fmt='o',color='black',alpha=0.5)
plt.plot(np.arange(-2,25,1),olsfit[0]*np.arange(-2,25,1)+olsfit[2],color='red',lw=2,alpha=0.7,label='OLS')
plt.plot(np.arange(-2,25,1),odrfit[0]*np.arange(-2,25,1)+odrfit[2],color='blue',lw=2,alpha=0.7,label='ODR')
plt.gcf().set_size_inches(10,10)
plt.legend(fontsize=18)
plt.xlim(-2,25)


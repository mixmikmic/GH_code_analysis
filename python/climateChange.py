import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

data = np.genfromtxt('JanuaryGlobalAverageLandTemperatures.csv',delimiter=',',dtype=float)

print(data)

year = data[:,0]
T = data[:,1]
dT = data[:,2]

print(year)

T_F = T * 1.8 + 32

print(T_F)

plt.errorbar(year,T,yerr=dT,fmt='o',ms=5,color='black',alpha=0.75)
plt.gcf().set_size_inches(15,10) # This sets the size of the plot
plt.ylim(-5,10) # This sets the range of the x-axis
plt.xlim(1750,2015) # This sets the range of the y-axis
plt.grid(True) # This toggles whether gridlines are displayed
plt.xlabel('Year',fontsize=16)
plt.ylabel('Average Global Surface Temperature in January [$^\circ$C]',fontsize=16)
plt.savefig('/Users/anneya/web/phys2300/assets/img/climatePlot1.png',bbox_inches='tight',dpi=300)

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

bestfit = OLSfit(year,T,dT)
print(bestfit)
slope = bestfit[0]
intercept = bestfit[2]

# Create an array of values starting at 1750, going up to 2100, with steps on 10
xForLine = np.arange(1750.,2150.,10.) 
# Calculate the value of temperature = slope * year + intercept at each x
yForLine = slope*xForLine + intercept

plt.errorbar(year,T,yerr=dT,fmt='o',ms=5,color='black',alpha=0.75)
plt.gcf().set_size_inches(15,10) # This sets the size of the plot
plt.ylim(-5,10) # This sets the range of the x-axis
plt.xlim(1750,2100) # This sets the range of the y-axis
plt.grid(True) # This toggles whether gridlines are displayed
plt.xlabel('Year',fontsize=16)
plt.ylabel('Average Global Surface Temperature in January [$^\circ$C]',fontsize=16)

# Add the line
plt.plot(xForLine,yForLine,lw=3,color='red',alpha=0.5,label='Line of Best Fit')

# Add a legend
plt.legend()
plt.savefig('/Users/anneya/web/phys2300/assets/img/climatePlot2.png',bbox_inches='tight',dpi=300)




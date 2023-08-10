# will use libs for numerics, html requests and plotting (directly to browser)
import numpy as np
import requests
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# function for requesting Kazachstan's machine for outputs on x_vals inputs

def GetY(x_vals):
    N = len(x_vals)
    y_vals = np.zeros(N)
    count = 0
    
    print('Requesting...')
    for x in x_vals:
        if (int( count % (N/10) ) == 0):
            print('Processed: ' + str(int(100*count/N)) + '%')
                
        url = "http://165.227.157.145:8080/api/do_measurement?x={}".format(x)
        r = requests.get(url)
        json_data = r.json()
        y_vals[count] = ( json_data['data']['y'] )
        count += 1
        
    print('Finished !')
    return y_vals

# requesting x=0 1,000 times
xArr = np.zeros(1000)
yArr = GetY(xArr)

plt.figure(num=None, figsize=(6, 4), dpi=100)
plt.hist(yArr, bins = 24)
plt.xlabel('f(x=0) outputs statistics')
plt.ylabel('Frequency')
plt.xlim(-10,-2)
plt.show()

# requesting x=0 - 10,000 times
xArr = np.zeros(10000)
yArr = GetY(xArr)

# normed to probability distribution

plt.figure(num=None, figsize=(12, 4), dpi=100)

plt.subplot(121)
nbins = 67 # check 'Last notes' for explanation of that 67
plt.hist(yArr, nbins, normed=1, facecolor='blue')
plt.xlabel('f(x=0) outputs statistics')
plt.ylabel('Probability')

# gaussian fit

plt.subplot(122)
n, bins, patches = plt.hist(yArr, nbins, normed=1, facecolor='blue', alpha=0.3)

from scipy.stats import norm
import matplotlib.mlab as mlab

(mu, sigma) = norm.fit(yArr)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=1)

plt.title('Well-fitted distribution')
plt.xlabel('f(x=0) outputs statistics')
plt.ylabel('Probability')

plt.show()

# choosing reasonable symmetric interval x = -50, -40, ...., 50.
# sample size = 2000 - just to see, whether the distribution changes off or not

for x in range(-50,60,10):
    xArr = x * np.ones(2000) 
    yArr = GetY(xArr)
    
    plt.figure(num=None, figsize=(6, 4), dpi=75)
    
    nbins = 30    #according to F-D rule
    n, bins, patches = plt.hist(yArr, nbins, normed=1, facecolor='blue', alpha=0.3)
    (mu, sigma) = norm.fit(yArr)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=1)
    
    plt.title('Input: 2000 times x={}'.format(x))
    plt.xlabel('Outputs statistics')
    plt.ylabel('Probability')
    plt.show()

# here are used just 200 samples of various x for purpose of μ(x), σ(x) calculation
mu = np.zeros(101)
sigma = np.zeros(101)

count=0
for x in range(-50,51):
    #if (x % 10 == 0):
    print('Run for x={}'.format(x))
    xArr = x * np.ones(200) 
    yArr = GetY(xArr)
    (mu[count], sigma[count]) = norm.fit(yArr)
    count += 1
    
print('FINISHED!')

xArr = np.linspace(-50,50,101)

plt.figure(num=None, figsize=(6, 4), dpi=100)
plt.plot(xArr, mu, 'bo')
plt.title('evolution of means $\mu(x)$')
plt.xlabel('inputs [$x$]')
plt.xlim(-55,55)
plt.ylabel('means of outputs [$\mu$]')
plt.show()

plt.figure(num=None, figsize=(6, 4), dpi=100)
plt.plot(xArr[xArr>0], mu[xArr>0], 'bo')
plt.plot(-xArr[xArr<0],mu[xArr<0], 'rx', markersize=7)
plt.title('symmetry of $\mu(x)$ (blue and red)')
plt.xlabel('inputs [$x$]')
plt.ylabel('means of outputs [$\mu$]')
plt.show()

plt.figure(num=None, figsize=(6, 4), dpi=100)
plt.plot(xArr[xArr>0], mu[xArr>0], 'bo')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('ln($x$)')
plt.ylabel('ln($\mu$)')
plt.show()

b = (np.log(mu[-1]) - np.log(mu[55])) / (np.log(xArr[-1]) - np.log(xArr[55]))
print(b)

a = mu[-1] / xArr[-1]**4.08
print(a)

# plot + fit with calculated parameters
plt.figure(num=None, figsize=(6, 4), dpi=100)
plt.plot(xArr, mu, 'bo')
plt.plot(xArr, 0.73*(abs(xArr)**(4.08) - 6), 'r--')
plt.title('Means $\mu(x)$ and polynomial fit $0.73x^{4.08}$')
plt.xlabel('inputs [$x$]')
plt.xlim(-55,55)
plt.ylabel('means of outputs [$\mu$]')
plt.show()

xArr = np.linspace(-50,50,101)

plt.figure(num=None, figsize=(6, 4), dpi=100)
plt.plot(xArr, sigma, 'bo')
plt.title('evolution of deviations $\sigma(x)$')
plt.xlabel('inputs [$x$]')
plt.xlim(-55,55)
plt.ylabel('deviations of outputs [$\sigma$]')
plt.show()

plt.figure(num=None, figsize=(6, 4), dpi=100)
plt.plot(xArr[xArr>0], sigma[xArr>0], 'bo')
plt.plot(-xArr[xArr<0],sigma[xArr<0], 'rx', markersize=7)
plt.title('symmetry of $\sigma(x)$ (blue and red)')
plt.xlabel('inputs [$x$]')
plt.ylabel('deviations of outputs [$\sigma$]')
plt.grid(True)
plt.show()

# construction of data matrix X
X = np.ones((51, 2))
X[:,1] = xArr[xArr>=0]

# calculating coeff vector beta
XT = np.matrix.transpose(X)
beta = np.matmul( np.matmul( np.linalg.inv( np.matmul(XT,X) ), XT), sigma[50:])
print(beta)

plt.figure(num=None, figsize=(6, 4), dpi=100)
plt.plot(xArr[xArr>0], sigma[xArr>0], 'bo')
plt.plot(-xArr[xArr<0],sigma[xArr<0], 'rx', markersize=7)
plt.plot(xArr[xArr>=0], 0.5*xArr[xArr>=0] + 0.08, 'k--', linewidth=3.0)
plt.title('$\sigma(x)$ + fit')
plt.xlabel('inputs [$x$]')
plt.ylabel('deviations of outputs [$\sigma$]')
plt.grid(True)
plt.show()

import scipy as sc
from scipy.stats import iqr
sc.stats.iqr(yArr)
FD = 2 * iqr(yArr) / ( 1000**(1/3) )
print( (max(yArr) - min(yArr)) / FD )


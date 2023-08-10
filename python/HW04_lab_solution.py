import numpy as np
import matplotlib.pyplot as plt

######################################################################
## CDF for exponential distribution
## with parameter lambda -- from HW02_lab.ipynb
######################################################################
def exp_cdf(arr,lam):
    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for x')
            return np.array([-1])

    return (1 - np.exp( -lam*arr ) )*(arr>0)

######################################################################
## probability density function for exponential distribution
## with parameter lambda -- from Lab04.ipynb
######################################################################
def exp_pdf(arr,lam):

    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for arr')
            return np.array([-1])
        
    return lam*np.exp(-lam*arr)*(arr>0)

######################################################################
## probability density function for exponential distribution
## with parameter lambda -- from Lab04.ipynb
######################################################################
def exp_pdf(arr,lam):

    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for arr')
            return np.array([-1])
        
    return lam*np.exp(-lam*arr)*(arr>0)


######################################################################
## CDF for exponential distribution
## with parameter lambda -- from HW02_lab.ipynb
######################################################################
def exp_cdf(arr,lam):
    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for x')
            return np.array([-1])

    return (1 - np.exp( -lam*arr ) )*(arr>0)

######################################################################
## inverse CDF for exponential distribution
## with parameter lambda -- calculation by hand
######################################################################

def invCDF(arr,lam):
    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for x')
            return np.array([])
        
  ######################################################################
  ## All values for arr, need to be between 0 and 1 as these values 
  ## correspond to the value probability 
  ######################################################################    
    if( np.sum(arr<0) + np.sum(arr>=1) > 0 ):
        print('wrong input, should be in [0,1)')
        return np.array([])

    
    return -np.log(1-arr)/lam

######################################################################
## All needed functions are defined, 
## now we can follow lab04_inversetransform.ipynb to plot
######################################################################

## the total number of samples
######################################################################

N = 10000 

## N random numbers from uniform distribution U[0,1]
######################################################################

R = np.random.rand(N)

## Use inverse sampling to generate samples follows exp distribution
##  with parameter lambda
######################################################################

lambdas = [0.5,1,1.5]

X = []
for lam in lambdas:
    X.append(invCDF(R,lam))

## plot  densitydensity  histogram
######################################################################

xvals=np.linspace(1e-5, 10, 1000)


for i in range(0,3):
    plt.figure(figsize=(8,5))
    lam = lambdas[i]
    plt.hist(X[i],  normed=1,#bins = 50 ,
             label=u"Density hist: $\lambda = %.1f$"%lam, 
             edgecolor='black', linewidth=1.2,alpha = lam/2);
    plt.plot(xvals, exp_cdf(xvals,lam), 'b', 
             label=u'actual CDF')

    plt.legend()
    plt.show()

X = []
for lam in lambdas:
    X.append(invCDF(R,lam))

## plot  densitydensity  histogram
######################################################################

xvals=np.linspace(1e-5, 10, 1000)


for i in range(0,3):
    plt.figure(figsize=(8,5))
    lam = lambdas[i]
    plt.hist(X[i],  normed=1,bins = 50 ,
             label=u"Density hist: $\lambda = %.1f$"%lam, 
             edgecolor='black', linewidth=1.2,alpha = lam/2);
    plt.plot(xvals, exp_cdf(xvals,lam), 'b', 
             label=u'actual CDF')

    plt.legend()
    plt.show()

# We consider the following array of values a --200 of them
aa = np.linspace(1e-5, 5, 200)

#for each number A in a, 
# we want to calculate the number of samples < A / total number of samples
# store the result in CDFarr

CDFarr = np.zeros(len(aa))

# for the case X1
X = X[0]
for i in range(0,len(aa)):
    A = aa[i]
    total     = np.sum(X<A)
    prob_A    = total/len(X)
    CDFarr[i] = prob_A
    

# plot CDF of exp distribution with parameter 0.5

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(aa,CDFarr,'.',color = 'blue',
         label = u'probability calculation using sample')
plt.legend()

plt.subplot(212)
plt.plot(aa,exp_cdf(aa,0.5),'-',
        label = u'actual cummulative density $1-e^{-0.5x}$')
plt.legend()

plt.show()




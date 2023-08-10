get_ipython().magic('pylab inline')

import pandas as pd                    # pandas is the "python data analysis package"
import os                              # operating system services
from scipy.optimize import curve_fit   # non-linear curve fitting

def getFileSizes(folder, minsize=2000, maxsize=10**9):
    """
    return a list of file sizes in a folder
    """
    sizes = []
    for (path, dirs, files) in os.walk(folder):
        for file in files:
            try:
                filename = os.path.join(path, file)
                size=os.path.getsize(filename)
                if size>minsize and size<maxsize:
                    sizes.append(size)
            except OSError:
                pass
    return sizes

if 0: # set this to "1" to collect data
    minsize=10**4
    maxsize=10**10
    sizes = getFileSizes('/Users/steve/Desktop',minsize,maxsize) # <--- edit this to match some folder with a lot of files
    print(len(sizes), 'files between', minsize, 'and', maxsize)


if 0: # set this to "1" to write the file
    df=pd.DataFrame({'file_sizes':sizes})
    df.to_csv('myData.csv', sep=',')
else:
    df=pd.read_csv('myData.csv')
    df.head()

ns,bins,patches = hist(log(array(sizes)+1),40)

xvals=(bins[:-1]+bins[1:])/2.0
counts=ns+0.5
yvals=log(counts)
title("distribution of file sizes")
ylabel("log(count)")
xlabel("log(size)")
plot(xvals,yvals,'b.')

#
# define the function values (these are the columns of the "M" matrix
#
# generate arrays that contain "functions" of x that are combined to form the model
#

def doFit(funcs, xvals, yvals, sigma):
    """
    doFit uses matrices to solve for the linear least squares 
    fit parameters based on the list of 'funcs', and the x-y values
    and their corresponding uncertainties (sigma).
    
    returns the array of model parameters
    """
    #
    # define the S matrix in terms of uncertainties
    #
    
    S=diag((1.0/sigma)**2)
    
    #
    # define the model matrix
    #
    
    M = array(funcs).T
    MtM = M.T.dot(S.dot(M))
    MtMInv = inv(MtM)
    MtY = M.T.dot(S.dot(yvals))
    alpha = MtMInv.dot(MtY)      # see eq 2 above.
    
    #
    # solve for the parameters, and return
    #
    ystar = M.dot(alpha)
    
    return (alpha, ystar, MtMInv)

fx1 = ones(len(xvals))   # X_1(x) = 1
fx2 = xvals              # X_2(x) = x

sigma=1.0/sqrt(abs(counts))

alpha, ystar, fcov = doFit([fx1, fx2], xvals, yvals, sigma)

mFit = alpha[1]
bFit = alpha[0]

errorbar(xvals, yvals, sigma, fmt='r.') 
title("File Sizes on My HD")
ylabel("log(count)")
xlabel("log(size)")
plot(xvals,ystar, 'g-')

#
# Now do monte-carlo data fabrication and fit analysis
#

N=1000    # number of samples to take
mList=[]  # keep track of monte-carlo fit parameters
bList=[]

for i in range(N):
    """
    Generate mc data with the same statistical properties as the real data.
    Repeat the fit for each set, and record the parameters.
    """
    mcY = ystar + sigma*normal(size=len(xvals))
    mcAlpha,mcYstar,mccov = doFit([fx1,fx2],xvals,mcY,sigma)   # repeatedly fit mc data
    mList.append(mcAlpha[1])
    bList.append(mcAlpha[0])
    
#
# Compute the statistics of the mc-results
#

marr=array(mList)
barr=array(bList)

mAvg = marr.sum()/N
bAvg = barr.sum()/N
delM = marr-mAvg
delB = barr-bAvg
sigM = sqrt((delM*delM).sum()/(N-1))  # sigM is the std-deviation of the m values
sigB = sqrt((delB*delB).sum()/(N-1))  # sigB is the std-deviation of the b values

#
# plot the fit
#
print("Slope=", mFit, '+/-', sigM, "(", mFit - sigM,",",mFit + sigM, ")")
print("Intercept=", bFit, '+/-', sigB, "(", bFit - sigB,",",bFit + sigB, ")")

#
# Just for fun print out the diagonals of the covariance matrix.
#
print # print a blank link
print("Compare to cov-matrix for fun:")
print("sqsrt(cov[1,1]) (should be sigma m)", sqrt(fcov[1,1]))
print("sqrt(cov[0,0]) (should be sigma b)", sqrt(fcov[0,0]))

hist(marr) # look at the variation in "m" values
print("m average:", mAvg)
print("m sigma:",sigM)

hist(barr)  # look at the variation in "b" values
print("b average:", bAvg)
print("b sigma:",sigB)

#
# Just to see same data fit using the non-linear scipy package "curve_fit":
#

def fLinear(x, m, b):
    return m*x + b

popt, pcov = curve_fit(fLinear, xvals, yvals, p0=(alpha[1],alpha[0]), sigma=sigma)

m=popt[0]          # slope
dm=sqrt(pcov[0,0]) # sqrt(variance(slope))
b=popt[1]          # int
db=sqrt(pcov[1,1]) # sqrt(variance(int))
ystar=fLinear(xvals, m, b)

N=1000    # number of samples to take
mList=[]  # keep track of monte-carlo fit parameters
bList=[]

for i in range(N):
    """
    Generate mc data with the same statistical properties as the real data.
    Repeat the fit for each set, and record the parameters.
    """
    mcY = ystar + sigma*normal(size=len(xvals))  # generate fabricated data to fit
    mcpopt, mcpcov = curve_fit(fLinear, xvals, mcY, p0=(m,b), sigma=sigma)
    mList.append(mcpopt[0])  # store the fit paramters for the fab-data
    bList.append(mcpopt[1])
    
#
# Compute the statistics of the mc-results
#
marr=array(mList)
barr=array(bList)

mAvg = marr.sum()/N
bAvg = barr.sum()/N
delM = marr-mAvg
delB = barr-bAvg
sigM = sqrt((delM*delM).sum()/(N-1))  # sigM is the std-deviation of the m values
sigB = sqrt((delB*delB).sum()/(N-1))  # sigB is the std-deviation of the b values

errorbar(xvals, yvals, sigma, fmt='r.') 
title("File Sizes on My HD")
ylabel("log(count)")
xlabel("log(size)")
plot(xvals,ystar, 'g-')
print("Slope=", m, '+/-', sigM, "(", m-sigM,",",m+sigM, ")")
print("Intercept=", b, '+/-', sigB, "(", b-db,",",b+sigB, ")")

print() # print a blank link
print("Compare to cov-matrix for fun:")
print("sqsrt(pcov[0,0]) (should be sigma m)", sqrt(pcov[0,0]))
print("sqrt(pcov[1,1]) (should be sigma b)", sqrt(pcov[1,1]))




import celerite
from celerite import terms
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
c = 2.99792458e8   # m/s

data = pickle.load(open( "binary_data.p", "rb" ))
print len(data)

wave_lo = np.log(4680.)
wave_hi = np.log(4700.)
subset = []
#for i in range(len(data)):
for i in range(0,len(data),10):
    m = (data[i][0] > wave_lo) & (data[i][0] < wave_hi)
    x = np.copy(data[i][0][m])
    y = np.log(np.copy(data[i][1][m]))
    yerr = np.copy(data[i][2][m]/data[i][1][m])
    subset.append((x,y,yerr))

kernel = terms.RealTerm(np.log(0.04), -np.log(0.001), bounds=((np.log(.01*0.04),np.log(100.*0.04)),(None, None)))
gp = celerite.GP(kernel,
                 log_white_noise=-9.6,
                 fit_white_noise=True)

gp.get_parameter_dict()

nepoch = len(subset)
eye = np.eye(nepoch)
ndata_byepoch = [len(d[0]) for d in subset]
design = np.repeat(eye, ndata_byepoch, axis=0)
print design.shape
print design

def shift_and_flatten(xis, data):
    ndata = sum([len(d[0]) for d in data])
    ndata_byepoch = [len(d[0]) for d in data]
    n = 0
    x = np.empty(ndata)
    y = np.empty(ndata)
    yerr = np.empty(ndata)
    for i, d in enumerate(data):
        length = len(d[0])
        x[n:n+length] = d[0] - xis[i]
        y[n:n+length] = d[1]
        yerr[n:n+length] = d[2]
        n += length
    return x, y, yerr

def set_params(params):
    xis, gp_par = params[0:len(subset)], params[len(subset):]
    x, y, yerr = shift_and_flatten(xis, subset)
    inds = np.argsort(x)
    x = x[inds]
    y = y[inds]
    yerr = yerr[inds]
    A = np.copy(design)[inds,:]    
    gp.set_parameter_vector(gp_par)
    gp.compute(x, yerr)
    scales = np.linalg.solve(np.dot(A.T, gp.apply_inverse(A)), np.dot(A.T,gp.apply_inverse(y)))
    ndata_byepoch = [len(d[0]) for d in subset]
    y[np.argsort(inds)] -= np.repeat(scales, ndata_byepoch)
    return scales, xis, y

def nll(params):
    scales, xis, y = set_params(params)
    return -gp.log_likelihood(y) + 1./2. # * np.sum(xis**2)

def xi_to_v(xi):
    # translate ln(wavelength) Doppler shift to a velocity in m/s
    return np.tanh(xi) * c
 
def v_to_xi(v):
    return np.arctanh(v/c)

from scipy.io.idl import readsav
print len(subset)
data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
pipeline = readsav(data_dir+'HIP14501_result.dat') 

xis0 = np.empty(len(subset))
rvs = np.empty(len(subset))
dates = np.empty(len(subset))
for i in range(len(subset)):
    rvs[i] = pipeline.rv[i*10] * 1.e3
    xis0[i] = v_to_xi(rvs[i])
    dates[i] = pipeline.date[i*10]

#xis0 = np.zeros(len(subset))

print rvs
print xis0

p0 = np.append(xis0, gp.get_parameter_vector())
print p0
bounds = [(-1e-4, 1e-4) for d in subset] + gp.get_parameter_bounds()
print bounds

scales, xis, y = set_params(p0)

soln = minimize(nll, p0, bounds=bounds, method='L-BFGS-B')
scales, xis, y = set_params(soln.x)

print soln
print xi_to_v(xis)

fig,ax = plt.subplots(1,1,figsize=(12,4))
for i,d in enumerate(subset):
    ax.plot(d[0] - xis[i],d[1] - scales[i])

def prediction(params):
    scales, xis, y = set_params(params)
    result_flat = gp.predict(y, return_cov=False)
    x, _, _ = shift_and_flatten(xis, subset)
    inds = np.argsort(x)
    result_sorted = result_flat[np.argsort(inds)]
    result = []
    n = 0
    for i,d in enumerate(subset):
        length = len(d[0])
        result.append(result_sorted[n:n+length] + scales[i])
        n += length
    return result

mu = prediction(soln.x)

print mu

fig,ax = plt.subplots(1,1,figsize=(12,4))
for i,d in enumerate(subset):
    ax.plot(d[0], d[1], color='black')
    ax.plot(d[0], mu[i], color='red')

fig,ax = plt.subplots(1,1,figsize=(12,4))
for i,d in enumerate(subset):
    ax.plot(d[0], (np.exp(d[1]) - np.exp(mu[i])) + 1000*i, color='black')

plt.scatter(pipeline.date, pipeline.rv*1.e3, color='black', label='HARPS RVs')
plt.scatter(dates, xi_to_v(xis), color='red', label='Avast RVs')
plt.xlabel('BJD')
plt.ylabel('RV (m/s)')

rvs1 = np.arange(5) + 2000.
rvs2 = np.arange(5)
xis1 = v_to_xi(rvs1)
xis2 = v_to_xi(rvs2)
print (xis1 - xis1[0]) - (xis2 - xis2[0])
print xi_to_v((xis2 - xis1[0]))




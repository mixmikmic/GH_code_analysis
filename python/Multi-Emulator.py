get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic("config InlineBackend.figure_format = 'retina' # I use a HiDPI screen")

import numpy as np
import matplotlib.pyplot as plt
import itertools

N = 60

def f( x1 ):
    mu, sigma = 0, 1 # mean and standard deviation
    return x1 + np.random.normal(mu, sigma)


params = [0,2,4,6,8]
sample_list = [[f(x) for i in range(N)] for x in params]

for s,p in zip(sample_list, params):
    plt.hist( s, label='$x=' + str(p) + r'$', alpha=0.5 )
plt.ylim(0,30)
plt.legend(ncol=2);

mean_stack = [np.mean(s_i) for s_i in sample_list]
cov = np.cov(sample_list[0])

from scipy.interpolate import Rbf
rbfi = Rbf(params, mean_stack)  # radial basis function interpolator instance

plt.plot( params, mean_stack, 'k-o', markersize=5, label='fake sim data' )

grid = np.linspace(0,8,100)
plt.plot( grid, rbfi(grid), 'r--', label='emulator')
plt.xlabel( '$x$' )
plt.ylabel( 'sample mean' )
plt.legend()

data = 4.
grid = np.linspace( 0, 8, 100 )

P_list = np.exp( -0.5 * np.array([(data - rbfi(x)) * 1./cov * (data-rbfi(x)) for x in grid]) )

plt.plot( grid, P_list / np.trapz(y=P_list,x=grid), label='Estimated' )
plt.plot( grid, np.exp( -(grid-data)**2 / 2 ) / (np.sqrt(2 * np.pi)),
         label='True' )
plt.ylabel('P')
plt.xlabel(r'$x$')
plt.ylim(0,1)
plt.legend()

# construct three emulators, 0-20, 20-40, and 40-60

emulator_list = []
cov = np.cov(sample_list[0])
for i in range(3):
    mean_stack = [np.mean(s_i[i*20:i*20+20]) for s_i in sample_list]
    rbfi = Rbf(params, mean_stack)  # radial basis function interpolator instance
    emulator_list.append(rbfi)

def get_combine_P( input_par ):
    chi_sum = 0.0
    data = 4.0
    count = 0
    for ema, emb in itertools.combinations( emulator_list, r=2):
        chi_sum += -0.5 * (data - ema(input_par)) * 1./cov * (data-emb(input_par)) 
        count += 1
    return np.exp(chi_sum/count)

P_list = [get_combine_P(x) for x in grid]

plt.plot( grid, P_list / np.trapz(y=P_list,x=grid), label='Estimated' )
plt.plot( grid, np.exp( -(grid-data)**2 / 2 ) / (np.sqrt(2 * np.pi)), label='True' )
plt.ylabel('P')
plt.xlabel(r'$x$')
plt.ylim(0,1)
plt.legend()










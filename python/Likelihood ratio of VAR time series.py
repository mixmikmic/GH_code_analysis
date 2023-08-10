get_ipython().magic('matplotlib inline')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import theano
import scipy.linalg as L
import pdb
import pickle

import carl

import multiprocessing

plt.set_cmap('viridis')

np.random.seed(1234)

sys.setrecursionlimit(10000)

import statsmodels.tsa.vector_ar.var_model as vector_ar

p_value_ = 1
ndims_ = 2
nparams_ = 2
N_ = 20000
T_ = 50
alpha_value_0 = 0.5
alpha_value_1 = 0.4
sigma_2_value = [[1.,0.],[0.,1.]]
intercept=[0.,0.]
alpha_value_0_bkg = 0.3
alpha_value_1_bkg = 0.3
sigma_2_value_bkg = [[1.,0.],[0.,1.]]

target_position = (alpha_value_0, alpha_value_1)
bkg_position = (alpha_value_0_bkg, alpha_value_1_bkg)
bounds = [(0.3, 0.6), (0.3, 0.6)]
n_points = 5

As = np.linspace(bounds[0][0],bounds[0][1], n_points)
Bs = np.linspace(bounds[1][0],bounds[1][1], n_points)
AA, BB = np.meshgrid(As, Bs)
AABB = np.hstack((AA.reshape(-1, 1),
               BB.reshape(-1, 1)))

class VAR:

    def sample(self, coefs, intercept, sigma_2, n_steps, rng=None):
        return vector_ar.util.varsim(coefs, intercept, sigma_2, steps=n_steps)
    
    def plot(self,steps=1000):
        self.VAR.plotsim(steps)

    def nll(self, X, alphas, intercept, sigma_2):
        """
        Compute the negative log likelihood for a VAR process, 
        using formula from Lutkepohl (2005)
        """
        lags = 1
        trend = 'c'
        alpha_0 = alphas[0]
        alpha_1 = alphas[1]
        coefs = np.array([alpha_0,0.,0.,alpha_1]
                        ).reshape(p_value_,ndims_,ndims_)
        
        VAR = vector_ar.VAR(X)
        VAR.k_trend = k_trend = vector_ar.util.get_trendorder(trend)

        offset = 0
        y = VAR.y[offset:]

        z = vector_ar.util.get_var_endog(y, lags, trend, 
                                         has_constant='skip')
        y_sample = y[lags:]
        intercept = intercept
        params = np.vstack((intercept, coefs.reshape((2,2))))
        #params = np.linalg.lstsq(z, y_sample)[0]
        resid = y_sample - np.dot(z, params)
        omega = np.array(sigma_2)
        
        lastterm = -np.trace(np.dot(np.dot(resid, L.inv(omega)),resid.T))
        
        varfit = vector_ar.VARResults(y, z, params, omega, lags, names=VAR.endog_names,
                    trend=trend, dates=VAR.data.dates, model=self)
        llf = -varfit.llf
        llf += 0.5*lastterm
        return -llf
    
    def rvs(self, n_samples, alpha_0, alpha_1, intercept, sigma_2, random_state=1234):
        rng = np.random.RandomState(random_state) if                 isinstance(random_state,int) else random_state
        coefs = np.array([alpha_0,0.,0.,alpha_1]
                        ).reshape(p_value_,ndims_,ndims_)
        return self.sample(coefs=coefs,intercept=intercept,
                           sigma_2=sigma_2,
                           n_steps=n_samples,rng=rng)

p0 = VAR()
p1 = VAR()
rng = np.random.RandomState(1234)

n_true = 1000
X_true = np.array([p0.rvs(T_, alpha_value_0, alpha_value_1, intercept,
                          sigma_2_value, random_state=np.random.randint(0,5000))
                   for i in range(n_true)])

from itertools import product
bounds_values = [(np.linspace(bounds[0][0],bounds[0][1], num=5)),
                 (np.linspace(bounds[1][0],bounds[1][1], num=5))]
combinations = list(product(*bounds_values))

def produce_data(value):
    return np.array([p0.rvs(T_, value[0], value[1], intercept, sigma_2_value,
                           random_state=np.random.randint(0,5000))
                   for i in range(N_//2)])   

Xs = []
ys = []
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)
Xs = pool.map(produce_data, combinations)
pool.close()
X1_ = np.array([p1.rvs(T_, alpha_value_0_bkg, alpha_value_1_bkg, 
                       intercept, sigma_2_value,
                       random_state=np.random.randint(0,5000))
               for i in range(N_//2)])
Xs = np.array(Xs)

X = Xs[0][0]

serie1 = X1_[0][:,0]
serie2 = Xs[0][0][:,0]
plt.figure(figsize=(20,10))
plt.subplot(211)
ts = pd.Series(serie1)
ts[0:100].plot()
plt.subplot(212)
ts = pd.Series(serie1)
ts.plot()

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, GRU, LSTM, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from carl.ratios import ClassifierRatio
from carl.learning import CalibratedClassifierCV
from carl.learning import as_classifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit

def make_model_join():
    model = Sequential()
    model.add(GRU(15, input_shape=(T_, ndims_,)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    sgd = Adam(clipnorm=40.)
    model.compile(loss="binary_crossentropy", optimizer='adam')
    return model

def make_ratio(num):
    X_num = Xs[num]
    X_den = X1_
    X = np.vstack((X_num, X_den))
    y = np.zeros(len(X_num) + len(X_den), dtype=np.int)
    y[len(X_num):] = 1

    clf = KerasClassifier(make_model_join, nb_epoch=50, verbose=0)

    cv =  StratifiedShuffleSplit(n_iter=1, test_size=0.5, random_state=1)

    ratio = ClassifierRatio(
        base_estimator=CalibratedClassifierCV(clf, cv=cv, bins=20),
        random_state=0)
    ratio.fit(X, y)

    return ratio

print('Start training')
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

#clf_ratios = []
n_sample_points = len(combinations)
points = list(range(n_sample_points))
for i, k in enumerate([points[n:n + num_cores] for n in range(0, n_sample_points, num_cores)]):
    if k == 0:
        continue
    print('Iteration {0}'.format(i))
    clf_ratios += pool.map(make_ratio, k)
#pool.close()
print('End training')

len(clf_ratios)

llr = []
zeros = []
distances = []

for i, theta in enumerate(combinations):
    print(i, theta)
    ratio = clf_ratios[i]
    ratios = ratio.predict(X_true, log=True)
    print ratios[np.isinf(ratios)].shape
    zeros.append(ratios[np.isinf(ratios)].shape[0])
    ratios = ratios[np.logical_and(np.isfinite(ratios),~np.isnan(ratios))]
    nllr = -np.mean(ratios) 
    llr.append(nllr)
    print(llr[-1])

llr = np.array(llr)
llr[np.isnan(llr)] = 0.

# Computing exact likelihood
exact_contours = np.zeros(len(AABB))
combinations = product(As, Bs)
i = 0
for a in As:    
    for b in Bs:
        exact_contours[i] = np.mean([p0.nll(X_true[k], [a,b], intercept, sigma_2_value) 
                                     for k in range(100)])
        i += 1

#llr, zeros = pickle.load(open('data/ar_results.dat', 'r'))

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
plt.rcParams["figure.figsize"] = (12, 10)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern


# Define a class that forces representation of float to look a certain way
# This remove trailing zero so '1.0' becomes '1'
class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.1f' % self.__float__()
        else:
            return '%.1f' % self.__float__()

# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
    fmt = r'%r '
else:
    fmt = '%r '

thetas = np.array([v for v in product(As, Bs)])

gp = GaussianProcessRegressor(alpha=0., normalize_y=True, 
                              kernel=C(1.0) * Matern(1.0, length_scale_bounds="fixed"))
gp.fit(thetas, exact_contours)

xi = np.linspace(bounds[0][0], bounds[0][1], 50)
yi = np.linspace(bounds[1][0], bounds[1][1], 50)
    
xx, yy = np.meshgrid(xi, yi)
zz, std = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
zi = zz.reshape(xx.shape)

mle = np.unravel_index(zi.argmin(),zi.shape)
zi -= zi[mle]
zi *= 2.

cs = plt.contour(xi, yi, zi, [1.,4.,9.,16.,25.], linewidths=0.5, colors='w')
plt.clabel(cs, cs.levels, inline=1, fontsize=10, fmt = fmt)

cs = plt.contourf(xi, yi, zi, 30,
                  vmax=abs(zi).max(), vmin=0.0)

#plt.colorbar()
plt.scatter(thetas[:,0],thetas[:,1], marker='o', c='w', s=50, lw=0, zorder=10)
plt.scatter([xi[mle[1]]], [yi[mle[0]]], marker='o', c='r', s=50, lw=0, zorder=10)
plt.scatter([alpha_value_0],[alpha_value_1], marker='o', c='g', s=50, lw=0, zorder=10)

plt.xlim(bounds[0][0], bounds[0][1])
plt.ylim(bounds[1][0], bounds[1][1])
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel(r"$\alpha_0$", fontsize=25)
plt.ylabel(r"$\alpha_1$", fontsize=25)
plt.tight_layout()
#plt.savefig('plots/alpha_gp_exact.pdf')
plt.show()

thetas = np.array([v for v in product(As, Bs)])

gp = GaussianProcessRegressor(alpha=0., normalize_y=True, 
                              kernel=C(1.0) * Matern(1.0, length_scale_bounds="fixed"))
gp.fit(thetas, llr)

xi = np.linspace(bounds[0][0], bounds[0][1], 50)
yi = np.linspace(bounds[1][0], bounds[1][1], 50)
    
xx, yy = np.meshgrid(xi, yi)
zz, std = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
zi = zz.reshape(xx.shape)

mle = np.unravel_index(zi.argmin(),zi.shape)
zi -= zi[mle]
zi *= 2.

cs = plt.contour(xi, yi, zi, [1.,4.,9.,16.,25.], linewidths=0.5, colors='w')
plt.clabel(cs, cs.levels, inline=1, fontsize=10, fmt = fmt)

cs = plt.contourf(xi, yi, zi, 30,
                  vmax=abs(zi).max(), vmin=0.0)

#plt.colorbar()
plt.scatter(thetas[:,0],thetas[:,1], marker='o', c='w', s=50, lw=0, zorder=10)
plt.scatter([xi[mle[1]]], [yi[mle[0]]], marker='o', c='r', s=50, lw=0, zorder=10)
plt.scatter([alpha_value_0],[alpha_value_1], marker='o', c='g', s=50, lw=0, zorder=10)

plt.xlim(bounds[0][0], bounds[0][1])
plt.ylim(bounds[1][0], bounds[1][1])
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel(r"$\alpha_0$", fontsize=25)
plt.ylabel(r"$\alpha_1$", fontsize=25)
plt.tight_layout()
#plt.savefig('plots/alpha_gp_approx.pdf')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
ax = axes.flat[0]

gp = GaussianProcessRegressor(alpha=0.0, normalize_y=True, 
                              kernel=C(1.0) * Matern(1.0, length_scale_bounds="fixed"))
gp.fit(thetas, llr)

xi = np.linspace(bounds[0][0], bounds[0][1], 500)
yi = np.linspace(bounds[1][0], bounds[1][1], 500)
    
xx, yy = np.meshgrid(xi, yi)
zz, std = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
zi = zz.reshape(xx.shape)

mle = np.unravel_index(zi.argmin(),zi.shape)
zi -= zi[mle]
zi *= 2.

cs = ax.contour(xi, yi, zi, [1.,4.,9.,16.,25.], linewidths=0.5, colors='w')
cs.levels = [nf(val) for val in cs.levels]
ax.clabel(cs, cs.levels, inline=1, fontsize=10, fmt = fmt)
cs = ax.contourf(xi, yi, zi, 30, cmap="viridis",
                  vmax=abs(zi).max(), vmin=0.0)

#plt.colorbar()
ax.scatter(thetas[:,0],thetas[:,1], marker='o', c='b', s=50, lw=0, zorder=10)
ax.scatter([xi[mle[1]]], [yi[mle[0]]], marker='o', c='r', s=50, lw=0, zorder=10)
ax.scatter([alpha_value_0],[alpha_value_1], marker='o', c='w', s=50, lw=0, zorder=10)

ax.set_xlim(bounds[0][0], bounds[0][1])
ax.set_ylim(bounds[1][0], bounds[1][1])
ax.set_xlabel(r"$\alpha_0$", size=16)
ax.set_ylabel(r"$\alpha_1$", size=16)
ax.set_title("Approx. -2lnL (VAR(2,1))")

ax = axes.flat[1]

gp = GaussianProcessRegressor(alpha=0.0, normalize_y=True, 
                              kernel=C(1.0) * Matern(1.0, length_scale_bounds="fixed"))
gp.fit(thetas, exact_contours)

xi = np.linspace(bounds[0][0], bounds[0][1], 500)
yi = np.linspace(bounds[1][0], bounds[1][1], 500)
    
xx, yy = np.meshgrid(xi, yi)
zz, std = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
zi = zz.reshape(xx.shape)

mle = np.unravel_index(zi.argmin(),zi.shape)
zi -= zi[mle]
zi *= 2.

cs2 = ax.contour(xi, yi, zi, [1.,4.,9.,16.,25.], linewidths=0.5, colors='w')
cs2.levels = [nf(val) for val in cs2.levels]
ax.clabel(cs2, cs2.levels, inline=1, fontsize=10, fmt = fmt)
cs2 = ax.contourf(xi, yi, zi, 30, cmap="viridis",
                  vmax=abs(zi).max(), vmin=0.0)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.12, 0.03, 0.775])
fig.colorbar(cs2, cax=cbar_ax)

ax.scatter(thetas[:,0],thetas[:,1], marker='o', c='b', s=50, lw=0, zorder=10)
smle = ax.scatter([xi[mle[1]]], [yi[mle[0]]], marker='o', c='r', s=50, lw=0, zorder=10)
sobs = ax.scatter([alpha_value_0],[alpha_value_1], marker='o', c='w', s=50, lw=0, zorder=10)

lines = [smle, sobs]
labels = ['MLE', 'Observed']

ax.legend(lines, labels, frameon=False, prop={'size':12}, scatterpoints=1)

ax.set_xlim(bounds[0][0], bounds[0][1])
ax.set_ylim(bounds[1][0], bounds[1][1])
ax.set_xlabel(r"$\alpha_0$", size=16)
ax.set_ylabel(r"$\alpha_1$", size=16)
ax.set_title("Exact. -2lnL (VAR(2,1))")

plt.show()
#plt.savefig('ar_2d_approx.pdf')

from carl.ratios.utils import plot_scores

plot_scores(clf_ratios[0:9], 
            positions=[(l,bkg_position) for l in combinations[0:9]])




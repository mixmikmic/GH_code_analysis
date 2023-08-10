get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import theano

import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder + '/carl')

matplotlib.style.use('ggplot')

np.random.seed(1234)

def logit(x):
    return np.log(x) - np.log(1-p)
def logistic(x):
    return 1. / (1. + np.exp(-x)) 

class Ricker:
    def __init__(self,r,sigma_2,phi):
        self.r = r
        self.sigma_2 = sigma_2
        self.phi = phi
    
    def sample(self,r=0.5, sigma_2=1., phi=0.1, start=0.1,n_timesteps=1000, rng=None):
        noise = rng.normal(0.,sigma_2, n_timesteps)
        time_serie = np.zeros(n_timesteps,dtype=np.float64)
        time_serie[0] = np.abs(noise[0])
        r_e = np.exp(r)
        for i in range(1,n_timesteps):
            time_serie[i] = (r_e*time_serie[i-1]*np.exp(-time_serie[i-1] + noise[i]))

        sampled = np.zeros(n_timesteps)
        sampled = np.array([rng.poisson(phi*time_serie[i]) for i in range(0,n_timesteps,1)])
        
        return_serie = np.log(time_serie[1:]/(time_serie[:-1]))
        
        return time_serie.reshape(time_serie.shape[0],1), sampled.reshape(sampled.shape[0],1),return_serie.reshape(return_serie.shape[0],1)
    def rvs(self,n_samples, random_state=1234):
        rng = np.random.RandomState(random_state) if                 isinstance(random_state,int) else random_state
        return self.sample(r=self.r.eval(),sigma_2=self.sigma_2.eval(),phi=self.phi.eval(),
                     n_timesteps=n_samples,rng=rng)[1]
                
#serie1,sampled1 = ricker(r=np.exp(3.8),sigma_2=0.3**2,phi=10.,n_timesteps=n_samples,start=0.5)
#serie2,sampled2 = ricker(r=np.exp(4.5), sigma_2=0.3**2,phi=10.,n_timesteps=n_samples,start=0.5)

#X_true,X_true_obs = ricker(r=np.exp(3.8),sigma_2=0.3**2,phi=10.,n_timesteps=1500,start=0.5)

r_value = 3.8
sigma_2_value = 0.3**2
phi_value = 5.

r_bkg = 5.
sigma_2_bkg = 0.3**2
phi_bkg = 5.


shared_r = theano.shared(r_value, name="r")
shared_sigma_2 = theano.shared(sigma_2_value,name="sigma_2")
shared_phi = theano.shared(phi_value, name="phi")

params_values = [r_value, sigma_2_value, phi_value]
params_shared = [shared_r, shared_sigma_2, shared_phi]
params_bkg = [r_bkg, sigma_2_bkg, phi_bkg]
params_bounds = [(3.5,4.2),(0.1**2,0.5**2),(3.5,7.5)]

par_idx = 0
par_value = params_values[par_idx]
par_shared = params_shared[par_idx]
par_bkg = params_bkg[par_idx]
par_bound = params_bounds[par_idx]

n_samples = 10000 # train
test_samples = 1500 # test
hist_samples = 10000 # histograms
n_bins = 15 # histogram construction
max_len = 4
n_sampled = 100

p0 = Ricker(r=shared_r,sigma_2=shared_sigma_2,phi=shared_phi)
#p1 = Ricker(r=np.exp(4.5), sigma_2=0.3**2,phi=10.)
p1 = Ricker(r = theano.shared(r_bkg, name="r_1"),
            sigma_2 = theano.shared(sigma_2_bkg,name="sigma_2_1"),
            phi = theano.shared(phi_bkg, name="phi_1")
            )
rng = np.random.RandomState(1234)

X_true = p0.rvs(test_samples, random_state=rng)
#print X_true

serie1 = p0.rvs(1000).ravel()
serie2 = p1.rvs(1000).ravel()
#serie1 = (serie1 - serie1.mean()) / serie1.std()
#serie2 = (serie2 - serie2.mean()) / serie2.std()
plt.figure(figsize=(20,10))
plt.subplot(221)
ts = pd.Series(serie1)
ts[0:100].plot()
plt.subplot(222)
ts = pd.Series(serie1)
ts.plot()
plt.subplot(223)
ts = pd.Series(serie2)
ts[0:100].plot()
plt.subplot(224)
ts = pd.Series(serie2)
ts.plot()
print serie1.min(),serie1.max()

from carl.learning import ParameterStacker

from itertools import product
def make_parametrized_classification_std(p0, p1, n_samples, params, rng):
    if not isinstance(params[0], tuple):
        X0 = p0.rvs(n_samples // 2, random_state=rng)
        X1 = p1.rvs(n_samples - (n_samples // 2), random_state=rng)
        X = ParameterStacker(params).transform(np.vstack((X0, X1)))
        y = np.zeros(n_samples)
        y[len(X0):] = 1

        return X, y

    elif isinstance(params[0], tuple):
        combinations = list(product(*[values for _, values in params]))

        all_X = []
        all_y = []

        for c in combinations:
            for i, v in enumerate(c):
                params[i][0].set_value(v)

            X, y = make_parametrized_classification_std(
                p0, p1,
                n_samples // len(combinations),
                [p for p, _ in params], rng=rng)

            all_X.append(X)
            all_y.append(y)

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        return X, y

    else:
        raise ValueError

Xs, ys = [],[]
for n in range(n_sampled):
    X, y = make_parametrized_classification_std(
        p0, p1,
        n_samples, 
        [(par_shared, np.linspace(par_bound[0],par_bound[1], num=15))],
        rng = n)
    Xs.append(X)
    ys.append(y)

X_series, y_series = [],[]

for n in range(n_sampled):
    X0_serie = []
    y0_serie = []
    X1_serie = []
    y1_serie = []

    X0 = Xs[n][ys[n] == 0]
    X1 = Xs[n][ys[n] == 1]
    for i in xrange(X0.shape[0]-max_len+1):
        # ensure that is from same time serie
        if X0[i][-1] == X0[i+max_len-1][-1]:
            X0_serie.append(X0[i:i+max_len])
        if X1[i][-1] == X1[i+max_len-1][-1]:
            X1_serie.append(X1[i:i+max_len])
    X0_serie = np.array(X0_serie)
    X1_serie = np.array(X1_serie)

    X0_serie = X0_serie[np.random.choice(X0_serie.shape[0],X0_serie.shape[0],replace=False)]
    X1_serie = X1_serie[np.random.choice(X1_serie.shape[0],X1_serie.shape[0],replace=False)]

    X_serie = np.vstack((
         X0_serie,
         X1_serie))
    y_serie = np.zeros(X0_serie.shape[0]*2,dtype=np.int)
    y_serie[X0_serie.shape[0]:] = 1
    
    X_series.append(X_serie)
    y_series.append(y_serie)

len(y_series)

X_serie = X_series[0]
y_serie = y_series[0]
for k in range(1,len(X_series)):
    X_serie = np.vstack((X_serie,X_series[k]))
    y_serie = np.hstack((y_serie,y_series[k]))

print (y_serie.shape)

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, GRU, LSTM, Dropout
from carl.learning import as_classifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD


def make_model():
    model = Sequential()
    model.add(SimpleRNN(10,input_shape=(max_len,2)))
    model.add(Dropout(0.3))
    model.add(Dense(5,activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    #sgd = SGD(lr=0.01, clipnorm=40.)
    model.compile(loss="binary_crossentropy", optimizer='adam')
    return model


clf = KerasClassifier(make_model, nb_epoch=10, verbose=2)
#clf = make_pipeline(StandardScaler(),as_classifier(clf))

clf.fit(X=X_serie, y=y_serie)
#clf.fit(X=X_serie, y=y_serie,nb_epoch=3,batch_size=32,verbose=2)

# 0.2278
#print clf.predict_proba(X_serie)
print(X_serie.shape)
choices = np.random.choice(X_serie.shape[0]//2,5000,replace=False)
X_serie_plot = np.vstack((X_serie[y_serie==0][choices],X_serie[y_serie==1][choices]))
y_serie_plot = np.zeros(X_serie_plot.shape[0])
y_serie_plot[5000:] = 1

print(X_serie_plot.shape)

from carl.ratios import ClassifierRatio
from carl.learning import CalibratedClassifierCV
from carl.learning import as_classifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cc_direct = ClassifierRatio(
    base_estimator=CalibratedClassifierCV(clf, bins='blocks', cv='prefit', 
                                         method='histogram'), 
    random_state=0)

cc_direct.fit(X=X_serie_plot,y=y_serie_plot)

reals2 = np.linspace(0, 1)
cal_num, cal_den = cc_direct.classifier_.calibrators_[0].calibrators
plt.plot(reals2, cal_num.pdf(reals2.reshape(-1, 1)), 
         label="p(s_num=c,den=c, x")
plt.plot(reals2, cal_den.pdf(reals2.reshape(-1, 1)), 
        label="p(s_num=c,den=c), x~c")
plt.legend()
plt.show()

from carl.learning import CalibratedClassifierCV
from carl.ratios import ClassifierRatio
import pdb

def vectorize(func):
    def wrapper(X):
        v = np.zeros(len(X))
        
        for i, x_i in enumerate(X):
            v[i] = func(x_i)
            
        return v.reshape(-1, 1)
    
    return wrapper

def objective(theta, random_state=0, X_true=None):    
    #print(theta)
    par_shared.set_value(theta[0])
    serie1 = p0.rvs(hist_samples)
    # Set parameter values   
    X0 = np.zeros((serie1.shape[0],serie1.shape[1]+1))
    X0[:,:-1] = serie1
    X0[:,-1] = theta
    
    # Fit ratio
    ratio = ClassifierRatio(CalibratedClassifierCV(
        base_estimator=clf, 
        cv="prefit",  # keep the pre-trained classifier
        method="histogram", bins=15))

    serie2 = p1.rvs(hist_samples,random_state=random_state)
    X1 = np.zeros((serie2.shape[0],serie2.shape[1]+1))
    X1[:,:-1] = serie2
    X1[:,-1] = par_bkg

    X0_serie = []
    X1_serie = []

    for i in xrange(serie1.shape[0]-max_len):
        X0_serie.append(X0[i:i+max_len])
        X1_serie.append(X1[i:i+max_len])
    X0_serie = np.array(X0_serie)
    X1_serie = np.array(X1_serie)
    X0_serie = X0_serie[np.random.choice(X0_serie.shape[0],X0_serie.shape[0],replace=False)]
    X1_serie = X1_serie[np.random.choice(X1_serie.shape[0],X1_serie.shape[0],replace=False)]

    X_serie = np.vstack((
         X0_serie,
         X1_serie))
    y_serie = np.zeros(X0_serie.shape[0]*2,dtype=np.int)
    y_serie[X0_serie.shape[0]:] = 1
    
    ratio.fit(X_serie, y_serie)
    
    # Evaluate log-likelihood ratio
    X_true_ = np.zeros((X_true.shape[0],X_true.shape[1]+1))
    X_true_[:,:-1] = X_true
    X_true_[:,-1] = theta
    
    X_true_serie = []
    for i in xrange(X_true_.shape[0]-max_len):
        X_true_serie.append(X_true_[i:i+max_len])
    X_true_serie = np.array(X_true_serie)
    X_true_serie = X_true_serie[np.random.choice(X_true_serie.shape[0],X_true_serie.shape[0],
                                                replace=False)]
    
    r = ratio.predict(X_true_serie, log=True)
    zeros = r[np.isinf(r)].shape

    r = r[np.isfinite(r)]
    value = -np.mean(r)

    return value, zeros[0]

n_points = 15
bounds = [par_bound]

par_shared.set_value(par_value)
X_true = p0.rvs(test_samples, random_state=rng)

As = np.linspace(bounds[0][0],bounds[0][1], n_points)

approx_contours = np.zeros(len(As))
zeroes = np.zeros(len(As))
for i,a in enumerate(As):
    approx_contours[i], zeroes[i] = objective([a],X_true=X_true)
    print('Iteration {0}: {1}'.format(i,a))


approx_contours = 2. * (approx_contours - approx_contours.min()) * len(X_true)

print(As)
print(approx_contours)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(As, approx_contours, label=r"Approx., $\theta_1=(r=e^{3.8},\sigma^2=0.3^2,\phi=10.)$")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$-2 \log \lambda(\theta)$")

ax.legend()
ax.set_xlim(bounds[0][0],bounds[0][1])
ax.axvline(x=par_value,linestyle='dashed',alpha=0.5)
plt.savefig('likelihood_comp.pdf')
plt.show()
plt.plot(As, zeroes)
plt.xlim(bounds[0][0],bounds[0][1])
plt.xlabel(r"$r$")
plt.ylabel(r"$zeros$")
plt.show()

n_hist = 50
n_points = 30
As = np.linspace(bounds[0][0],bounds[0][1], n_points)

hist_values = np.zeros(n_hist)

for h in range(n_hist):
    par_shared.set_value(par_value)
    X_true = p0.rvs(test_samples,random_state=rng)
    approx_contours = np.zeros(len(As))
    zeroes = np.zeros(len(As))
    for i,a in enumerate(As):
        approx_contours[i], zeroes[i] = objective([a], X_true=X_true)
    min_value = As[np.argmin(approx_contours)]
    print '{0} '.format(min_value),
    hist_values[h] = min_value
#approx_contours = 2. * (approx_contours - approx_contours.min()) * len(X_true)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(hist_values, range=(bounds[0][0],bounds[0][1]), 
        label=r"Approx., $\theta_1=(r=e^{3.8},\sigma^2=0.3^2,\phi=10.)$", bins=18)
ax.set_xlabel(r"$approx r$")
ax.set_ylabel(r"$Count$")

ax.legend()
ax.axvline(x=r_value,linestyle='dashed',alpha=0.5)
plt.savefig('likelihood_comp.pdf')
plt.show()




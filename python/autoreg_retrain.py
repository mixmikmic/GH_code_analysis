get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import theano
matplotlib.style.use('ggplot')

np.random.seed(1234)

n_samples = 250000
alpha_value = 0.3
sigma_2_value = 1.**2

shared_alpha = theano.shared(alpha_value, name="alpha")
shared_sigma_2 = theano.shared(sigma_2_value,name="sigma_2")

class AR:
    def __init__(self,alpha,sigma_2):
        self.alpha = alpha
        self.sigma_2 = sigma_2
    
    def sample(self, alpha=0.2, sigma_2=1.,n_timesteps=1000, rng=None):
        noise = rng.normal(0.,sigma_2, n_timesteps)
        time_serie = np.zeros(n_timesteps,dtype=np.float64)
        time_serie[0] = np.abs(noise[0])
        for i in range(1,n_timesteps):
            time_serie[i] = alpha*time_serie[i-1] + noise[i]

        return time_serie.reshape(time_serie.shape[0],1), time_serie.reshape(time_serie.shape[0],1)
    def rvs(self,n_samples, random_state=1234):
        rng = np.random.RandomState(random_state) if                 isinstance(random_state,int) else random_state
        return self.sample(alpha=self.alpha.eval(),sigma_2=self.sigma_2.eval(),
                     n_timesteps=n_samples,rng=rng)[0]
                
    
#serie1,sampled1 = ricker(r=np.exp(3.8),sigma_2=0.3**2,phi=10.,n_timesteps=n_samples,start=0.5)
#serie2,sampled2 = ricker(r=np.exp(4.5), sigma_2=0.3**2,phi=10.,n_timesteps=n_samples,start=0.5)

#X_true,X_true_obs = ricker(r=np.exp(3.8),sigma_2=0.3**2,phi=10.,n_timesteps=1500,start=0.5)

p0 = AR(alpha=shared_alpha,sigma_2=shared_sigma_2)
#p1 = Ricker(r=np.exp(4.5), sigma_2=0.3**2,phi=10.)
p1 = AR(alpha = theano.shared(0.5, name="alpha_1"),
        sigma_2 = theano.shared(sigma_2_value,name="sigma_2_1")
            )
rng = np.random.RandomState(1234)

X_true = p0.rvs(15000,random_state=rng)
print X_true

serie1 = p0.rvs(n_samples).ravel()
serie2 = p1.rvs(n_samples).ravel()
plt.figure(1)
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
print serie2.min(),serie2.max()

from carl.learning import make_parameterized_classification
n_samples_points = 10
samples_set = []

parameters_points = np.linspace(0.1, 0.9, n_samples_points)
n_samples = 1000000 // n_samples_points

for k in range(n_samples_points):
    shared_alpha.set_value(parameters_points[k])
    x0 = p0.rvs(n_samples // 2, random_state = 1234)
    x1 = p1.rvs(n_samples // 2, random_state = 1234)
    x = np.vstack((x0,x1))
    y = np.zeros(n_samples)
    y[len(x0):] = 1
    samples_set.append((x,y))

print(samples_set[-1])

import pdb
max_len = 2
series_set = []

for (X,y) in samples_set:
    X0_serie = []
    y0_serie = []
    X1_serie = []
    y1_serie = []

    X0 = X[y == 0]
    X1 = X[y == 1]
    for i in xrange(X0.shape[0]-max_len+1):
        # ensure that is from same time serie
        X0_serie.append(X0[i:i+max_len])
        X1_serie.append(X1[i:i+max_len])

    X0_serie = np.array(X0_serie)
    X1_serie = np.array(X1_serie)
    X_serie = np.vstack((
         X0_serie,
         X1_serie))
    y_serie = np.zeros(X0_serie.shape[0]*2,dtype=np.int)
    y_serie[X0_serie.shape[0]:] = 1
    series_set.append((X_serie, y_serie))

print(series_set[0][0].shape)

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, GRU, LSTM, Dropout
from carl.learning import as_classifier
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def scheduler(epoch):
    if (epoch + 1) % 25 == 0:
        lr_val = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(lr_val*0.5)
    return float(model.optimizer.lr.get_value())

sgd = SGD(lr=0.01, clipnorm = 40.)

classifiers = []
from keras_wrapper import KerasClassifier
lr_schedule = LearningRateScheduler(scheduler)

for k,(X_serie,y_serie) in enumerate(series_set):
    print ('Training classifier {0}'.format(k))
    model = Sequential()
    model.add(SimpleRNN(10,input_shape=(max_len,1)))
    model.add(Dropout(0.3))
    model.add(Dense(5,activation='tanh'))
    model.add(Dense(1,activation='tanh'))

    #model.compile(loss='mean_squared_error', optimizer='rmsprop')

    clf = KerasClassifier(model=model, loss='mean_squared_error', optimizer=sgd, nb_epoch=15, verbose=2)
    #clf = make_pipeline(StandardScaler(),as_classifier(clf))

    clf.fit(X=X_serie, y=y_serie)
    classifiers.append(clf)
    #clf.fit(X=X_serie, y=y_serie,nb_epoch=3,batch_size=32,verbose=2)

from carl.learning import CalibratedClassifierCV
from carl.ratios import ClassifierRatio
ratios = []

for k,(X_serie, y_serie) in enumerate(series_set):
    # Fit ratio
    clf = classifiers[k]
    ratio = ClassifierRatio(CalibratedClassifierCV(
        base_estimator=clf, 
        cv="prefit",  # keep the pre-trained classifier
        method="histogram", bins=50))

    ratio.fit(X_serie, y_serie)

    # Evaluate log-likelihood ratio
    X_true_ = X_true

    X_true_serie = []
    for i in xrange(X_true_.shape[0]-max_len):
        X_true_serie.append(X_true_[i:i+max_len])
    X_true_serie = np.array(X_true_serie)

    r = ratio.predict(X_true_serie, log=True)
    print r[np.isfinite(r)].shape
    value = -np.mean(r[np.isfinite(r)])  # optimization is more stable using mean
                                         # this will need to be rescaled by len(X_true)
    ratios.append(value)

plt.plot(parameters_points, np.array(ratios))
print ratios




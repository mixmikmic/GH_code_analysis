get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pystan
plt.rcParams['figure.figsize'] = [12,10]

npts = 100
f = np.linspace(108,112,npts)
b = 2.0
h = -1.0
w = -2.0
nu = 110.0
model = np.exp(b) + (np.exp(h)/((nu - f)**2 + (np.exp(w))**2/4))
data = model * np.random.chisquare(2, npts)
fig, ax = plt.subplots()
ax.plot(f, model, zorder=5)
ax.plot(f, data)
ax.set_xlabel('Frequency')
ax.set_ylabel('Power')

mode_code = '''
functions {
    real lor(real f, real nu, real h, real w){
        return h / (pow(nu - f, 2) + w^2/4);
    }
}
data {
    int<lower = 0> N;
    vector[N] f;
    vector[N] p;
}
parameters {
    real<lower=108.0, upper=112.0> nu;
    real<lower = -3, upper = 1> w;
    real<lower = -2, upper = 8> h;
    real<lower = -4, upper = 8> b;
}
model {
    vector[N] tmp;
    for (i in 1:N)
        tmp[i] = 0.5 / (exp(b) + lor(f[i], nu, exp(h), exp(w))); 
    p ~ gamma(1, tmp);
}
'''
sm = pystan.StanModel(model_code=mode_code, model_name='Modefit')

dat = {'N': len(f),
      'f': f,
      'p': data}
fit = sm.sampling(data=dat, iter=2000, chains=2)
fit.plot()

print(fit)

npts = 10000
sns.distplot(np.random.chisquare(2, npts))
sns.distplot(np.random.gamma(1.0, 2.0, npts))
sns.distplot(np.random.exponential(2.0, npts))
print(np.mean(np.random.chisquare(2, npts)))
sns.distplot(data/model)




import numpy as np
import matplotlib.pyplot as plt
import pystan
import seaborn as sns

npts = 1000
sigma = 0.2
mean = -1.61
M_true = np.random.randn(npts) * sigma + mean
unc = 1.0
M = M_true + np.random.randn(npts) * unc
sns.distplot(M)
sns.distplot(M_true)

clump_model = '''
data {
    int<lower=0> N;
    real M[N];
    real<lower = 0> tau;
}
parameters {
    real mu;
    real<lower =0.001> sigma;
    real M_true_std[N];
}
transformed parameters {
    real M_true[N]; // Transform from N(0,1) back to M
    for (i in 1:N)
        M_true[i] = mu + sigma * M_true_std[i];
}
model {
    M_true_std ~ normal(0, 1); // prior but transformed
    M ~ normal(M_true, tau); // Measurement uncertainty
}
'''
sm = pystan.StanModel(model_code=clump_model, model_name='ClumpModel')

dat = {'N': len(M),
      'M': M,
      'tau': unc}
fit = sm.sampling(data=dat, iter=2000, chains=4, pars=['mu', 'sigma'])
fit.plot()

print(fit)




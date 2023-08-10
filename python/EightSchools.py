get_ipython().run_line_magic('matplotlib', 'inline')

import pystan
import numpy as np
import matplotlib.pyplot as plt

schools_code = '''
data {
    int<lower=0> j; // Number of schools
    real y[j]; // Treatment effects
    real<lower=0> s[j]; // s.d. of effects
}
parameters {
    real mu;
    real<lower=0> tau;
    real eta[j];
}
transformed parameters {
    real theta[j];
    for (i in 1:j)
    theta[i] = mu + tau * eta[i];
}
model {
    eta ~ normal(0, 1);
    y ~ normal(theta, s);
}

'''


sm = pystan.StanModel(model_code=schools_code, model_name='EightSchools')

schools_dat = {'j': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               's': [15, 10, 16, 11,  9, 11, 10, 18]}
fit = sm.sampling(data=schools_dat, iter=1000, chains=4, sample_file='samples')
fit.plot()

fit2 = sm.sampling(data=schools_dat, iter=10000, chains=4)

fit2.plot()




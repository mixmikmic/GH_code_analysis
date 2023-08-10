get_ipython().run_line_magic('matplotlib', 'inline')

import pystan
import matplotlib.pyplot as plt
import numpy as np

hours = [0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50]
passed = [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]

fig, ax = plt.subplots()
ax.scatter(hours, passed)
ax.set_xlabel('Hours studied')
ax.set_ylabel('Passed exam')
dat = {'N': len(hours),
      'x': hours,
      'y': passed}

lr_code = '''
data {
    int<lower=0> N;
    vector[N] x;
    int<lower=0, upper=1> y[N];
}
parameters {
    real alpha;
    real beta;
}
model {
    y ~ bernoulli_logit(alpha + beta * x);
}
''' 
sm = pystan.StanModel(model_code=lr_code, model_name='LogisiticReg')

fit = sm.sampling(data=dat, iter=2000, chains=4)
fit.plot()

print(fit)

xx = np.linspace(0.5, 5.5, 100)
fig, ax = plt.subplots()
ax.scatter(hours, passed)
ax.plot(xx, (1.0 + np.exp(-(fit['alpha'].mean() + fit['beta'].mean() * xx)))**-1, 'k--', lw=3)
a = fit['alpha']
b = fit['beta']
for i in range(0,len(a),10):
    ax.plot(xx, (1.0 + np.exp(-(a[i] + b[i] * xx)))**-1, 'r-', alpha=0.01)
ax.set_xlabel('Hours studied')
ax.set_ylabel('Probability of passing exam')




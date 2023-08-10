#%matplotlib inline

import pystan
import matplotlib.pyplot as plt
import numpy as np

npts = 100
x = np.random.rand(npts)
m = 3.3
c = 1.2
s = 1.1
e = np.random.randn(npts) * s
y = m*x + c + e
sel = np.random.randint(0,npts-1, 20)
print(sel)
y[sel] *= 10

fig, ax = plt.subplots()
ax.scatter(x, y)

dat = {'N': npts,
      'y': y,
      'x': x}

lin_code = '''
data {
    int<lower = 0> N;
    vector[N] x;
    vector[N] y;
    
}
parameters {
    real alpha;
    real beta;
    real<lower = 0> sigma;
    real<lower = 0> nu;
}
model {
    y ~ student_t(nu, alpha + beta * x, sigma);
}
'''
sm = pystan.StanModel(model_code=lin_code, model_name='RobustLinReg')

fit = sm.sampling(data=dat, iter=1000, chains=4)
fit.plot()

print(fit)

print(fit['alpha'].mean())

fig, ax = plt.subplots()
ax.scatter(x, y)
xx = np.linspace(0,1,100)
ax.plot(xx, fit['alpha'].mean() + fit['beta'].mean() * xx, 'k--', lw=3)
a = fit['alpha']
b = fit['beta']
for i in range(len(a)):
    ax.plot(xx, a[i] + b[i]*xx, 'r-', alpha=0.01)




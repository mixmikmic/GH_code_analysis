import GPy, numpy as np, pandas as pd
from GPy.kern import LinearSlopeBasisFuncKernel, DomainKernel, ChangePointBasisFuncKernel
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

np.random.seed(12345)

x = np.random.uniform(0, 10, 40)[:,None]
x.sort(0)
starts, stops = np.arange(0, 10, 3), np.arange(1, 11, 3)

k_lin = LinearSlopeBasisFuncKernel(1, starts, stops, variance=1., ARD=1)
Phi = k_lin.phi(x)
_ = plt.plot(x, Phi)

k = GPy.kern.Matern32(1, .3)
Kf = k.K(x)

k_per = GPy.kern.PeriodicMatern32(1, variance=100, period=1)
k_per.period.fix()
k_dom = DomainKernel(1, 1., 5.)
k_perdom = k_per * k_dom
Kpd = k_perdom.K(x)

np.random.seed(1234)
alpha = np.random.gamma(3, 1, Phi.shape[1])
w = np.random.normal(0, alpha)[:,None]

f_SE = np.random.multivariate_normal(np.zeros(x.shape[0]), Kf)[:, None]
f_perdom = np.random.multivariate_normal(np.zeros(x.shape[0]), Kpd)[:, None]
f_w = Phi.dot(w)

f = f_SE + f_w + f_perdom

y = f + np.random.normal(0, .1, f.shape)

plt.plot(x, f_w)
_ = plt.plot(x, y)
# Make sure the function is driven by the linear trend, as there can be a difficulty in identifiability.

k = (GPy.kern.Bias(1) 
     + GPy.kern.Matern52(1) 
     + LinearSlopeBasisFuncKernel(1, ARD=1, start=starts, stop=stops, variance=.1, name='linear_slopes')
     + k_perdom.copy()
    )

k.randomize()
m = GPy.models.GPRegression(x, y, k)

m.checkgrad()

m.optimize()

m.plot()

x_pred = np.linspace(0, 10, 500)[:,None]
pred_SE, var_SE = m._raw_predict(x_pred, kern=m.kern.Mat52)
pred_per, var_per = m._raw_predict(x_pred, kern=m.kern.mul)
pred_bias, var_bias = m._raw_predict(x_pred, kern=m.kern.bias)
pred_lin, var_lin = m._raw_predict(x_pred, kern=m.kern.linear_slopes)

m.plot_f(resolution=500, predict_kw=dict(kern=m.kern.Mat52), plot_data=False)
plt.plot(x, f_SE)

m.plot_f(resolution=500, predict_kw=dict(kern=m.kern.mul), plot_data=False)
plt.plot(x, f_perdom)

m.plot_f(resolution=500, predict_kw=dict(kern=m.kern.linear_slopes), plot_data=False)
plt.plot(x, f_w)

w_pred, w_var = m.kern.linear_slopes.posterior_inf()

df = pd.DataFrame(w, columns=['truth'], index=np.arange(Phi.shape[1]))
df['mean'] = w_pred
df['std'] = np.sqrt(w_var.diagonal())

np.round(df, 2)


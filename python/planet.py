get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPython.matplotlib.backend = "retina"')
from matplotlib import rcParams
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 150

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from maelstrom.kepler import kepler

kicid = 7917485
# kicid = 9837267

data = np.loadtxt("../data/kic{0}_lc.txt".format(kicid))
fulltimes = data[:, 0] # days
tmid = 0.5*(fulltimes[0] + fulltimes[-1])
times = fulltimes - tmid
dmmags = data[:, 1] * 1000. # mmags

# times = times[2500:]
# dmmags = dmmags[2500:]

metadata = np.loadtxt("../data/kic{0}_metadata.csv".format(kicid), delimiter=",", skiprows=1)

plt.plot(times,dmmags)

nu_arr = metadata[::6]
# nu_arr = nu_arr[:2]
# m = np.ones_like(nu_arr, dtype=bool)
# m[[0, 3, 4, 5]] = False
# nu_arr = nu_arr[m]
# nu_arr = nu_arr[[0] + list(range(3, len(nu_arr)))]
nu_arr

orbits = pd.read_csv("data/orbits.csv").rename(columns = lambda x: x.strip())
orbits.columns

orb_params = orbits[orbits.Name == "kic{0}".format(kicid)].iloc[0]
porb = orb_params.Porb
a1 = orb_params["a1sini/c"]
tp = orb_params["t_p"] - tmid
e = orb_params["e"]
varpi = orb_params["varpi"]
a1d = a1/86400.0
e_param = np.log(e) - np.log(1.0 - e)

a1d

sess = tf.InteractiveSession()

T = tf.float64

# First the variables that we might want to optimize:
porb_tensor = tf.Variable(porb, dtype=T)
tp_tensor = tf.Variable(tp, dtype=T)
nu_tensor = tf.Variable(nu_arr, dtype=T)
e_param_tensor = tf.Variable(e_param, dtype=T)  # This forces the ecc to be between 0 and 1
e_tensor = 1.0 / (1.0 + tf.exp(-e_param_tensor))
varpi_tensor = tf.Variable(varpi, dtype=T)
log_sigma2_tensor = tf.Variable(0.0, dtype=T)  # Variance from observational uncertainties and model misspecification

ad_tensor = tf.Variable(a1d + np.zeros_like(nu_arr), dtype=T)

# These are some placeholders for the data:
times_tensor = tf.placeholder(T, times.shape)
dmmags_tensor = tf.placeholder(T, dmmags.shape)

# Solve Kepler's equation
mean_anom = 2.0 * np.pi * (times_tensor - tp_tensor) / porb_tensor
ecc_anom = kepler(mean_anom, e_tensor)
true_anom = 2.0 * tf.atan2(tf.sqrt(1.0+e_tensor)*tf.tan(0.5*ecc_anom), tf.sqrt(1.0-e_tensor) + tf.zeros_like(times_tensor))

# Here we define how the time delay will be calculated:
tau_tensor = -(1.0 - tf.square(e_tensor)) * tf.sin(true_anom + varpi_tensor) / (1.0 + e_tensor*tf.cos(true_anom))

# And the design matrix:
arg_tensor = 2.0 * np.pi * nu_tensor[None, :] * (times_tensor[:, None] - ad_tensor[None, :] * tau_tensor[:, None])
D_tensor = tf.concat([tf.cos(arg_tensor), tf.sin(arg_tensor)], axis=1)

# Define the linear solve for W_hat:
DTD_tensor = tf.matmul(D_tensor, D_tensor, transpose_a=True)
DTy_tensor = tf.matmul(D_tensor, dmmags_tensor[:, None], transpose_a=True)
W_hat_tensor = tf.linalg.solve(DTD_tensor, DTy_tensor)

# Finally, the model and the chi^2 objective:
model_tensor = tf.squeeze(tf.matmul(D_tensor, W_hat_tensor))
chi2_tensor = tf.reduce_sum(tf.square(dmmags_tensor - model_tensor)) * tf.exp(-log_sigma2_tensor)
chi2_tensor += len(times) * log_sigma2_tensor

# We need to initialize the variables:
tf.global_variables_initializer().run()

# We'll also need to pass in the data:
data = {times_tensor: times, dmmags_tensor: dmmags}

# Let's plot the initial time delay
initial_tau = sess.run(tau_tensor, feed_dict=data)
plt.plot(times+tmid, initial_tau, ".", ms=2)
plt.ylabel(r"$\tau(t)$")
plt.xlabel("$t$");

initial_model = sess.run(model_tensor, feed_dict=data)
plt.plot(times, dmmags, ".k")
plt.plot(times, initial_model)
# plt.xlim(100, 102)
# plt.ylim(-75, 75)
plt.xlabel("t")
plt.ylabel("L(t)");

old_chi2 = sess.run(chi2_tensor, feed_dict=data)
for i in range(5):
    params = [log_sigma2_tensor, porb_tensor, tp_tensor]
    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2_tensor, params, method="L-BFGS-B")
    opt.minimize(sess, feed_dict=data)
    
    params.append(ad_tensor)
    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2_tensor, params, method="L-BFGS-B")
    opt.minimize(sess, feed_dict=data)

    params += [e_param_tensor, varpi_tensor]
    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2_tensor, params, method="L-BFGS-B")
    opt.minimize(sess, feed_dict=data)
        
    new_chi2 = sess.run(chi2_tensor, feed_dict=data)
    print(old_chi2 - new_chi2)
    if np.abs(old_chi2 - new_chi2) < 1.0:
        break
    old_chi2 = new_chi2

final_tau = sess.run(tau_tensor, feed_dict=data)
plt.plot(times+tmid, initial_tau, ".", ms=2)
plt.plot(times+tmid, final_tau, ".", ms=2)
plt.ylabel(r"$\tau(t) / a$")
plt.xlabel("$t$");

models = tau_tensor[:, None] * ad_tensor[None, :]
plt.plot(times+tmid, sess.run(models, feed_dict=data), ".", ms=3)
plt.ylabel(r"$\tau(t)$")
plt.xlabel("$t$");

ivar = -np.diag(sess.run(tf.hessians(-0.5*chi2_tensor, ad_tensor), feed_dict=data)[0])
ad = sess.run(ad_tensor)
ad *= np.sign(ad[0])
sig = 1.0 / np.sqrt(ivar)
plt.errorbar(np.arange(len(ad)), ad, yerr=sig, fmt=".");

m = np.ones_like(ad, dtype=bool)
while True:
    var = 1.0 / np.sum(ivar[m])
    mu = np.sum(ivar[m] * ad[m]) * var
    var2 = np.sum(ivar[m] * (mu - ad[m])**2) * var
    m_new = np.abs(ad - mu) / np.sqrt(var) < 7.0
    if m.sum() == m_new.sum():
        m = m_new
        break
    m = m_new
ad = ad[m]
ivar = ivar[m]

m

sig = 1.0 / np.sqrt(ivar)
plt.errorbar(np.arange(len(ad)), ad, yerr=sig, fmt=".");

ad * np.sqrt(ivar)

if np.any(ad * np.sqrt(ivar) < -1.0):
    m2 = ad * np.sqrt(ivar) < -1.0
    m1 = ~m2
    ad = [
        np.sum(ivar[m1]*ad[m1]) / np.sum(ivar[m1]),
        np.sum(ivar[m2]*ad[m2]) / np.sum(ivar[m2]),
    ]
else:
    ad = [np.sum(ivar*ad) / np.sum(ivar)]

ad

if len(ad) > 1:
    inds = tf.cast(0.5 - 0.5 * (ad_tensor / tf.abs(ad_tensor)), tf.int32)
else:
    inds = tf.zeros_like(ad_tensor, dtype=tf.int32)
ad_params = tf.Variable(ad, dtype=T)
sess.run(ad_params.initializer)
ad_tensor = tf.gather(ad_params, inds)

# And the design matrix:
arg_tensor = 2.0 * np.pi * nu_tensor[None, :] * (times_tensor[:, None] - ad_tensor[None, :] * tau_tensor[:, None])
D_tensor = tf.concat([tf.cos(arg_tensor), tf.sin(arg_tensor)], axis=1)

# Define the linear solve for W_hat:
DTD_tensor = tf.matmul(D_tensor, D_tensor, transpose_a=True)
DTy_tensor = tf.matmul(D_tensor, dmmags_tensor[:, None], transpose_a=True)
W_hat_tensor = tf.linalg.solve(DTD_tensor, DTy_tensor)

# Finally, the model and the chi^2 objective:
model_tensor = tf.squeeze(tf.matmul(D_tensor, W_hat_tensor))
chi2_tensor = tf.reduce_sum(tf.square(dmmags_tensor - model_tensor)) * tf.exp(-log_sigma2_tensor)
chi2_tensor += len(times) * log_sigma2_tensor

old_chi2 = sess.run(chi2_tensor, feed_dict=data)
for i in range(5):
    params = [log_sigma2_tensor, porb_tensor, tp_tensor]
    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2_tensor, params, method="L-BFGS-B")
    opt.minimize(sess, feed_dict=data)
    
    params.append(ad_params)
    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2_tensor, params, method="L-BFGS-B")
    opt.minimize(sess, feed_dict=data)

    params += [e_param_tensor, varpi_tensor]
    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2_tensor, params, method="L-BFGS-B")
    opt.minimize(sess, feed_dict=data)
    
    params.append(nu_tensor)
    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2_tensor, params, method="L-BFGS-B")
    opt.minimize(sess, feed_dict=data)
    
    new_chi2 = sess.run(chi2_tensor, feed_dict=data)
    print(old_chi2 - new_chi2)
    if np.abs(old_chi2 - new_chi2) < 1.0:
        break
    old_chi2 = new_chi2

models = tau_tensor[:, None] * ad_tensor[None, :]
plt.plot(times+tmid, 86400.0 * sess.run(models, feed_dict=data), ".", ms=2);
plt.ylabel(r"$\tau(t)$")
plt.xlabel("$t$");

sess.run(e_tensor), e

hess_tensor = tf.hessians(-0.5*chi2_tensor, params[:-1])

hess = sess.run(hess_tensor, feed_dict=data)

1. / np.sqrt(-hess[1])

hess

np.sqrt(-np.diag(np.linalg.inv(hess[3])))

sess.run(ad_tensor)

a1d / np.sqrt(-np.diag(np.linalg.inv(hess[3])))

porb

sess.run(porb_tensor)






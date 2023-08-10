get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPython.matplotlib.backend = "retina"')
from matplotlib import rcParams
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 150

import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from astropy.stats import LombScargle

from maelstrom.maelstrom import Maelstrom
from maelstrom.model import InterpMaelstrom
from maelstrom.estimator import estimate_frequencies

kicid = 11754974
data = np.loadtxt("data/kic{0}_lc.txt".format(kicid))
fulltimes = data[:, 0] # days
tmid = 0.5*(fulltimes[0] + fulltimes[-1])
times = fulltimes - tmid
dmmags = data[:, 1] * 1000. # mmags

metadata = np.loadtxt("data/kic{0}_metadata_more.csv".format(kicid), delimiter=",", skiprows=1)
nu_arr = metadata[::6]

plt.plot(times, dmmags, "k")
plt.xlabel(r"$t$")
plt.ylabel(r"$L(t)$");

nnu = 8
nu_arr

N = 1000
model = InterpMaelstrom(times, dmmags, nu_arr[:nnu], log_sigma2=0.0,
                        interp_x=np.linspace(times.min(), times.max(), N),
                        interp_y=1e-5*np.random.randn(N))

plt.plot(times, model.run(model.tau)[:, 0], ".k", ms=1)
plt.ylabel(r"$\tau(t)$")
plt.xlabel("$t$");

model.optimize([model.log_sigma2, model.nu])

opt = tf.train.AdamOptimizer().minimize(
    model.chi2, var_list=[model.interp_y, model.log_sigma2])
model.run(tf.global_variables_initializer())

bar = tqdm.trange(1000)
for i in bar:
    chi2, _ = model.run([model.chi2, opt])
    bar.set_postfix(chi2="{0:.0f}".format(chi2))

plt.plot(times, model.run(model.tau)[:, 0], ".k", ms=1)
plt.ylabel(r"$\tau(t)$")
plt.xlabel("$t$");

bar = tqdm.trange(1000)
for i in bar:
    chi2, _ = model.run([model.chi2, opt])
    bar.set_postfix(chi2="{0:.0f}".format(chi2))

plt.plot(times, model.run(model.tau)[:, 0], ".k", ms=1)
plt.ylabel(r"$\tau(t)$")
plt.xlabel("$t$");

plt.plot(times, model.run(model.tau)[:, 0], ".k", ms=1)
plt.xlim(-600, -500)
plt.ylabel(r"$\tau(t)$")
plt.xlabel("$t$");

x, y = model.run([model.time, model.psi])
df = 0.5 / (times.max() - times.min())
freq = np.arange(df, 1.0, 0.1*df)
power = LombScargle(x, y).power(freq)

peaks = estimate_frequencies(x, y, max_peaks=2)
print(1.0 / peaks)

plt.loglog(freq, power, "k")
plt.ylabel("power")
plt.xlabel("frequency [1 / day]");

kep_model_1 = Maelstrom(times, dmmags, nu_arr[:nnu])
kep_model_1.init_from_orbit(period=1.0 / peaks[0], lighttime=np.std(y), tref=0.0)
kep_model_1.pin_lighttime_values()
for i in range(3):
    kep_model_1.optimize([kep_model_1.log_sigma2])
    kep_model_1.optimize([kep_model_1.lighttime, kep_model_1.tref])
    kep_model_1.optimize([kep_model_1.period, kep_model_1.lighttime, kep_model_1.tref])
    kep_model_1.optimize([kep_model_1.period, kep_model_1.lighttime, kep_model_1.tref,
                          kep_model_1.eccen_param, kep_model_1.varpi])
    kep_model_1.optimize([kep_model_1.nu])

plt.plot(times, model.run(model.tau)[:, 0], ".k", ms=1)
plt.plot(times, kep_model_1.run(kep_model_1.tau)[:, 0], ".g", ms=1)
plt.ylabel(r"$\tau(t)$")
plt.xlabel("$t$");

tau1 = kep_model_1.run(kep_model_1.tau)[:, 0]
kep_model_2 = Maelstrom(times - tau1, dmmags, nu_arr[:nnu])
kep_model_2.init_from_orbit(period=1.0 / peaks[1], lighttime=0.5*np.std(y), tref=0.0)
kep_model_2.pin_lighttime_values()
for i in range(3):
    kep_model_2.optimize([kep_model_2.log_sigma2])
    kep_model_2.optimize([kep_model_2.lighttime, kep_model_2.tref])
    kep_model_2.optimize([kep_model_2.period, kep_model_2.lighttime, kep_model_2.tref])
    kep_model_2.optimize([kep_model_2.period, kep_model_2.lighttime, kep_model_2.tref,
                          kep_model_2.eccen_param, kep_model_2.varpi, kep_model_2.tref])
    kep_model_2.optimize([kep_model_2.nu])

period = kep_model_2.run(kep_model_2.period)
plt.plot(times % period, model.run(model.tau)[:, 0] - tau1, ".k", ms=1)
plt.plot(times % period, kep_model_2.run(kep_model_2.tau)[:, 0], ".g", ms=1)
plt.ylabel(r"$\tau(t) - \tau_1(t)$")
plt.xlabel("orbital phase");

print("period: {0:.5f} day".format(kep_model_2.run(kep_model_2.period)))
print("a1 sin(i) / c: {0:.5f} sec".format(86400.0 * np.abs(kep_model_2.run(kep_model_2.lighttime[0]))))

resid = dmmags - model.run(model.model)
x = (times - model.run(model.tau)[:, 0]) % period

bins = np.linspace(0, period, 200)
num, _ = np.histogram(x, bins, weights=resid)
denom, _ = np.histogram(x, bins)

plt.scatter(x, resid, c=times, s=2)
plt.plot(0.5*(bins[1:]+bins[:-1]), num / denom)






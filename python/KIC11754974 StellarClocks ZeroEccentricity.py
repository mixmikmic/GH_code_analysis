get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op

data = np.loadtxt("data/kic11754974_lc.txt")
fulltimes = data[:, 0] # days
tmid = 0.5*(fulltimes[0] + fulltimes[-1])
times = fulltimes - tmid
dmmags = data[:, 1] * 1000. # mmags
metadata = np.loadtxt("data/kic11754974_metadata.csv", delimiter=",", skiprows=1)
tdelay = np.loadtxt("data/kic11754974_unwrapped_time_delay.csv", delimiter=",", skiprows=1)
tdzero = float(next(open("data/kic11754974_unwrapped_time_delay.csv")))
tdtime = tdelay[:,0] + tdzero
tdtau = tdelay[:,1] / 86400
tdunc = tdelay[:,2] / 86400

plt.plot(times,dmmags)

plt.errorbar(tdtime,tdtau,tdunc,fmt=".")

nu_arr = metadata[::6]
nu_arr

orbits = pd.read_csv("data/orbits.csv").rename(columns = lambda x: x.strip())

orbits.columns

orb_params = orbits[orbits.Name == "kic11754974"].iloc[0]
porb = orb_params.Porb
a1 = orb_params["a1sini/c"]
tp = orb_params["t_p"] - tmid
a1d = a1/86400.0

def get_tau(times, porb, a1d, tp):
    return -a1d * np.sin(2 * np.pi * (times - tp)/ porb)
    
def get_design_matrix(times, nu_arr, porb, a1d, tp):
    tau = get_tau(times, porb, a1d, tp)
    arg = 2 * np.pi * nu_arr[None, :] * (times - tau)[:, None]
    N_nu = len(nu_arr)
    D = np.empty((len(times),2*N_nu))
    D[:,:N_nu] = np.cos(arg)
    D[:,N_nu:] = np.sin(arg)
    return D

def get_W_hat(D, dmmags):
    return np.linalg.solve(np.dot(D.T, D), np.dot(D.T, dmmags))

D = get_design_matrix(times, nu_arr, porb, a1d, tp)

W_hat = get_W_hat(D, dmmags)
W_hat

def chisq(times, dmmags, nu_arr, porb, a1d, tp):
    D = get_design_matrix(times, nu_arr, porb, a1d, tp)
    W_hat = get_W_hat(D, dmmags)
    model = np.dot(D, W_hat)
    chi_sq = np.sum((model - dmmags)**2) # no sigma yet
    return chi_sq

chisq(times, dmmags, nu_arr, porb, a1d, tp)

t0s = tp + np.linspace(0, porb,500)
chisqs = np.empty_like(t0s)
for i, t0 in enumerate(t0s):
    chisqs[i] = chisq(times, dmmags, nu_arr, porb, a1d, t0)

plt.plot(t0s, chisqs)
ind = np.argmin(chisqs)
plt.plot(t0s[ind], chisqs[ind], "or")

nus = np.linspace(16, 17, 500)
chisqs = np.empty_like(nus)
for i, nu_something in enumerate(nus):
    chisqs[i] = chisq(times, dmmags, nu_something, porb, a1d, tp)

plt.plot(nus, chisqs)
ind = np.argmin(chisqs)
plt.plot(nus[ind], chisqs[ind], "or")

nus[ind]-nu

plt.plot(times+tmid, get_tau(times, porb, a1d, tp))
plt.plot(times+tmid, get_tau(times, porb+1, a1d, tp))

periods = porb + np.linspace(-50, 50,1000)
chisqs = np.empty_like(periods)
for i, period in enumerate(periods):
    chisqs[i] = chisq(times, dmmags, nu_arr, period, a1d, tp)

plt.plot(periods, chisqs)
ind = np.argmin(chisqs)
plt.plot(periods[ind], chisqs[ind], "or")

def objective(orbparams, times, dmmags, nu_arr):
    return chisq(times, dmmags, nu_arr, *orbparams)

init = [porb, a1d, tp]
soln = op.minimize(objective, init, args=(times, dmmags, nu_arr))
soln

plt.plot(times+tmid, get_tau(times, porb, a1d, tp))
plt.plot(times+tmid, get_tau(times, *(soln.x)))
plt.errorbar(tdtime,tdtau,tdunc,fmt=".")




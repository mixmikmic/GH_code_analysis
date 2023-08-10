get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pystan
import glob
import pandas as pd
from keplerdata import Dataset
plt.rcParams['figure.figsize'] = [12,10]
from tqdm import tqdm

datadir = '/Users/davies/Dropbox/K2_seismo_pipes/20Stars/Data/'
sfiles = glob.glob(datadir + '*.dat')
df = pd.read_csv(datadir + '20stars.csv')
#df = df[0:8]
dss = [Dataset(row.EPIC, datadir + 'kplr' + str(int(row.EPIC)) + '_llc_concat.dat') for idx, row in df.iterrows()]
for ds in tqdm(dss):
    ds.power_spectrum(dfN=(0.01, 28800))
    ds.rebin_quick(100)
power = [ds.smoo_power for ds in dss]
freq = dss[0].smoo_freq

data = np.zeros([len(dss[0].smoo_power), len(dss)])
for idx, p in enumerate(power):
    data[:,idx] = p
print(df)
numaxs = df.numax.values
kps = df.kic_kepmag.values
teffs = df.Teff.values
dnus = df.Dnu.values
masses = (numaxs / 3090.0)**3 * (dnus / 135.1)**-4 * (teffs / 5777.0)**1.5
#freq = freq[2:]
#data = data[2:, :]

fig, ax = plt.subplots()
for idx, d in enumerate(data[0,:]):
    plt.plot(freq, data[:, idx])
ax.plot(freq, data[:,0])
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Frequency')
ax.set_ylabel('Rebinned power')

whites = np.median(data[-3:,:], axis=0)

fig, ax = plt.subplots()
CS = ax.scatter(kps, whites, c=numaxs)
cbar = fig.colorbar(CS)
ax.set_xlabel('Kp')
ax.set_ylabel('High frequency noise')

def harvey(f, a, b, c=4.0):
    return 0.9*a**2/b/(1 + (f/b)**c)
def gaussian(f, numax, width, height):
    return height * np.exp(-0.5 * (f - numax)**2 / width**2)
def apod(f, nyq):
    x = f / 2.0 / nyq
    
    return np.sinc(x)**2
def model(f, ak, ae, at, bk, be, ck, ce, ct, dk, de, numax, wk, we, wt, hk, he, white, mass):
    a = 10**(ak + numax*ae + mass*at)
    b = 10**(bk + numax*be)
    c = 10**(ck + numax*ce + mass*ct)
    d = 10**(dk + numax*de)
    width = 10**(wk + numax*we + mass*wt)
    height = 10**(hk + numax*he)
    model = (apod(f, 288.8) * (harvey(f, a, b) + harvey(f, c, d) + gaussian(f, 10**numax, width, height))) + 10**white
    return model
ak, ae, at = 3.52, -0.58, -0.26
bk, be = -0.48, 0.94
ck, ce, ct = 3.55, -0.64, -0.26
dk, de = 0.01, 0.97
wk, we = -0.83, 1.0
hk, he, wt = 6.93, -2.14, -0.3
print(ak, ck, hk)
n = 4
fig, ax = plt.subplots(n,n, figsize=[12, 12])
for i in range(0, n*n):
    x = i // n
    y = i % n
    fit0 = model(freq, 
                 ak, ae, at, bk, be, 
                 ck, ce, ct, dk, de, np.log10(numaxs[i]), 
                 wk, we, wt, hk, he, 
                 np.log10(whites[i]), np.log10(masses[i]))
    ax[x,y].plot(freq, data[:,i], 'r-')
    ax[x,y].plot(freq, fit0, 'k-')
    ax[x,y].set_xscale('log')
    ax[x,y].set_yscale('log')
    ax[x, y].set_ylim([1, 1e5])
    ax[x, y].set_xlim([1 ,388])
    ax[x, y].axvline(numaxs[i], c='b')
    ax[x, y].set_yticks([])
    ax[x, y].set_xticks([])
    ax[x, y].text(2, 1e1, str(np.round(teffs[i])))
    ax[x, y].text(2, 1e2, str(np.round(masses[i] * 10.)/10.0))

print(numaxs)
print(whites)

code = '''
functions {
    real harvey(real f, real a, real b, real c){
        return 0.9*a^2/b/(1.0 + (f/b)^c);
    }
    real gaussian(real f, real numax, real width, real height){
        return height * exp(-0.5 * (f - numax)^2 / width^2);
    }
    real apod(real f, real nyq){
        real x = 3.14 / 2.0 * f / nyq;
        return (sin(x) / x)^2;
    }
}
data {
    int N;
    int M;
    int dof;
    vector[N] f;
    real p[N, M];
    vector[M] numax_est;
    vector[M] white_est;
    vector[M] mass_est;

}
parameters {
    real<lower = 0> numax[M];
    real<lower = 0> white[M];
    real mass[M];
    real ae;
    real be;
    real ce;
    real de;
    real ak;
    real bk;
    real ck;
    real dk;
    real at;
    real ct;
    real wk;
    real we;
    real wt;
    real hk;
    real he; 
}
transformed parameters {
    vector[M] a;
    vector[M] b;
    vector[M] c;
    vector[M] d;
    vector[M] w;
    vector[M] h;
    for (j in 1:M){
        a[j] = 10^(ak + numax[j] * ae + mass[j] * at);
        b[j] = 10^(bk + numax[j] * be);
        c[j] = 10^(ck + numax[j] * ce + mass[j] * ct);
        d[j] = 10^(dk + numax[j] * de);
        w[j] = 10^(wk + numax[j] * we + mass[j] * wt);
        h[j] = 10^(hk + numax[j] * he);
    }
}
model {
    real beta[N, M];
    for (j in 1:M){
        for (i in 1:N){
            beta[i, j] = dof / (apod(f[i], 288.8)
                        * (harvey(f[i], a[j], b[j], 4.0)
                        + harvey(f[i], c[j], d[j], 4.0)
                        + gaussian(f[i], 10^numax[j], w[j], h[j]))
                        + 10^white[j]);
        }
        p[1:N, j] ~ gamma(dof, beta[1:N, j]);
    }
    ak ~ normal(3.45, 0.01); // log10
    ae ~ normal(-0.51, 0.01);
    at ~ normal(-0.26, 0.01);
    bk ~ normal(-0.46, 0.01); //log10
    be ~ normal(0.89, 0.01);
    ck ~ normal(3.59, 0.01); // log10
    ce ~ normal(-0.61, 0.01);
    ct ~ normal(-0.39, 0.01);
    dk ~ normal(0.04, 0.01); //log10
    de ~ normal(0.95, 0.01);
    wk ~ normal(-0.83, 0.005); // log10
    we ~ normal(1.0, 0.01);
    wt ~ normal(-0.3, 0.01);
    hk ~ normal(6.93, 0.008); // log10
    he ~ normal(-2.18, 0.005);
    numax ~ normal(numax_est, 0.01);
    white ~ normal(white_est, 0.01);
    mass ~ normal(mass_est, 0.03);
}
'''
sm = pystan.StanModel(model_code=code, model_name='backfit')

dat = {'N': len(freq),
       'M': len(numaxs),
       'dof': 100,
      'f': freq,
      'p': data,
      'numax_est': np.log10(numaxs),
      'white_est': np.log10(whites),
      'mass_est': np.log10(masses)}
# Note dof is not actually dof but is in fact the number of bins over which we have smoothed.
fit = sm.sampling(data=dat, iter=3000, chains=1)
fit.plot()

print(fit)

diff = 10**fit['numax'].mean(axis=0) - numaxs
print(diff)

n = 4
fig, ax = plt.subplots(n,n, figsize=[12, 12])
ak = fit['ak'].mean()
ae = fit['ae'].mean()
at = fit['at'].mean()
bk = fit['bk'].mean()
be = fit['be'].mean()
ck = fit['ck'].mean()
ce = fit['ce'].mean()
ct = fit['ct'].mean()
dk = fit['dk'].mean()
de = fit['de'].mean()
wk = fit['wk'].mean()
we = fit['we'].mean()
wt = fit['wt'].mean()
hk = fit['hk'].mean()
he = fit['he'].mean()
numax = fit['numax'].mean(axis=0)
white = fit['white'].mean(axis=0)
mass = fit['mass'].mean(axis=0)
mass_err = np.std(fit['mass'], axis=0)
for i in range(0, n*n):
    fit0 = model(freq, ak, ae, at, bk, be, ck, ce, ct, dk, de, numax[i], wk, we, wt, hk, he, white[i], mass[i])
    x = i // n
    y = i % n
    ax[x, y].plot(freq, data[:,i], 'r-')
    ax[x, y].plot(freq, fit0, 'k-')
    ax[x, y].set_xscale('log')
    ax[x, y].set_yscale('log')
    ax[x, y].set_ylim([1, 1e6])
    ax[x, y].set_xlim([1 ,288])
    ax[x, y].axvline(10**numax[i], c='g', linestyle='--')
    ax[x, y].axvline(numaxs[i], c='b')
    ax[x, y].set_yticks([])
    ax[x, y].set_xticks([])
    ax[x, y].text(3, 10, str(np.round(teffs[i])))
    ax[x, y].text(3, 100, str(np.round(10**mass[i] * 100)/100.0))




get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from scipy.optimize import curve_fit
import markdown
import sys
sys.path.append('/Users/vs/Dropbox/Python')
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
import seaborn as sns
import os
import glob
import linecache
import gloess_fits as gf
import re
from IPython.display import Image
import itertools
import reddening_laws as red
from astropy.stats import sigma_clip
import scipy.optimize as op
import emcee
import corner
from matplotlib.ticker import MaxNLocator
from astroquery.irsa_dust import IrsaDust
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap





bigfontsize=20
labelfontsize=16
tickfontsize=16
sns.set_context('talk')
mp.rcParams.update({'font.size': bigfontsize,
                     'axes.labelsize':labelfontsize,
                     'xtick.labelsize':tickfontsize,
                     'ytick.labelsize':tickfontsize,
                     'legend.fontsize':tickfontsize,
                     })

path = '/Users/vs/Dropbox/Gaia/'
os.chdir(path)

gaia_df = pd.read_csv('vizer_crossmatch.tsv', skiprows=166, skipinitialspace=True, names=('input', 'rad', 'HIP', 'TYC2', 'SolID', 'Source', 'RandomI', 'Epoch', 'RA_ICRS', 'e_RA_ICRS', 'DE_ICRS', 'e_DE_ICRS', 'Plx', 'e_Plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'RADEcor', 'RAPlxcor', 'RApmRAcor', 'RApmDEcor', 'DEPlxcor', 'DEpmRAcor', 'DEpmDEcor', 'PlxpmRAcor', 'PlxpmDEcor', 'pmRApmDEcor', 'NAL', 'NAC', 'NgAL', 'NgAC', 'NbAL', 'NbAC', 'DQ', 'epsi', 'sepsi', 'APF', 'ARF', 'WAL', 'WAC', 'Apr', 'MatchObs', 'Dup', 'sK1', 'sK2', 'sK3', 'sK4', 'mK1', 'mK2', 'mK3', 'mK4', 'o_<Gmag>', '<FG>', 'e_<FG>', '<Gmag>', 'Var', 'GLON', 'GLAT', 'ELON', 'ELAT'), na_values='NOT_AVAILABLE', sep=';', comment='#')
gaia_df['ID'] = gaia_df.input.str.split('\t',0).str.get(0)

gaia_df = gaia_df.drop('input', 1)
#gaia_df = gaia_df.drop('rad', 1)
gaia_df = gaia_df.replace('', np.nan)
gaia_df['ID'] = gaia_df['ID'].replace(regex=True, to_replace=r' ',value='_')
gaia_df['id_compare'] = map(str.lower, gaia_df.ID)
gaia_df['id_compare'] = gaia_df['id_compare'].replace(regex=True, to_replace=r'_',value='')

info_df = pd.read_csv('rrl_average_mags', delim_whitespace=True)
info_df

printcols = ['ID', 'rad']
gaia_df[printcols]
gaia_df = gaia_df.groupby(['ID']).min()
#gaia_df = gaia_df.reset_index(drop=True)

merged_df = info_df.merge(gaia_df, on='id_compare')
useful = ['Name', 'Period', 'Type', 'mag_3p6', 'err_3p6', 'amp_3p6', 'mag_4p5', 'err_4p5', 'amp_4p5', 'RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx', 'id_compare']
analysis_df = merged_df[useful]
analysis_df = analysis_df.reset_index(drop=True)
analysis_df

Image("gould_kollmeier_abstract.png")

analysis_df['e_gks'] = analysis_df.apply(lambda x : np.sqrt((0.79*x['e_Plx'])**2 - (0.10)**2), axis=1)

analysis_df

def grab_extinction(row):
    star = row.Name
    ra = row.RA_ICRS
    dec = row.DE_ICRS
    coord_string = str(ra) + 'd ' +  str(dec) + 'd'
    C = coord.SkyCoord(coord_string, frame='fk5')
    table = IrsaDust.get_extinction_table(C)
    irac_1_sandf = table[19][3]
    irac_2_sandf = table[20][3]
    analysis_df.ix[analysis_df.Name==star, 'A_3p6'] = irac_1_sandf
    analysis_df.ix[analysis_df.Name==star, 'A_4p5'] = irac_2_sandf
    print star, irac_1_sandf, irac_2_sandf
    return(0)
    

analysis_df.apply(lambda line: grab_extinction(line), axis=1);

def abs_mag_errs(row):
    mag_36 = row.mag_3p6
    mag_45 = row.mag_4p5
    e_mag_36 = row.err_3p6
    e_mag_45 = row.err_4p5
    plx = row.Plx
    e_plx = row.e_gks
    a_36 = row.A_3p6
    a_45 = row.A_4p5
    star = row.Name
    
    d = 1./(plx * 1e-3)
    mu = 5.0*np.log10(d) - 5.0
    abs_36 = mag_36 - mu - a_36
    abs_45 = mag_45 - mu - a_45
    
    sigma_a = 0.005
    
    variance_abs_36 = e_mag_36**2 + ((5* e_plx**2)/(plx*np.log(10.)))**2 + sigma_a**2
    variance_abs_45 = e_mag_45**2 + ((5* e_plx**2)/(plx*np.log(10.)))**2 + sigma_a**2

    sigma_abs_36 = np.sqrt(variance_abs_36)
    sigma_abs_45 = np.sqrt(variance_abs_45)

    analysis_df.ix[analysis_df.Name==star, 'M_3p6'] = abs_36
    analysis_df.ix[analysis_df.Name==star, 'M_4p5'] = abs_45


    analysis_df.ix[analysis_df.Name==star, 'e_M_3p6'] = sigma_abs_36
    analysis_df.ix[analysis_df.Name==star, 'e_M_4p5'] = sigma_abs_45
    print star, abs_36, abs_45, variance_abs_36, variance_abs_45, sigma_abs_36, sigma_abs_45
    return(0)
    

analysis_df['M_3p6'] = np.nan
analysis_df['M_4p5'] = np.nan
analysis_df['e_M_3p6'] = np.nan
analysis_df['e_M_4p5'] = np.nan

analysis_df.apply(lambda line: abs_mag_errs(line), axis=1)

min(analysis_df.e_M_4p5), max(analysis_df.e_M_4p5)

analysis_df['log_P'] = np.log10(analysis_df['Period'])
analysis_df['logP_f'] = np.where(analysis_df['Type']=='c', analysis_df['log_P'] + 0.127, analysis_df['log_P'])

feast_df = pd.read_csv('Feast_2008.tsv', sep=';', skiprows=64, names=('HIP', 'Name', 'plx', 'e_plx', 'Vmag', 'Jmag', 'Hmag', 'Ksmag', 'Per', '[Fe/H]', 'E(B-V)', 'Type', 'Simbad', '_RA', '_DE'))

feast_df['id_compare'] = map(str.lower, feast_df.Name)
feast_df['id_compare'] = feast_df['id_compare'].replace(regex=True, to_replace=r' ',value='')
feast_metals = ['id_compare', '[Fe/H]']
analysis_df = analysis_df.merge(feast_df[feast_metals], on='id_compare')

analysis_df.to_csv('test_output', delim_whitespace=True, index=False, header=False, columns=('logP_f', 'M_3p6', 'e_M_3p6'))

Image("neeley_m4_pls.png")

### Analysing 3.6, allowing dispersion as a free parameter

x = analysis_df.logP_f
y = analysis_df.M_3p6
dy = analysis_df.e_M_3p6


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

def single_point_likelihoods(x, y, yvar, m, b, Pbad, Ybad, Vbad):   
    return np.array(((1 - Pbad) / np.sqrt(2.*np.pi*yvar) * np.exp(-0.5 * (y - m*x - b)**2 / yvar) +
            Pbad / np.sqrt(2.*np.pi * Vbad) * np.exp(-0.5 * (y - Ybad)**2 / Vbad)))

def likelihood(params, x, y, yvar):
    b = params[0]
    m = params[1]
    Pbad = params[2] ## prior probability a point is bad
    Ybad = params[3] ## Mean of bad points
    Vbad = params[4] ## Variance of bad points
    
    return np.array(np.prod(single_point_likelihoods(x, y, yvar, m, b, Pbad, Ybad, Vbad)))

    
    
def prior(params):
    b = params[0]
    m = params[1]
    Pb = params[2] ## prior probability a point is bad
    Yb = params[3] ## Mean of bad points
    Vb = params[4] ## Variance of bad points

    return np.array((Pbad >= 0) * (Pbad < 1) * (Vbad > 0))

def posterior(params, x, y, yerr):
    post = likelihood(params, x, y, yerr) * prior(params)
    if not np.isfinite(post):
        return -np.inf
    return post

def ln_like(params, x, y, yvar):
    b = params[0]
    m = params[1]
    Pbad = params[2] ## prior probability a point is bad
    Ybad = params[3] ## Mean of bad points
    Vbad = params[4] ## Variance of bad points
    
    return np.array(np.sum(single_point_likelihoods(x, y, yvar, m, b, Pbad, Ybad, Vbad)))

def ln_prior(params):
    b = params[0]
    m = params[1]
    Pb = params[2] ## prior probability a point is bad
    Yb = params[3] ## Mean of bad points
    Vb = params[4] ## Variance of bad points

    if (Pbad >= 0) and (Pbad < 1) and (Vbad > 0) and m > -10 and m < 10 and b < 10000 and b > -10000:
        return 0
    return -np.inf

def ln_posterior(params, x, y, yerr):
    post = ln_like(params, x, y, yerr) + ln_prior(params)
    if not np.isfinite(post):
        return -np.inf
    return post

## if fixed_slope=True then set the slopeval to the correct value, mscale to the fixed slope uncertainty.
## bscale should be set to the median uncertainty on the absolute magnitude for the sample.

def pick_new_parameters(nsteps, params, fixed_slope=False, slopeval=np.nan, mscale=np.nan, bscale=0.1):
    b = params[0]
    m = params[1]
    Pb = params[2] ## prior probability a point is bad
    Yb = params[3] ## Mean of bad points
    Vb = params[4] ## Variance of bad points
    if (fixed_slope == False):
        mscale = 0.1 ## from m4 paper

    
    if (fixed_slope == True):
        if not np.isfinite(slopeval):
            print 'you must enter a fixed slope value for slopeval'
            return -np.inf
        if not np.isfinite(mscale):
            print 'you must enter a fixed slope uncertinty value for mscale'
            return -np.inf
        m = slopeval

    # burn-in slope and intercept
    if nsteps > 10000:
        pbadscale = 0.1
        ybadscale = bscale
        vbadscale = 10.
    else:
        pbadscale = 0
        ybadscale = 0
        vbadscale = 0
    newb = b + bscale * np.random.normal()
    newm = m + mscale * np.random.normal()
    newPbad = Pbad + pbadscale * np.random.normal()
    newYbad = Ybad + ybadscale * np.random.normal()
    newVbad = Vbad + vbadscale * np.random.normal()
    newparams = np.array([newb, newm, newPbad, newYbad, newVbad])
    return (newparams)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.332, mscale=0.106, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        


sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 1)

palette = itertools.cycle(sns.color_palette())
col=next(palette)

xfit = np.arange(-1, 1, 0.1)

ax1 = mp.subplot(111)

mp.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_3p6'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
mp.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_3p6'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

#ax1.plot(xfit, b_ls + m_ls*xfit, 'k--')
mp.xlim(-0.6, -0.1)
mp.ylim(1, -3.0)


#plot_text = 'y = ({0:.3f} $\pm$ {1:.3f}) x + ({2:.3f} $\pm$ {3:.3f})'.format(m_ls, np.sqrt(cov[1][1]), b_ls, np.sqrt(cov[0][0]))
mp.annotate(plot_text, xy=(75, 75), xycoords='data', size=12)

xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

## Picking 50 chains at random to plot
Nchain = len(chain)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:50]
for i in I:
    (p, params) = chain[i]
    (b, m, Pbad, Ybad, Vbad) = params
    ys = m*xfit + b
    ax1.plot(xfit, ys, color='k', alpha=0.1)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

mp.plot(xfit, best_m*xfit + best_b, 'r--')

mp.ylabel('Absolute Magnitude 3.6 $\mu$m')
mp.xlabel('log P (days)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.suptitle('Whole TGAS sample, RRab + RRc (fundamentalised), M4 slope')

plot_text = 'M$_{4}$ = ({0:.3f} $\pm$ {1:.3f}) log P + ({2:.3f} $\pm$ {3:.3f})'.format(best_m, np.std(ms), best_b, np.std(bs), '{[3.6]}')
mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_3p6_m4slope_mcmc_rrab_rrc.pdf')

### Analysing 4.5

x = analysis_df.logP_f
y = analysis_df.M_4p5
dy = analysis_df.e_M_4p5


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.336, mscale=0.105, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 1)

palette = itertools.cycle(sns.color_palette())
col=next(palette)

xfit = np.arange(-1, 1, 0.1)

ax1 = mp.subplot(111)

mp.errorbar(analysis_df.logP_f, analysis_df.M_4p5, yerr = analysis_df.e_M_4p5, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_4p5'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
mp.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_4p5'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

#ax1.plot(xfit, b_ls + m_ls*xfit, 'k--')
mp.xlim(-0.6, -0.1)
mp.ylim(1, -3.0)


xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

## Picking 50 chains at random to plot
Nchain = len(chain)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:50]
for i in I:
    (p, params) = chain[i]
    (b, m, Pbad, Ybad, Vbad) = params
    ys = m*xfit + b
    ax1.plot(xfit, ys, color='k', alpha=0.1)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

mp.plot(xfit, best_m*xfit + best_b, 'r--')

mp.ylabel('Absolute Magnitude 4.5 $\mu$m')
mp.xlabel('log P (days)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.suptitle('Whole TGAS sample, RRab + RRc (fundamentalised), M4 slope')

plot_text = 'M$_{4}$ = ({0:.3f} $\pm$ {1:.3f}) log P + ({2:.3f} $\pm$ {3:.3f})'.format(best_m, np.std(ms), best_b, np.std(bs), '{[4.5]}')
mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_4p5_m4slope_mcmc_rrab_rrc.pdf')

### Analysing 3.6, RRab only

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

x = ab_df.log_P
y = ab_df.M_3p6
dy = ab_df.e_M_3p6


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.370, mscale=0.139, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 1)

palette = itertools.cycle(sns.color_palette())
col=next(palette)

xfit = np.arange(-1, 1, 0.1)

ax1 = mp.subplot(111)


mp.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')

#ax1.plot(xfit, b_ls + m_ls*xfit, 'k--')
mp.xlim(-0.6, -0.1)
mp.ylim(1, -3.0)


xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

## Picking 50 chains at random to plot
Nchain = len(chain)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:50]
for i in I:
    (p, params) = chain[i]
    (b, m, Pbad, Ybad, Vbad) = params
    ys = m*xfit + b
    ax1.plot(xfit, ys, color='k', alpha=0.1)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

mp.plot(xfit, best_m*xfit + best_b, 'r--')

mp.ylabel('Absolute Magnitude 3.6 $\mu$m')
mp.xlabel('log P (days)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.suptitle('Whole TGAS sample, RRab only, M4 slope')

plot_text = 'M$_{4}$ = ({0:.3f} $\pm$ {1:.3f}) log P + ({2:.3f} $\pm$ {3:.3f})'.format(best_m, np.std(ms), best_b, np.std(bs), '{[3.6]}')
mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_3p6_m4slope_mcmc_rrab_only.pdf')

### Analysing 4.5, RRab only

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

x = ab_df.log_P
y = ab_df.M_4p5
dy = ab_df.e_M_4p5


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.355, mscale=0.168, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 1)

palette = itertools.cycle(sns.color_palette())
col=next(palette)

xfit = np.arange(-1, 1, 0.1)

ax1 = mp.subplot(111)


mp.errorbar(ab_df.log_P, ab_df.M_4p5, yerr = ab_df.e_M_4p5, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(ab_df.log_P, ab_df.M_4p5, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')

#ax1.plot(xfit, b_ls + m_ls*xfit, 'k--')
mp.xlim(-0.6, -0.1)
mp.ylim(1, -3.0)


xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

## Picking 50 chains at random to plot
Nchain = len(chain)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:50]
for i in I:
    (p, params) = chain[i]
    (b, m, Pbad, Ybad, Vbad) = params
    ys = m*xfit + b
    ax1.plot(xfit, ys, color='k', alpha=0.1)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

mp.plot(xfit, best_m*xfit + best_b, 'r--')

mp.ylabel('Absolute Magnitude 4.5 $\mu$m')
mp.xlabel('log P (days)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.suptitle('Whole TGAS sample, RRab only, M4 slope')

plot_text = 'M$_{4}$ = ({0:.3f} $\pm$ {1:.3f}) log P + ({2:.3f} $\pm$ {3:.3f})'.format(best_m, np.std(ms), best_b, np.std(bs), '{[4.5]}')
mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_4p5_m4slope_mcmc_rrab_only.pdf')

### Analysing 3.6, RRc only

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

x = c_df.log_P
y = c_df.M_3p6
dy = c_df.e_M_3p6


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.658, mscale=0.428, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 1)

palette = itertools.cycle(sns.color_palette())
col=next(palette)

xfit = np.arange(-1, 1, 0.1)

ax1 = mp.subplot(111)


mp.errorbar(c_df.log_P, c_df.M_3p6, yerr = c_df.e_M_3p6, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(c_df.log_P, c_df.M_3p6, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

#ax1.plot(xfit, b_ls + m_ls*xfit, 'k--')
mp.xlim(-0.6, -0.1)
mp.ylim(1, -3.0)


xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

## Picking 50 chains at random to plot
Nchain = len(chain)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:50]
for i in I:
    (p, params) = chain[i]
    (b, m, Pbad, Ybad, Vbad) = params
    ys = m*xfit + b
    ax1.plot(xfit, ys, color='k', alpha=0.1)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

mp.plot(xfit, best_m*xfit + best_b, 'r--')

mp.ylabel('Absolute Magnitude 3.6 $\mu$m')
mp.xlabel('log P (days)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.suptitle('Whole TGAS sample, RRc only, M4 slope')

plot_text = 'M$_{4}$ = ({0:.3f} $\pm$ {1:.3f}) log P + ({2:.3f} $\pm$ {3:.3f})'.format(best_m, np.std(ms), best_b, np.std(bs), '{[3.6]}')
mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_3p6_m4slope_mcmc_rrc_only.pdf')

### Analysing 4.5, RRc only

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

x = c_df.log_P
y = c_df.M_4p5
dy = c_df.e_M_4p5


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.979, mscale=0.337, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 1)

palette = itertools.cycle(sns.color_palette())
col=next(palette)

xfit = np.arange(-1, 1, 0.1)

ax1 = mp.subplot(111)


mp.errorbar(c_df.log_P, c_df.M_4p5, yerr = c_df.e_M_4p5, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(c_df.log_P, c_df.M_4p5, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

#ax1.plot(xfit, b_ls + m_ls*xfit, 'k--')
mp.xlim(-0.6, -0.1)
mp.ylim(1, -3.0)


xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

## Picking 50 chains at random to plot
Nchain = len(chain)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:50]
for i in I:
    (p, params) = chain[i]
    (b, m, Pbad, Ybad, Vbad) = params
    ys = m*xfit + b
    ax1.plot(xfit, ys, color='k', alpha=0.1)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

mp.plot(xfit, best_m*xfit + best_b, 'r--')

mp.ylabel('Absolute Magnitude 4.5 $\mu$m')
mp.xlabel('log P (days)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.suptitle('Whole TGAS sample, RRc only, M4 slope')

plot_text = 'M$_{4}$ = ({0:.3f} $\pm$ {1:.3f}) log P + ({2:.3f} $\pm$ {3:.3f})'.format(best_m, np.std(ms), best_b, np.std(bs), '{[4.5]}')
mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_4p5_m4slope_mcmc_rrc_only.pdf')

### Analysing 3.6, RRab only

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

x = ab_df.log_P
y = ab_df.M_3p6
dy = ab_df.e_M_3p6


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.370, mscale=0.139, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        

sns.set_style("white")
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("husl", 10)

colors = sns.color_palette()

xfit = np.arange(-1, 1, 0.1)

ax1 = mp.subplot(111)


mp.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=colors[7], ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')

#ax1.plot(xfit, b_ls + m_ls*xfit, 'k--')
mp.xlim(-0.6, -0.1)
mp.ylim(1, -1.75)


xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

## Picking 25 chains at random to plot
Nchain = len(chain)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:25]
for i in I:
    (p, params) = chain[i]
    (b, m, Pbad, Ybad, Vbad) = params
    ys = m*xfit + b
    ax1.plot(xfit, ys, color='k', alpha=0.05)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

mp.plot(xfit, best_m*xfit + best_b, color=colors[7], ls='--')

mp.ylabel('Absolute Magnitude 3.6 $\mu$m')
mp.xlabel('log P (days)')
#mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.suptitle('Whole TGAS sample, RRab only, M4 slope')

if best_b < 0: 
    bsign = '-' 
else:
    bsign = '+'
if best_m < 0:
    msign = '-'
else:
    msign = ''
    

plot_text = 'M$_{4}$ = ${5}${0:.3f} ($\pm$ {1:.3f}) log P ${6}${2:.3f} ($\pm$ {3:.3f})'.format(np.abs(best_m), np.std(ms), np.abs(best_b), np.std(bs), '{[3.6]}', msign, bsign)
mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_3p6_m4slope_mcmc_rrab_only_paper.pdf')

m_ab_36 = best_m
e_m_ab_36 = np.std(ms)
b_ab_36 = best_b
e_b_ab_36 = np.std(bs)


### Analysing 4.5, RRab only

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

x = ab_df.log_P
y = ab_df.M_4p5
dy = ab_df.e_M_4p5


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.355, mscale=0.168, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        

sns.set_style("white")
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette(sns.husl_palette(10, l=.4))

colors = sns.color_palette()

xfit = np.arange(-1, 1, 0.1)

ax1 = mp.subplot(111)


mp.errorbar(ab_df.log_P, ab_df.M_4p5, yerr = ab_df.e_M_4p5, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(ab_df.log_P, ab_df.M_4p5, 'o', color=colors[0], ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')

#ax1.plot(xfit, b_ls + m_ls*xfit, 'k--')
mp.xlim(-0.6, -0.1)
mp.ylim(1, -1.75)


xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

## Picking 25 chains at random to plot
Nchain = len(chain)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:25]
for i in I:
    (p, params) = chain[i]
    (b, m, Pbad, Ybad, Vbad) = params
    ys = m*xfit + b
    ax1.plot(xfit, ys, color='k', alpha=0.05)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

mp.plot(xfit, best_m*xfit + best_b, color=colors[0], ls='--')

mp.ylabel('Absolute Magnitude 4.5 $\mu$m')
mp.xlabel('log P (days)')
#mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.suptitle('Whole TGAS sample, RRab only, M4 slope')

if best_b < 0: 
    bsign = '-' 
else:
    bsign = '+'
if best_m < 0:
    msign = '-'
else:
    msign = ''
    

plot_text = 'M$_{4}$ = ${5}${0:.3f} ($\pm$ {1:.3f}) log P ${6}${2:.3f} ($\pm$ {3:.3f})'.format(np.abs(best_m), np.std(ms), np.abs(best_b), np.std(bs), '{[4.5]}', msign, bsign)
mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_4p5_m4slope_mcmc_rrab_only_paper.pdf')

m_ab_45 = best_m
e_m_ab_45 = np.std(ms)
b_ab_45 = best_b
e_b_ab_45 = np.std(bs)



### Analysing 4.5, RRab only

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

x = ab_df.log_P
y = ab_df.M_3p6
dy = ab_df.e_M_3p6

axp1 = mp.subplot(111)

A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain_36 = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.370, mscale=0.139, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain_36.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain_36.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        

sns.set_style("white")
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("husl", 10)

colors = sns.color_palette()

xfit = np.arange(-1, 1, 0.1)

axp1.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp1.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=colors[7], ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='[3.6]')

mp.xlim(-0.6, -0.1)
mp.ylim(0.75, -3.25)

xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()


## Picking 25 chains at random to plot
Nchain = len(chain_36)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:25]
for i in I:
    (p, params) = chain_36[i]
    (b_36, m_36, Pbad, Ybad, Vbad) = params
    ys_36 = m_36*xfit + b_36
    axp1.plot(xfit, ys_36, color=colors[7], alpha=0.05)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain_36[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain_36[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)

m_ab_36 = best_m
e_m_ab_36 = np.std(ms)
b_ab_36 = best_b
e_b_ab_36 = np.std(bs)


axp1.plot(xfit, m_ab_36*xfit + b_ab_36, color=colors[7], ls='--')

mp.ylabel('Absolute Magnitude')
mp.xlabel('log P (days)')
#mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))

if best_b < 0: 
    bsign = '-' 
else:
    bsign = '+'
if best_m < 0:
    msign = '-'
else:
    msign = ''

x = ab_df.log_P
y = ab_df.M_4p5
dy = ab_df.e_M_4p5


A = np.vstack((np.ones(len(x)), x)).T
C = np.diag(dy*dy)

## These are the initial guesses for the MCMC

bestfit= np.dot(np.linalg.inv(C),y)
bestfit= np.dot(A.T,bestfit)
bestfitvar= np.dot(np.linalg.inv(C),A)
bestfitvar= np.dot(A.T,bestfitvar)
bestfitvar= np.linalg.inv(bestfitvar)
bestfit= np.dot(bestfitvar,bestfit)

np.random.seed(10)
Pbad = 0.5 ## setting Pbad = 0.5 for mixture model
Ybad = np.mean(y)
Vbad = np.mean((y-Ybad)**2)

params = np.array([bestfit[0], bestfit[1], Pbad, Ybad, Vbad])

print params

p = posterior(params, x, y, dy)
print 'starting p = ', p

chain_45 = []
oldp = p
oldparams = params
bestparams = oldparams
bestp = oldp
nsteps = 0
naccepts = 0
NSTEPS = 10000

bscale = np.median(dy)

print 'Doing ', NSTEPS, 'steps of MCMC...'
while nsteps < NSTEPS:
    newparams = pick_new_parameters(nsteps, oldparams, fixed_slope=True, slopeval=-2.355, mscale=0.168, bscale=bscale)
    p = posterior(newparams, x, y, dy)
    if p/oldp > np.random.uniform():
        chain_45.append((p, newparams))
        oldparams = newparams
        oldp = p
        if p > bestp:
            bestp = p
            bestparams = newparams
        naccepts += 1
    else:
        chain_45.append((oldp, oldparams))
    nsteps += 1
    if (nsteps % 5000 == 1):
        print nsteps, naccepts, (naccepts/float(nsteps)), oldp, bestp, bestparams
print 'acceptance fraction', (naccepts/float(nsteps))        
sns.set_palette(sns.husl_palette(10, l=.4))

colors = sns.color_palette()

axp1.errorbar(ab_df.log_P, ab_df.M_4p5-1.5, yerr = ab_df.e_M_4p5, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp1.plot(ab_df.log_P, ab_df.M_4p5-1.5, 'o', color=colors[0], ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='[4.5]$-$1.5')


## Picking 25 chains at random to plot
Nchain = len(chain_45)
I = Nchain / 2 + np.random.permutation(Nchain/2)[:25]
for i in I:
    (p, params) = chain_45[i]
    (b_45, m_45, Pbad, Ybad, Vbad) = params
    ys_45 = m_45*xfit + b_45-1.5
    axp1.plot(xfit, ys_45, color=colors[0], alpha=0.05)
### Using a mixture model of 'bad data' and 'good data'
bgp = np.zeros(len(x))
fgp = np.zeros(len(x))

ms = np.array([m for (p, (b, m, Pbad, Ybad, Vbad)) in chain_45[Nchain/2:]])
bs = np.array([b for (p, (b, m, Pbad, Ybad, Vbad)) in chain_45[Nchain/2:]])


best_m = np.median(ms)
best_b = np.median(bs)
m_ab_45 = best_m
e_m_ab_45 = np.std(ms)
b_ab_45 = best_b
e_b_ab_45 = np.std(bs)

axp1.plot(xfit, m_ab_45*xfit + b_ab_45-1.5, color=colors[0], ls='--')

#mp.ylabel('Absolute Magnitude 4.5 $\mu$m')
mp.xlabel('log P (days)')
handles, labels = axp1.get_legend_handles_labels()
mp.legend(handles[::-1],labels[::-1],loc='upper left', numpoints=1)
#mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
#mp.suptitle('Whole TGAS sample, RRab only, M4 slope')

if best_b < 0: 
    bsign = '-' 
else:
    bsign = '+'
if best_m < 0:
    msign = '-'
else:
    msign = ''
    

#plot_text = 'M$_{4}$ = ${5}${0:.3f} ($\pm$ {1:.3f}) log P ${6}${2:.3f} ($\pm$ {3:.3f})'.format(np.abs(best_m), np.std(ms), np.abs(best_b), np.std(bs), '{[4.5]}', msign, bsign)
#mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_m4slope_mcmc_rrab_only_paper.pdf')



### Analysing 4.5, RRab only

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

x = ab_df.log_P
y = ab_df.M_3p6
dy = ab_df.e_M_3p6

axp1 = mp.subplot(111)

def free_fit_36(logp, zp):
    av_p = np.mean(logp)
    return -2.332*(logp - av_p) + zp

def free_fit_45(logp, zp):
    av_p = np.mean(logp)
    return -2.355*(logp - av_p) + zp



m4_ab_slope_36 = -2.370
m4_ab_slope_45 = -2.355


popt, pcov = curve_fit(free_fit_36, x, y, sigma=dy, absolute_sigma=True)

b_36 = popt[0]
b_36_err = pcov[0]

sns.set_style("white")
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("husl", 10)

colors = sns.color_palette()

xfit = np.arange(-1, 1, 0.1)

axp1.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp1.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=colors[7], ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='[3.6]')

mean_36_logp = np.mean(ab_df.log_P)

mp.xlim(-0.6, -0.1)
mp.ylim(0.75, -3.25)

xmin, xmax = mp.xlim()
ymin, ymax = mp.ylim()

axp1.plot(xfit, m4_ab_slope_36*(xfit-mean_36_logp) + b_ab_36, color=colors[7], ls='--')

zp_36 = b_ab_36 + m4_ab_slope_36*(-mean_36_logp)

mp.ylabel('Absolute Magnitude')
mp.xlabel('log P (days)')
#mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
x = ab_df.log_P
y = ab_df.M_4p5
dy = ab_df.e_M_4p5



## These are the initial guesses for the MCMC

popt, pcov = curve_fit(free_fit_45, x, y, sigma=dy, absolute_sigma=True)
b_45 = popt[0]
b_45_err = pcov[0]



sns.set_palette(sns.husl_palette(10, l=.4))

colors = sns.color_palette()

axp1.errorbar(ab_df.log_P, ab_df.M_4p5-1.5, yerr = ab_df.e_M_4p5, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp1.plot(ab_df.log_P, ab_df.M_4p5-1.5, 'o', color=colors[0], ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='[4.5]$-$1.5')

mean_45_logp = np.mean(ab_df.log_P)

axp1.plot(xfit, m4_ab_slope_45*(xfit-mean_45_logp) + b_ab_45-1.5, color=colors[0], ls='--')
zp_45 = b_ab_45 + m4_ab_slope_45*(-mean_45_logp)


#mp.ylabel('Absolute Magnitude 4.5 $\mu$m')
mp.xlabel('log P (days)')
handles, labels = axp1.get_legend_handles_labels()
mp.legend(handles[::-1],labels[::-1],loc='upper left', numpoints=1)
#mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
#mp.suptitle('Whole TGAS sample, RRab only, M4 slope')

#plot_text = 'M$_{4}$ = ${5}${0:.3f} ($\pm$ {1:.3f}) log P ${6}${2:.3f} ($\pm$ {3:.3f})'.format(np.abs(best_m), np.std(ms), np.abs(best_b), np.std(bs), '{[4.5]}', msign, bsign)
#mp.annotate(plot_text, xy=(-0.575, 0.75), xycoords='data', size=12)

mp.savefig('tgas_m4slope_lsq_rrab_only_paper.pdf')

print zp_36, b_36_err, zp_45, b_45_err



b_ab_36

+ m4_ab_slope_36*(-mean_36_logp)

analysis_df

mp.hist(analysis_df.A_3p6, bins=20)


mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.0)

mp.scatter(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_3p6'], c=analysis_df.ix[analysis_df.Type=='ab', '[Fe/H]'], cmap=cm.Spectral_r, marker='o', zorder=4, label='RRab')




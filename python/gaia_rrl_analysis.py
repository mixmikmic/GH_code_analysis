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
#all_files = glob.glob(os.path.join(path, "TgasSource*.csv"))
#gaia_df = pd.concat(pd.read_csv(f) for f in all_files)
#gaia_df = gaia_df.reset_index(drop=True)

gaia_df = pd.read_csv('vizer_crossmatch.tsv', skiprows=166, skipinitialspace=True, names=('input', 'rad', 'HIP', 'TYC2', 'SolID', 'Source', 'RandomI', 'Epoch', 'RA_ICRS', 'e_RA_ICRS', 'DE_ICRS', 'e_DE_ICRS', 'Plx', 'e_Plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'RADEcor', 'RAPlxcor', 'RApmRAcor', 'RApmDEcor', 'DEPlxcor', 'DEpmRAcor', 'DEpmDEcor', 'PlxpmRAcor', 'PlxpmDEcor', 'pmRApmDEcor', 'NAL', 'NAC', 'NgAL', 'NgAC', 'NbAL', 'NbAC', 'DQ', 'epsi', 'sepsi', 'APF', 'ARF', 'WAL', 'WAC', 'Apr', 'MatchObs', 'Dup', 'sK1', 'sK2', 'sK3', 'sK4', 'mK1', 'mK2', 'mK3', 'mK4', 'o_<Gmag>', '<FG>', 'e_<FG>', '<Gmag>', 'Var', 'GLON', 'GLAT', 'ELON', 'ELAT'), na_values='NOT_AVAILABLE', sep=';', comment='#')
gaia_df['ID'] = gaia_df.input.str.split('\t',0).str.get(0)

gaia_df = gaia_df.drop('input', 1)
#gaia_df = gaia_df.drop('rad', 1)
gaia_df = gaia_df.replace('', np.nan)
gaia_df['ID'] = gaia_df['ID'].replace(regex=True, to_replace=r' ',value='_')
gaia_df['id_compare'] = map(str.lower, gaia_df.ID)
gaia_df['id_compare'] = gaia_df['id_compare'].replace(regex=True, to_replace=r'_',value='')

gaia_df

info_df = pd.read_csv('rrl_average_mags', delim_whitespace=True)
info_df

def all_the_gloess(row):
    cepID, df, sm_params = gloess_setup(row['Name'])
    #print sm_params
    name = row.Name
    period = row.Period
    df['phase'] = (df['MJD'] / period) - np.floor(df['MJD'] / period)
    print name, period, sm_params
    bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '3p6', '4p5', '5p8', '8p0']
    offsets= [3, 1.5, 1.2, 0.7, 0.2, 0, -0.4, -0.8, -1.4, -1.8, -2.2, -2.6]
    colors = ['Violet', 'MediumSlateBlue', 'DodgerBlue', 'Turquoise', 'LawnGreen', 'Gold', 'DarkOrange', 'Red', 'MediumVioletRed', 'DeepPink', 'HotPink', 'PeachPuff']
    av_mags = np.zeros(12) + np.nan   
    av_errs = np.zeros(12) + np.nan   
    av_amps = np.zeros(12) + np.nan   
    mp.close()
    mp.clf()
    max_cur = 0
    min_cur = 99
    fig = mp.figure(figsize=(10,10))
    ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])    
    titlestring = name + ', P = ' + str(np.round(period, decimals=4)) + ' days'
    mp.title(titlestring, fontsize=20)
    ax1.set_ylabel('Magnitude')
    ax1.set_xlabel('Phase $\phi$')

    for band in np.arange(len(bands)):
        mag = 'mag_' + str(bands[band])
        err = 'err_' + str(bands[band])
        amplitude = 'amp_' + str(bands[band])
            
        ### Removing data where mag=0 and error=9.99
        #if (df.loc[df[err]==9.99, mag] == 0):
        df.loc[df[mag]==0.00, mag] = np.nan        
        #df.loc[mag==0.00, err==9.99]
        
        ### Attempting dynamic smoothing parameters

        smooth = float(sm_params[band])
        
        pullmags = df[mag]
        n_good = len(pullmags < 50)
        #if (n_good <= 5):
        #    smooth = 1
        #elif (n_good >= 50):
        #    smooth = 0.1
        pullerrs = df[err]
        pullphase = df['phase']
        
        data1, x, y, yerr, xphase = gf.fit_one_band(df[mag], df[err], df['phase'],len(df[mag]) ,smooth)
        
        ave, adev, sdev, var, skew, kurtosis, amp = gf.moment(data1[200:300],100)
        #print ave, (sdev/(np.sqrt(len(data1)))), amp
        
        info_df.ix[info_df.Name==name, mag]=ave
        info_df.ix[info_df.Name==name, err]=sdev/(np.sqrt(len(data1)))
        info_df.ix[info_df.Name==name, amplitude]=amp
        av_mags[band] = ave
        av_errs[band] = sdev/(np.sqrt(len(data1)))
        av_amps[band] = amp
               
        plotmag = pullmags[pullmags<50]
        plotphase = pullphase[pullmags<50]

        plotmag = np.concatenate((plotmag,plotmag,plotmag,plotmag,plotmag))
        plotphase = np.concatenate((plotphase,(plotphase+1.0),(plotphase+2.0),(plotphase+3.0),(plotphase+4.0)))
        size_of_data = len(plotmag)
        
        if np.sign(offsets[band]) == -1: 
            offstring = ' - ' + str(abs(offsets[band]))
        elif np.sign(offsets[band]) == 0: 
            offstring = ''
        else:
            offstring = ' + ' + str(offsets[band])
        lab_text = str(bands[band]) + str(offstring)
        ax1.plot(x,data1+offsets[band],'k-', zorder=4)
        ax1.plot(plotphase, plotmag+offsets[band],color=colors[band],marker='o',ls='None', label=lab_text)
        maxval = np.max(data1[200:300]+offsets[band])
        minval = np.min(data1[200:300]+offsets[band])
        if(maxval > max_cur):
            max_cur = maxval
        if(minval < min_cur):
            min_cur = minval

        
    maxlim = max_cur + 1.5
    minlim = min_cur - 1.5
    ax1.axis([1,3.5,(maxlim),(minlim)])
    handles, labels = ax1.get_legend_handles_labels() 
    mp.legend(handles[::-1],labels[::-1], numpoints=1,prop={'size':10})
    mp.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    mp.savefig(cepID + '.pdf')
    bands_df = pd.DataFrame(bands)
    bands_df.columns = ['Band']
    av_mags_df = pd.DataFrame(av_mags)
    av_mags_df.columns = ['AverageMag']
    av_errs_df = pd.DataFrame(av_errs)
    av_errs_df.columns = ['AverageErr']
    av_amps_df = pd.DataFrame(av_amps)
    av_amps_df.columns = ['AverageAmp']


    cep_out_df = pd.concat([bands_df, av_mags_df, av_errs_df, av_amps_df], axis=1, ignore_index=True)
    print cep_out_df


def gloess_setup(star):
    mag_columns = []
    err_columns = []
    cols = ['MJD']
    bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '3p6', '4p5', '5p8', '8p0']
    for wlen in np.arange(len(bands)):
        mag_name = ('mag_' + str(bands[wlen]))
        err_name = ('err_' + str(bands[wlen]))
        mag_columns.append(mag_name)
        err_columns.append(err_name)
        cols.append(mag_name)
        cols.append(err_name)
    
    cols.append('Reference')
    
    gloess_file = "/Users/vs/Dropbox/CRRP/RR_Lyrae_lightcurves/S19p2_reduction/" + star + '.gloess_in'
    
    linecache.clearcache()
    smooth_line = linecache.getline(gloess_file, 4).strip()
    smooth_line = re.sub("[\[\]\'\",]"," ", smooth_line)
    smooth = smooth_line.split()
    df = pd.read_csv(gloess_file, header=None, skiprows=4, names=(cols), comment='-', delim_whitespace=True)
    return(star, df, smooth)

## only run this if running from scratch

#spitzer_dir = "/Users/vs/Dropbox/CRRP/RR_Lyrae_lightcurves/S19p2_reduction"
#info_file = spitzer_dir + '/rrl_periods'
#info_df = pd.read_csv(info_file, delim_whitespace=True, header=None, names=['Name', 'Period', 'Type'])
#info_df['Name_Lower'] = map(str.lower, info_df.Name)


#av_cols = []
#bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '3p6', '4p5', '5p8', '8p0']
#for wlen in np.arange(len(bands)):
#    mag_name = ('mag_' + str(bands[wlen]))
#    err_name = ('err_' + str(bands[wlen]))
#    amp_name = ('amp_' + str(bands[wlen]))
#    av_cols.append(mag_name)
#    av_cols.append(err_name)
#    av_cols.append(amp_name)
#av_cols

#for columns in np.arange(len(av_cols)):
#    info_df[av_cols[columns]] = np.nan


#info_df.apply(lambda line: all_the_gloess(line), axis=1)


#info_df = info_df.dropna(axis=1, how='all')

#info_df = info_df.rename(columns={'Name_Lower':'id_compare'})
#info_df.to_csv('rrl_average_mags', index=False, header=True, sep=' ', float_format='%4.3f', na_rep= 99.99)

printcols = ['ID', 'rad']
gaia_df[printcols]
gaia_df = gaia_df.groupby(['ID']).min()
#gaia_df = gaia_df.reset_index(drop=True)

merged_df = info_df.merge(gaia_df, on='id_compare')

## Useful to check that the matching worked
#printcols = ['Name', 'id_compare']
#merged_df[printcols]

merged_df.columns

useful = ['Name', 'Period', 'Type', 'mag_3p6', 'err_3p6', 'amp_3p6', 'mag_4p5', 'err_4p5', 'amp_4p5', 'RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx', 'id_compare']
analysis_df = merged_df[useful]

analysis_df

Image("gould_kollmeier_abstract.png")

analysis_df['e_gks'] = np.sqrt((0.79*analysis_df['e_Plx'])**2 - (0.10)**2)

analysis_df

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())


mp.plot(analysis_df['Plx'], analysis_df['e_Plx'], 'o', ls='None', ms=10, zorder=4, color=next(palette), label='TGAS error')
mp.plot(analysis_df['Plx'], analysis_df['e_gks'], 'o', ls='None', ms=10, zorder=4, color=next(palette), label='GKS error')

mp.xlabel('Parallax (mas)')
mp.ylabel('Parallax Uncertainty (mas)')
mp.suptitle('Comparison of original TGAS and Gould-Kollmeier-Sesar uncertainties')

mp.legend(loc='best')

## Correct values of distances and uncertainties
## Deriving distances directly from parallax measurements, no lutz kelker correction, 50% uncertainties

analysis_df = analysis_df.convert_objects(convert_numeric=True)

analysis_df['distance_pc'] = (1./ (analysis_df.Plx * 10**(-3)))
#analysis_df['e_distance_pc'] = analysis_df.distance_pc * analysis_df.e_gks / analysis_df.Plx
analysis_df['mu_dist'] = 5. * np.log10(analysis_df.distance_pc) - 5.0

### Calculating TGAS uncertainties too to show on plots

#analysis_df['e_dist_tgas'] = analysis_df.distance_pc * analysis_df.e_Plx / analysis_df.Plx

## Calculate average errorbars in both cases. 
## This code gives asymmetric error bars too, but not sure how to fit them right now.

## Reduced errors
#analysis_df['mu_p_1sig'] = 5*np.log10(analysis_df.distance_pc + analysis_df.e_distance_pc) - 5.
#analysis_df['mu_m_1sig'] = 5*np.log10(analysis_df.distance_pc - analysis_df.e_distance_pc) - 5.
#analysis_df['mu_av_err'] = (analysis_df.mu_p_1sig - analysis_df.mu_m_1sig)/2.



## TGAS errors
#analysis_df['mu_p_1sig_tgas'] = 5*np.log10(analysis_df.distance_pc + analysis_df.e_dist_tgas) - 5.
#analysis_df['mu_m_1sig_tgas'] = 5*np.log10(analysis_df.distance_pc - analysis_df.e_dist_tgas) - 5.
#analysis_df['mu_av_err_tgas'] = (analysis_df.mu_p_1sig_tgas - analysis_df.mu_m_1sig_tgas)/2.

analysis_df['good_err'] = np.where(((analysis_df['Plx'] - analysis_df['e_Plx'])>0), True, np.nan)

analysis_df = analysis_df.dropna(axis=0, how='all', subset=['good_err'])

## Calculating infrared extinction

analysis_df['A_v'] = np.nan

analysis_df['A_3p6'] = red.ccm_nearir(2.19, 3.1)*red.indebetouw_ir(3.545)*analysis_df.A_v
analysis_df['A_4p5'] = red.ccm_nearir(2.19, 3.1)*red.indebetouw_ir(4.442)*analysis_df.A_v

## Calculate abs mags

analysis_df['M_3p6'] = analysis_df.mag_3p6 - analysis_df.mu_dist #- analysis_df.A36
analysis_df['M_4p5'] = analysis_df.mag_4p5 - analysis_df.mu_dist #- analysis_df.A45

### Correct uncertainties 

analysis_df['e_M_3p6_gks'] = np.abs(np.sqrt((analysis_df.err_3p6/analysis_df.mag_3p6)**2 + (5*analysis_df.e_gks/np.log(10)*analysis_df.Plx)**2)*analysis_df.M_3p6)
analysis_df['e_M_4p5_gks'] = np.abs(np.sqrt((analysis_df.err_4p5/analysis_df.mag_4p5)**2 + (5*analysis_df.e_gks/np.log(10)*analysis_df.Plx)**2)*analysis_df.M_4p5)

analysis_df['e_M_3p6_tgas'] = np.abs(np.sqrt((analysis_df.err_3p6/analysis_df.mag_3p6)**2 + (5*analysis_df.e_Plx/np.log(10)*analysis_df.Plx)**2)*analysis_df.M_3p6)
analysis_df['e_M_4p5_tgas'] = np.abs(np.sqrt((analysis_df.err_4p5/analysis_df.mag_4p5)**2 + (5*analysis_df.e_Plx/np.log(10)*analysis_df.Plx)**2)*analysis_df.M_4p5)

analysis_df

analysis_df['log_P'] = np.log10(analysis_df['Period'])
analysis_df['logP_f'] = np.where(analysis_df['Type']=='c', analysis_df['log_P'] + 0.127, analysis_df['log_P'])

Image("neeley_m4_pls.png")

def M4_ab_36(logp, zp):
    return -2.370*(logp + 0.26) + zp

def M4_ab_45(logp, zp):
    return -2.355*(logp + 0.26) + zp

def M4_c_36(logp, zp):
    return -2.658*(logp + 0.55) + zp

def M4_c_45(logp, zp):
    return -2.979*(logp + 0.55) + zp

def M4_fund_36(logp, zp):
    return -2.332*(logp + 0.30) + zp

def M4_fund_45(logp, zp):
    return -2.336*(logp + 0.30) + zp

def free_fit(logp, slope, zp):
    av_p = np.mean(logp)
    return slope*(logp - av_p) + zp

m4_fund_slope_36 = -2.332
m4_fund_slope_45 = -2.336

m4_fund_avp = 0.30

## Fit and plot the fundamentalised PL relations:

columns = ['slope', 'e_slope', 'zeropoint', 'e_zeropoint', 'mean_logp', 'source', 'type', 'band', 'sample', 'n_stars']

fit_df = pd.DataFrame(columns=columns)

#analysis_df['abs_m_err'] = abs(analysis_df.mu_m_1sig - analysis_df.mu_dist)
#analysis_df['abs_p_err'] = abs(analysis_df.mu_p_1sig - analysis_df.mu_dist)
#analysis_df['abs_av_err'] = (analysis_df.abs_m_err + analysis_df.abs_p_err) / 2.0

#analysis_df['tgas_m_err'] = abs(analysis_df.mu_m_1sig_tgas - analysis_df.mu_dist)
#analysis_df['tgas_p_err'] = abs(analysis_df.mu_p_1sig_tgas - analysis_df.mu_dist)
#analysis_df['tgas_av_err'] = (analysis_df.tgas_m_err + analysis_df.tgas_p_err) / 2.0

### Use error columns e_M_3p6_gks etc

p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_3p6)

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(analysis_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
mp.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='TGAS errors')
mp.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='GKS errors')
mp.plot(analysis_df.logP_f, analysis_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[3.6]')

popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_4p5)

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(analysis_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp)+1.0, ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp)+1.0, alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp+1.0, ls='--', label="[4.5]+1.0", color=col)
mp.errorbar(analysis_df.logP_f, analysis_df.M_4p5+1.0, yerr = analysis_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
mp.errorbar(analysis_df.logP_f, analysis_df.M_4p5+1.0, yerr = analysis_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
mp.plot(analysis_df.logP_f, analysis_df.M_4p5+1.0, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[4.5]+1.0')



mp.xlim(-0.6, -0.1)
mp.ylim(2.0, -2.5)
mp.xlabel('log P (days)')
mp.ylabel('Absolute Magnitude')
mp.suptitle('Calibrator RRL PL - RRab + RRc fundamentalised')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))


fit_df

p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_3p6'])

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(ab_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])


col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
mp.errorbar(ab_df['log_P'], ab_df['M_3p6'], yerr = ab_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='TGAS errors')
mp.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='GKS errors')
mp.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[3.6]')

popt, pcov = curve_fit(free_fit, ab_df.log_P, ab_df.M_4p5)

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(ab_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp)+1.0, ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp)+1.0, alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp+1.0, ls='--', label="[4.5]+1.0", color=col)
mp.errorbar(ab_df.log_P, ab_df.M_4p5+1.0, yerr = ab_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
mp.errorbar(ab_df.log_P, ab_df.M_4p5+1.0, yerr = ab_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
mp.plot(ab_df.log_P, ab_df.M_4p5+1.0, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[4.5]+1.0')



mp.xlim(-0.6, -0.1)
mp.ylim(2.0, -2.0)
mp.xlabel('log P (days)')
mp.ylabel('Absolute Magnitude')
mp.suptitle('Calibrator RRL PL - RRab only')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))


p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_3p6'])

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(c_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
mp.errorbar(c_df.log_P, c_df.M_3p6, yerr=c_df.e_M_3p6_tgas, ls='None', zorder=2, color='Grey', label='TGAS errors')
mp.errorbar(c_df.log_P, c_df.M_3p6, yerr=c_df.e_M_3p6_gks, ls='None', zorder=4, color=col, label='GKS errors')
mp.plot(c_df.log_P, c_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[3.6]')

popt, pcov = curve_fit(free_fit, c_df.log_P, c_df.M_4p5)

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(c_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])

col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp)+1.0, ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp)+1.0, alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp+1.0, ls='--', label="[4.5]+1.0", color=col)
mp.errorbar(c_df.log_P, c_df.M_4p5+1.0, yerr = c_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
mp.errorbar(c_df.log_P, c_df.M_4p5+1.0, yerr = c_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
mp.plot(c_df.log_P, c_df.M_4p5+1.0, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[4.5]+1.0')



mp.xlim(-0.6, -0.1)
mp.ylim(2.0, -3.0)
mp.xlabel('log P (days)')
mp.ylabel('Absolute Magnitude')
mp.suptitle('Calibrator RRL PL - RRc only')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))


sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

col = next(palette)
mp.hist(analysis_df[~np.isnan(analysis_df['e_Plx'])].e_Plx, color=col, label='TGAS', alpha=0.6)
col = next(palette)
mp.hist(analysis_df[~np.isnan(analysis_df['e_gks'])].e_gks, color=col, label='GKS', alpha=0.6)

mp.xlabel('Parallax Uncertainty (mas)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.xlim(0.1, 0.5)
mp.ylim(0, 15)
mp.suptitle('Entire Sample')


sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

col = next(palette)
mp.hist(ab_df[~np.isnan(ab_df['e_Plx'])].e_Plx, color=col, label='TGAS', alpha=0.6)
col = next(palette)
mp.hist(ab_df[~np.isnan(ab_df['e_gks'])].e_gks, color=col, label='GKS', alpha=0.6)

mp.xlabel('Parallax Uncertainty (mas)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.xlim(0.1, 0.5)
mp.ylim(0, 12)
mp.suptitle('RRab Only')


sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

col = next(palette)
mp.hist(c_df[~np.isnan(c_df['e_Plx'])].e_Plx, color=col, label='TGAS', alpha=0.6, bins=5)
col = next(palette)
mp.hist(c_df[~np.isnan(c_df['e_gks'])].e_gks, color=col, label='GKS', alpha=0.6, bins=5)

mp.xlabel('Parallax Uncertainty (mas)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.xlim(0.1, 0.5)
mp.ylim(0, 5)
mp.suptitle('RRc Only')


p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

ecut1_df = analysis_df.where(analysis_df.e_gks<0.25).dropna(axis=0, how='all')
ecut1_df = ecut1_df.reset_index(drop=True)
popt, pcov = curve_fit(free_fit, ecut1_df.logP_f, ecut1_df.M_3p6)

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ecut1_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '3p6', 'sample' : 'ecut1', 'n_stars': len(ecut1_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ecut1_df.logP_f)

col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
mp.errorbar(ecut1_df.logP_f, ecut1_df.M_3p6, yerr = ecut1_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='TGAS errors')
mp.errorbar(ecut1_df.logP_f, ecut1_df.M_3p6, yerr = ecut1_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='GKS errors')
mp.plot(ecut1_df.logP_f, ecut1_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[3.6]')

popt, pcov = curve_fit(free_fit, ecut1_df.logP_f, ecut1_df.M_4p5)

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ecut1_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '4p5', 'sample' : 'ecut1', 'n_stars': len(ecut1_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ecut1_df.logP_f)

col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp)+1.0, ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp)+1.0, alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp+1.0, ls='--', label="[4.5]+1.0", color=col)
mp.errorbar(ecut1_df.logP_f, ecut1_df.M_4p5+1.0, yerr = ecut1_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
mp.errorbar(ecut1_df.logP_f, ecut1_df.M_4p5+10, yerr = ecut1_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
mp.plot(ecut1_df.logP_f, ecut1_df.M_4p5+1.0, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[4.5]+1.0')



mp.xlim(-0.6, -0.1)
mp.ylim(2.0, -4.0)
mp.xlabel('log P (days)')
mp.ylabel('Absolute Magnitude')
mp.suptitle('Calibrator RRL PL - RRab + RRc fundamentalised, GKS < 0.25 mas')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))


p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

ab_df = ecut1_df.where(ecut1_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_3p6'])

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '3p6', 'sample' : 'ecut1', 'n_stars': len(ab_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])


col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
mp.errorbar(ab_df['log_P'], ab_df['M_3p6'], yerr = ab_df.e_M_3p6_tgas, ls='None',zorder=2, color='Grey', label='TGAS errors')
mp.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='GKS errors')
mp.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[3.6]')

popt, pcov = curve_fit(free_fit, ab_df.log_P, ab_df.M_4p5)

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '4p5', 'sample' : 'ecut1', 'n_stars': len(ab_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp)+1.0, ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp)+1.0, alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp+1.0, ls='--', label="[4.5]+1.0", color=col)
mp.errorbar(ab_df.log_P, ab_df.M_4p5+1.0, yerr = ab_df.e_M_4p5_tgas, ls='None',zorder=2, color='Grey', label='_nolegend_')
mp.errorbar(ab_df.log_P, ab_df.M_4p5+1.0, yerr = ab_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
mp.plot(ab_df.log_P, ab_df.M_4p5+1.0, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[4.5]+1.0')



mp.xlim(-0.6, -0.1)
mp.ylim(2.0, -3.0)
mp.xlabel('log P (days)')
mp.ylabel('Absolute Magnitude')
mp.suptitle('Calibrator RRL PL - RRab only, GKS < 0.25 mas')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))


p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

c_df = ecut1_df.where(ecut1_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_3p6'])

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '3p6', 'sample' : 'ecut1', 'n_stars': len(c_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
mp.errorbar(c_df['log_P'], c_df['M_3p6'], yerr = c_df.e_M_3p6_tgas, ls='None',zorder=2, color='Grey', label='TGAS errors')
mp.errorbar(c_df.log_P, c_df.M_3p6, yerr = c_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='GKS errors')
mp.plot(c_df.log_P, c_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[3.6]')

popt, pcov = curve_fit(free_fit, c_df.log_P, c_df.M_4p5)

fit_df = fit_df.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '4p5', 'sample' : 'ecut1', 'n_stars': len(c_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])

col = next(palette)

mp.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp)+1.0, ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp)+1.0, alpha = 0.3, color=col)

mp.plot(p1, slope*(p1-mean)+ zp+1.0, ls='--', label="[4.5]+1.0", color=col)
mp.errorbar(c_df.log_P, c_df.M_4p5+1.0, yerr = c_df.e_M_4p5_tgas, ls='None',zorder=2, color='Grey', label='_nolegend_')
mp.errorbar(c_df.log_P, c_df.M_4p5+1.0, yerr = c_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
mp.plot(c_df.log_P, c_df.M_4p5+1.0, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='M[4.5]+1.0')



mp.xlim(-0.6, -0.1)
mp.ylim(2.0, -8.0)
mp.xlabel('log P (days)')
mp.ylabel('Absolute Magnitude')
mp.suptitle('Calibrator RRL PL - RRc only')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))


fit_df

### Fit 3.6 in all variants

columns = ['slope', 'e_slope', 'zeropoint', 'e_zeropoint', 'mean_logp', 'source', 'type', 'band', 'sample', 'n_stars']

fit_df_final = pd.DataFrame(columns=columns)

p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

## Whole sample, no cuts, RRc fundamentalised

popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_3p6)

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(analysis_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)

axp1 = mp.subplot(311)
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -6.0)

axp1.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp1.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp1.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='TGAS errors')
axp1.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='GKS errors')
axp1.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_3p6'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
axp1.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_3p6'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

xticklabels = axp1.get_xticklabels()
mp.setp(xticklabels, visible=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, -0.5))
axp1.set_yticks([0, -2, -4, -6])



## whole sample, no cuts, RRab only

axp2 = mp.subplot(312, sharex=axp1, sharey=axp1)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_3p6'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(ab_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_3p6'], yerr = ab_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp2.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
mp.ylabel('Absolute Magnitude 3.6 $\mu$m')

xticklabels = axp2.get_xticklabels()
mp.setp(xticklabels, visible=False)
axp2.set_yticks([0, -2, -4, -6])

## whole sample, no cuts, RRc only

axp3 = mp.subplot(313, sharex=axp1, sharey=axp1)

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_3p6'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(c_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


axp3.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp3.plot(p1, slope*(p1-mean)+ zp, ls='--', label="_nolegend_", color=col)
axp3.errorbar(c_df['log_P'], c_df['M_3p6'], yerr = c_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp3.errorbar(c_df.log_P, c_df.M_3p6, yerr = c_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp3.plot(c_df.log_P, c_df.M_3p6, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp3.set_yticks([0, -2, -4, -6])


mp.xlabel('log P (days)')
mp.suptitle('Entire Sample of TGAS-CRRP RRL')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.savefig('crrp_tgas_3p6um.pdf')

### Fit 4.5 in all variants

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

## Whole sample, no cuts, RRc fundamentalised

popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_4p5)

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(analysis_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)
col = next(palette)


axp1 = mp.subplot(311)
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -6.0)

axp1.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp1.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp1.errorbar(analysis_df.logP_f, analysis_df.M_4p5, yerr = analysis_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='TGAS errors')
axp1.errorbar(analysis_df.logP_f, analysis_df.M_4p5, yerr = analysis_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='GKS errors')
axp1.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_4p5'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
axp1.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_4p5'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

xticklabels = axp1.get_xticklabels()
mp.setp(xticklabels, visible=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, -0.5))
axp1.set_yticks([0, -2, -4, -6])



## whole sample, no cuts, RRab only

axp2 = mp.subplot(312, sharex=axp1, sharey=axp1)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_4p5'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(ab_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_4p5'], yerr = ab_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp2.errorbar(ab_df.log_P, ab_df.M_4p5, yerr = ab_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_4p5, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
mp.ylabel('Absolute Magnitude 4.5 $\mu$m')

xticklabels = axp2.get_xticklabels()
mp.setp(xticklabels, visible=False)
axp2.set_yticks([0, -2, -4, -6])

## whole sample, no cuts, RRc only

axp3 = mp.subplot(313, sharex=axp1, sharey=axp1)

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_4p5'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(c_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


axp3.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp3.plot(p1, slope*(p1-mean)+ zp, ls='--', label="_nolegend_", color=col)
axp3.errorbar(c_df['log_P'], c_df['M_4p5'], yerr = c_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp3.errorbar(c_df.log_P, c_df.M_4p5, yerr = c_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp3.plot(c_df.log_P, c_df.M_4p5, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp3.set_yticks([0, -2, -4, -6])


mp.xlabel('log P (days)')
mp.suptitle('Entire Sample of TGAS-CRRP RRL')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)


mp.savefig('crrp_tgas_4p5um.pdf')

fit_df_final

fit_df_final.to_csv('crrp_tgas_noclipping.csv', index=False, header=True, sep=' ', float_format='%4.5f', na_rep= 99.99)


palette = itertools.cycle(sns.color_palette())

mp.plot(analysis_df['log_P'], analysis_df['e_Plx'], 'o', color=next(palette), ls='None', label='TGAS')
mp.plot(analysis_df['log_P'], analysis_df['e_gks'], 'o', color=next(palette), ls='None', label='GKS')
mp.xlabel('log P (days)')
mp.ylabel('Parallax uncertainty (mas)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.title('No correlation between log P and parallax uncertainty')

palette = itertools.cycle(sns.color_palette())

mp.plot(analysis_df['log_P'], analysis_df.e_M_3p6_gks, 'o', color=next(palette), ls='None', label='[3.6]')
mp.plot(analysis_df['log_P'], analysis_df.e_M_4p5_gks, 'o', color=next(palette), ls='None', label='[4.5]')
mp.xlabel('log P (days)')
mp.ylabel('Absolute Magnitude uncertainty (mag)')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
mp.title('Correlation between log P and magnitude uncertainty?')

### Fit 3.6 in all variants

columns = ['slope', 'e_slope', 'zeropoint', 'e_zeropoint', 'mean_logp', 'source', 'type', 'band', 'sample', 'n_stars']

fit_df_weighted = pd.DataFrame(columns=columns)

p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

## Whole sample, weighted fit, RRc fundamentalised


popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_3p6, sigma=analysis_df.e_M_3p6_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '3p6', 'sample' : 'weighted', 'n_stars': len(analysis_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)

axp1 = mp.subplot(311)
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -6.0)

axp1.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp1.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp1.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_gks, ls='None',zorder=4, color='Grey', label='GKS errors')
axp1.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_3p6'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
axp1.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_3p6'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

xticklabels = axp1.get_xticklabels()
mp.setp(xticklabels, visible=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, -0.5))
axp1.set_yticks([0, -2, -4, -6])



## whole sample, no cuts, RRab only

axp2 = mp.subplot(312, sharex=axp1, sharey=axp1)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_3p6'], sigma=ab_df.e_M_3p6_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '3p6', 'sample' : 'weighted', 'n_stars': len(ab_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_3p6'], yerr = ab_df.e_M_3p6_gks, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
mp.ylabel('Absolute Magnitude 3.6 $\mu$m')

xticklabels = axp2.get_xticklabels()
mp.setp(xticklabels, visible=False)
axp2.set_yticks([0, -2, -4, -6])

## whole sample, no cuts, RRc only

axp3 = mp.subplot(313, sharex=axp1, sharey=axp1)

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_3p6'], sigma=c_df.e_M_3p6_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '3p6', 'sample' : 'weighted', 'n_stars': len(c_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


axp3.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp3.plot(p1, slope*(p1-mean)+ zp, ls='--', label="_nolegend_", color=col)
axp3.errorbar(c_df['log_P'], c_df['M_3p6'], yerr = c_df.e_M_3p6_gks, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp3.plot(c_df.log_P, c_df.M_3p6, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp3.set_yticks([0, -2, -4, -6])


mp.xlabel('log P (days)')
mp.suptitle('Entire Sample of TGAS-CRRP RRL, Weighted Fitting')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.0)

mp.savefig('crrp_tgas_3p6um_weighted.pdf')

### Fit 4.5 in all variants


sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

## Whole sample, weighted fit, RRc fundamentalised


popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_4p5, sigma=analysis_df.e_M_4p5_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '4p5', 'sample' : 'weighted', 'n_stars': len(analysis_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)
col = next(palette)

axp1 = mp.subplot(311)
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -6.0)

axp1.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp1.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[4.5]", color=col)
axp1.errorbar(analysis_df.logP_f, analysis_df.M_4p5, yerr = analysis_df.e_M_4p5_gks, ls='None',zorder=4, color='Grey', label='GKS errors')
axp1.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_4p5'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
axp1.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_4p5'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

xticklabels = axp1.get_xticklabels()
mp.setp(xticklabels, visible=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, -0.5))
axp1.set_yticks([0, -2, -4, -6])



## whole sample, no cuts, RRab only

axp2 = mp.subplot(312, sharex=axp1, sharey=axp1)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_4p5'], sigma=ab_df.e_M_4p5_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '4p5', 'sample' : 'weighted', 'n_stars': len(ab_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[4.5]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_4p5'], yerr = ab_df.e_M_4p5_gks, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_4p5, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
mp.ylabel('Absolute Magnitude 4.5 $\mu$m')

xticklabels = axp2.get_xticklabels()
mp.setp(xticklabels, visible=False)
axp2.set_yticks([0, -2, -4, -6])

## whole sample, no cuts, RRc only

axp3 = mp.subplot(313, sharex=axp1, sharey=axp1)

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_4p5'], sigma=c_df.e_M_4p5_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '4p5', 'sample' : 'weighted', 'n_stars': len(c_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


axp3.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp3.plot(p1, slope*(p1-mean)+ zp, ls='--', label="_nolegend_", color=col)
axp3.errorbar(c_df['log_P'], c_df['M_4p5'], yerr = c_df.e_M_4p5_gks, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp3.plot(c_df.log_P, c_df.M_4p5, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp3.set_yticks([0, -2, -4, -6])


mp.xlabel('log P (days)')
mp.suptitle('Entire Sample of TGAS-CRRP RRL, Weighted Fitting')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.0)

mp.savefig('crrp_tgas_4p5um_weighted.pdf')

fit_df_weighted

min(analysis_df.e_M_3p6_gks), max(analysis_df.e_M_3p6_gks), np.mean(analysis_df.e_M_3p6_gks)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

fig = mp.figure(figsize=(10,5))
ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
mp.xlabel('$\sigma_{M}$ (mag)')
axp1 = fig.add_subplot(121)
axp2 = fig.add_subplot(122)

axp1.hist(analysis_df.e_M_3p6_gks, color=next(palette), label='[3.6]', alpha=0.6)
axp2.hist(analysis_df.e_M_4p5_gks, color=next(palette), label='[4.5]', alpha=0.6)

axp1.annotate('[3.6]', xy=(0.7, 12), xycoords='data')
axp2.annotate('[4.5]', xy=(0.7, 12), xycoords='data')

mp.suptitle('Absolute Magnitude uncertainties using $\sigma_{\pi, GKS}$')
#mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))



sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

fig = mp.figure(figsize=(10,5))
ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
mp.ylabel('$\sigma_{M}$ (mag)')
mp.xlabel('Absolute Magnitude (mag)')
axp1 = fig.add_subplot(121)
axp2 = fig.add_subplot(122)

axp1.plot(analysis_df.M_3p6, analysis_df.e_M_3p6_gks, 'o', color=next(palette), label='[3.6]')

axp2.plot(analysis_df.M_4p5, analysis_df.e_M_3p6_gks, 'o', color=next(palette), label='[4.5]')

axp1.annotate('[3.6]', xy=(0.7, 12), xycoords='data')
axp2.annotate('[4.5]', xy=(0.7, 12), xycoords='data')
yticklabels = axp2.get_yticklabels()
mp.setp(yticklabels, visible=False)

mp.suptitle('Absolute Magnitude uncertainties using $\sigma_{\pi, GKS}$')
#mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))



feast_df = pd.read_csv('Feast_2008.tsv', sep=';', skiprows=64, names=('HIP', 'Name', 'plx', 'e_plx', 'Vmag', 'Jmag', 'Hmag', 'Ksmag', 'Per', '[Fe/H]', 'E(B-V)', 'Type', 'Simbad', '_RA', '_DE'))

feast_df['id_compare'] = map(str.lower, feast_df.Name)
feast_df['id_compare'] = feast_df['id_compare'].replace(regex=True, to_replace=r' ',value='')

feast_metals = ['id_compare', '[Fe/H]']



analysis_df = analysis_df.reset_index(drop=True)

analysis_df = analysis_df.merge(feast_df[feast_metals], on='id_compare')



def brani_pl_w1(logp, feh):
    return -1.54*(logp + 0.52854) + 0.25*(feh + 1.4) +zp

### Fit 3.6 in all variants

columns = ['slope', 'e_slope', 'zeropoint', 'e_zeropoint', 'mean_logp', 'source', 'type', 'band', 'sample', 'n_stars']

fit_df_weighted = pd.DataFrame(columns=columns)

p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

## Whole sample, weighted fit, RRc fundamentalised


popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_3p6, sigma=analysis_df.e_M_3p6_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '3p6', 'sample' : 'weighted', 'n_stars': len(analysis_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

mean_feh = np.mean(analysis_df['[Fe/H]'])

b_popt, b_pcov = curve_fit(brani_pl_w1, (analysis_df.logP_f, analysis_df['[Fe/H]']), analysis_df.M_3p6)

col = next(palette)

axp1 = mp.subplot(311)
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -6.0)

#axp1.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp1.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp1.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_gks, ls='None',zorder=4, color='Grey', label='GKS errors')
axp1.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_3p6'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
axp1.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_3p6'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')
axp1.plot(p1, brani_pl_w1(p1,mean_feh), ls='--', label="[3.6]", color='red')
xticklabels = axp1.get_xticklabels()
mp.setp(xticklabels, visible=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, -0.5))
axp1.set_yticks([0, -2, -4, -6])



## whole sample, no cuts, RRab only

axp2 = mp.subplot(312, sharex=axp1, sharey=axp1)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_3p6'], sigma=ab_df.e_M_3p6_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '3p6', 'sample' : 'weighted', 'n_stars': len(ab_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])
mean_feh = np.mean(ab_df['[Fe/H]'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_3p6'], yerr = ab_df.e_M_3p6_gks, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp2.plot(p1, brani_pl_w1(p1,mean_feh), ls='--', label="[3.6]", color='red')

mp.ylabel('Absolute Magnitude 3.6 $\mu$m')

xticklabels = axp2.get_xticklabels()
mp.setp(xticklabels, visible=False)
axp2.set_yticks([0, -2, -4, -6])

## whole sample, no cuts, RRc only

axp3 = mp.subplot(313, sharex=axp1, sharey=axp1)

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_3p6'], sigma=c_df.e_M_3p6_gks, absolute_sigma=True)

fit_df_weighted = fit_df_weighted.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '3p6', 'sample' : 'weighted', 'n_stars': len(c_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


axp3.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp3.plot(p1, slope*(p1-mean)+ zp, ls='--', label="_nolegend_", color=col)
axp3.errorbar(c_df['log_P'], c_df['M_3p6'], yerr = c_df.e_M_3p6_gks, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp3.plot(c_df.log_P, c_df.M_3p6, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp3.set_yticks([0, -2, -4, -6])


mp.xlabel('log P (days)')
mp.suptitle('Entire Sample of TGAS-CRRP RRL, Weighted Fitting')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.0)

mp.savefig('crrp_tgas_3p6um_weighted.pdf')





# Define the probability function as likelihood * prior.
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

slope_guess = -1.0
zp_guess = -0.6
e_guess = 1.0

np.random.seed(123)

x = ab_df.log_P
y = ab_df.M_3p6
yerr = ab_df.e_M_3p6_gks

chi2 = lambda *args: -2 * lnlike(*args)
result = op.minimize(chi2, [slope_guess, zp_guess, np.log(e_guess)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result["x"]
print("""Maximum likelihood result:
    m = {0} (truth: {1})
    b = {2} (truth: {3})
    f = {4} (truth: {5})
""".format(m_ml, slope_guess, b_ml, zp_guess, np.exp(lnf_ml), e_guess))

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

x1 = np.arange(-1,0.1,0.1)

palette = itertools.cycle(sns.color_palette())
col=next(palette)

mp.errorbar(x, y, yerr=yerr, ls='None',zorder=4, color='Grey', label='GKS errors')
mp.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_3p6'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
#mp.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_3p6'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

mp.plot(x1, m_ml*x1+b_ml, ls='--', color=col)

mp.ylabel('Absolute Magnitude 3.6 $\mu$m')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.0)
mp.xlabel('log P (days)')

analysis_df.to_csv('analysis_rrl.csv', index=False, header=True, sep=' ', na_rep= 99.99)

analysis_df

analysis_df.ix[0]

ra = analysis_df.ix[0].RA_ICRS
dec = analysis_df.ix[0].DE_ICRS
coord_string = str(ra) + 'd ' +  str(dec) + 'd'
C = coord.SkyCoord(coord_string, frame='fk5')

table = IrsaDust.get_extinction_table(C)

table

def grab_extinction(row):
    star = row.Name
    #print star
    ra = row.RA_ICRS
    #print ra
    dec = row.DE_ICRS
    #print dec
    coord_string = str(ra) + 'd ' +  str(dec) + 'd'
    #print coord_string
    C = coord.SkyCoord(coord_string, frame='fk5')
    #print C
    table = IrsaDust.get_extinction_table(C)
    #print table
    irac_1_sandf = table[19][3]
    irac_2_sandf = table[20][3]
    #print irac_1_sandf, irac_2_sandf
    analysis_df.ix[analysis_df.Name==star, 'A_3p6'] = irac_1_sandf
    analysis_df.ix[analysis_df.Name==star, 'A_4p5'] = irac_2_sandf

    

table[19], table[20]

analysis_df.apply(lambda line: grab_extinction(line), axis=1)

analysis_df

min(analysis_df.A_3p6), max(analysis_df.A_3p6)

analysis_df['M_3p6'] = analysis_df.mag_3p6 - analysis_df.mu_dist - analysis_df.A_3p6
analysis_df['M_4p5'] = analysis_df.mag_4p5 - analysis_df.mu_dist - analysis_df.A_4p5

### Fit 3.6 in all variants

columns = ['slope', 'e_slope', 'zeropoint', 'e_zeropoint', 'mean_logp', 'source', 'type', 'band', 'sample', 'n_stars']

fit_df_final = pd.DataFrame(columns=columns)

p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

## Whole sample, no cuts, RRc fundamentalised

popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_3p6)

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(analysis_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)

axp1 = mp.subplot(311)
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -6.0)

axp1.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp1.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp1.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='TGAS errors')
axp1.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='GKS errors')
axp1.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_3p6'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
axp1.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_3p6'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

xticklabels = axp1.get_xticklabels()
mp.setp(xticklabels, visible=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, -0.5))
axp1.set_yticks([0, -2, -4, -6])



## whole sample, no cuts, RRab only

axp2 = mp.subplot(312, sharex=axp1, sharey=axp1)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_3p6'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(ab_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_3p6'], yerr = ab_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp2.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
mp.ylabel('Absolute Magnitude 3.6 $\mu$m')

xticklabels = axp2.get_xticklabels()
mp.setp(xticklabels, visible=False)
axp2.set_yticks([0, -2, -4, -6])

## whole sample, no cuts, RRc only

axp3 = mp.subplot(313, sharex=axp1, sharey=axp1)

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_3p6'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(c_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


axp3.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp3.plot(p1, slope*(p1-mean)+ zp, ls='--', label="_nolegend_", color=col)
axp3.errorbar(c_df['log_P'], c_df['M_3p6'], yerr = c_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp3.errorbar(c_df.log_P, c_df.M_3p6, yerr = c_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp3.plot(c_df.log_P, c_df.M_3p6, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp3.set_yticks([0, -2, -4, -6])


mp.xlabel('log P (days)')
mp.suptitle('Entire Sample of TGAS-CRRP RRL, extinction corrected')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.savefig('crrp_tgas_3p6um_ext_corr.pdf')

### Fit 3.6 in all variants

columns = ['slope', 'e_slope', 'zeropoint', 'e_zeropoint', 'mean_logp', 'source', 'type', 'band', 'sample', 'n_stars']

fit_df_final = pd.DataFrame(columns=columns)

p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette(sns.husl_palette(10, l=.4))
colors = sns.color_palette()

col=colors[0]

mp.xlim(-0.6, -0.1)
#mp.ylim(1.0, -6.0)

## whole sample, no cuts, RRab only

axp2 = mp.subplot(111)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_3p6'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(ab_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_3p6'], yerr = ab_df.e_M_3p6_tgas, ls='None',zorder=4, color='black', label='TGAS errors')
axp2.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='GKS errors')
mp.ylabel('Absolute Magnitude 3.6 $\mu$m')

xticklabels = axp2.get_xticklabels()
axp2.set_yticks([1, 0, -1, -2, -3, -4])

titletext = "TGAS Spitzer RRab, extinction corrected, d = 1 \/ $\varpi$"

mp.xlabel('log P (days)')
#mp.suptitle("TGAS Spitzer RRab, extinction corrected, d = 1 \/$\varpi$")
mp.suptitle(r"TGAS Spitzer RRab, extinction corrected, d = 1 / $\varpi$")
mp.xlim(-0.6, -0.1)
#mp.gca().invert_yaxis()
mp.ylim(1, -3)
mp.legend(loc='center right', bbox_to_anchor=(0.35, 0.85))

mp.savefig('crrp_tgas_3p6um_gs.png')

### Fit 4.5 in all variants

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())

## Whole sample, no cuts, RRc fundamentalised

popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_4p5)

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(analysis_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)
col = next(palette)


axp1 = mp.subplot(311)
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -6.0)

axp1.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp1.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp1.errorbar(analysis_df.logP_f, analysis_df.M_4p5, yerr = analysis_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='TGAS errors')
axp1.errorbar(analysis_df.logP_f, analysis_df.M_4p5, yerr = analysis_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='GKS errors')
axp1.plot(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_4p5'], 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRab')
axp1.plot(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_4p5'], '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='RRc')

xticklabels = axp1.get_xticklabels()
mp.setp(xticklabels, visible=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, -0.5))
axp1.set_yticks([0, -2, -4, -6])



## whole sample, no cuts, RRab only

axp2 = mp.subplot(312, sharex=axp1, sharey=axp1)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_4p5'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(ab_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_4p5'], yerr = ab_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp2.errorbar(ab_df.log_P, ab_df.M_4p5, yerr = ab_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_4p5, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
mp.ylabel('Absolute Magnitude 4.5 $\mu$m')

xticklabels = axp2.get_xticklabels()
mp.setp(xticklabels, visible=False)
axp2.set_yticks([0, -2, -4, -6])

## whole sample, no cuts, RRc only

axp3 = mp.subplot(313, sharex=axp1, sharey=axp1)

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_4p5'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '4p5', 'sample' : 'complete', 'n_stars': len(c_df['M_4p5']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


axp3.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp3.plot(p1, slope*(p1-mean)+ zp, ls='--', label="_nolegend_", color=col)
axp3.errorbar(c_df['log_P'], c_df['M_4p5'], yerr = c_df.e_M_4p5_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp3.errorbar(c_df.log_P, c_df.M_4p5, yerr = c_df.e_M_4p5_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp3.plot(c_df.log_P, c_df.M_4p5, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp3.set_yticks([0, -2, -4, -6])


mp.xlabel('log P (days)')
mp.suptitle('Entire Sample of TGAS-CRRP RRL, Extinction corrected')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)


mp.savefig('crrp_tgas_4p5um_ext_corr.pdf')

def log_prior(theta):
    if theta[2] <= 0 or np.any(np.abs(theta[:2]) > 1000):   ##### first part of this is the case where there is no scatter
        return -np.inf  # log(0)
    else:
        # Jeffreys Prior
        return -np.log(theta[2])
    
def log_likelihood(theta, x, y, dy):
    y_model = theta[0] + theta[1] * x
    S = dy ** 2 + theta[2] ** 2
    return -0.5 * np.sum(np.log(2 * np.pi * S) +
                         (y - y_model) ** 2 / S)

def log_posterior(theta, x, y, dy):
    return log_prior(theta) + log_likelihood(theta, x, y, dy)

x = ab_df.log_P
y = ab_df.M_3p6
dy = ab_df.e_M_3p6_gks

ndim = 3  # number of parameters in the model -- intrinsic scatter is now an extra parameter
nwalkers = 50  # number of MCMC walkers

# initialize walkers
starting_guesses = np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=[x, y, dy])
pos, prob, state = sampler.run_mcmc(starting_guesses, 200)

fig, ax = mp.subplots(3, sharex=True)
for i in range(3):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);

# Are your chains stabilized? Reset them and get a clean sample
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

# Use corner.py to visualize the three-dimensional posterior
fig, ax = mp.subplots(3, sharex=True)
for i in range(3):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);

corner.corner(sampler.flatchain, labels=['intercept', 'slope', 'scatter']);

# Next plot ~100 of the samples as models over the data to get an idea of the fit

chain = sampler.flatchain

mp.errorbar(x, y, dy, fmt='o');

thetas = [chain[i] for i in np.random.choice(chain.shape[0], 200)]

xfit = np.arange(-1,0.1,0.1)
for i in range(100):
    theta = thetas[i]
    mp.plot(xfit, theta[0] + theta[1] * xfit,
             color='black', alpha=0.05);
    
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.xlabel('log P (days)')
mp.ylabel('Absolute mag [3.6]')

theta_best = chain.mean(0)
theta_std = chain.std(0)


print 'slope = ', theta_best[1], 'zeropoint = ', theta_best[0], 'instrinsic scatter = ', theta_best[2]

print 'standard deviations: ', theta_std[1], theta_std[0], theta_std[2]

def log_prior(theta):
    if np.all(np.abs(theta) < 1000):
        return 0
    else:
        return -np.inf  # log(0)
    
def log_likelihood(theta, x, y, dy):
    y_model = theta[0] + theta[1] * x
    return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) +
                         (y - y_model) ** 2 / dy ** 2)

def log_posterior(theta, x, y, dy):
    return log_prior(theta) + log_likelihood(theta, x, y, dy)

ndim = 2  # number of parameters in the model
nwalkers = 100  # number of MCMC walkers

# initialize walkers
starting_guesses = np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=[x, y, dy])
pos, prob, state = sampler.run_mcmc(starting_guesses, 200)

fig, ax = mp.subplots(2, sharex=True)
for i in range(2):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);

sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

fig, ax = mp.subplots(2, sharex=True)
for i in range(2):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);

corner.corner(sampler.flatchain, labels=['intercept', 'slope']);

chain = sampler.flatchain

mp.errorbar(x, y, dy, fmt='o');

thetas = [chain[i] for i in np.random.choice(chain.shape[0], 100)]

xfit = np.arange(-1,0,0.1)
for i in range(100):
    theta = thetas[i]
    mp.plot(xfit, theta[0] + theta[1] * xfit,
             color='black', alpha=0.05);
    
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.xlabel('log P (days)')
mp.ylabel('Absolute mag [3.6]')

theta_best = chain.mean(0)
theta_std = chain.std(0)

mp.plot(xfit, theta_best[0] + theta_best[1]*xfit, 'r--')


print 'slope = ', theta_best[1], 'zeropoint = ', theta_best[0]

print 'standard deviations: ', theta_std[1], theta_std[0]

def log_prior(theta):
    if theta[2] <= 0 or np.any(np.abs(theta[:2]) > 10):   ##### making limit on theta smaller so they're reasonable
        return -np.inf  # log(0)
    else:
        # Jeffreys Prior
        return -np.log(theta[2])
    
def log_likelihood(theta, x, y, dy):
    y_model = theta[0] + theta[1] * x
    S = dy ** 2 + theta[2] ** 2
    return -0.5 * np.sum(np.log(2 * np.pi * S) +
                         (y - y_model) ** 2 / S)

def log_posterior(theta, x, y, dy):
    return log_prior(theta) + log_likelihood(theta, x, y, dy)

ndim = 3  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers

# initialize walkers
starting_guesses = np.random.randn(nwalkers, ndim)
starting_guesses[:, 2] = np.random.rand(nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=[x, y, dy])
pos, prob, state = sampler.run_mcmc(starting_guesses, 200)

# Plot the three chains as above

fig, ax = mp.subplots(3, sharex=True)
for i in range(3):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);

sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

fig, ax = mp.subplots(3, sharex=True)
for i in range(3):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);

# Use corner.py to visualize the three-dimensional posterior
corner.corner(sampler.flatchain, labels=['intercept', 'slope', 'scatter']);

chain = sampler.flatchain

mp.errorbar(x, y, dy, fmt='o');

thetas = [chain[i] for i in np.random.choice(chain.shape[0], 100)]

xfit = np.arange(-1, 0, 0.1)
for i in range(100):
    theta = thetas[i]
    mp.plot(xfit, theta[0] + theta[1] * xfit,
             color='black', alpha=0.05);
    
    
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.xlabel('log P (days)')
mp.ylabel('Absolute mag [3.6]')

theta_best = chain.mean(0)
theta_std = chain.std(0)

mp.plot(xfit, theta_best[0] + theta_best[1]*xfit, 'r--')


print 'slope = ', theta_best[1], 'zeropoint = ', theta_best[0], 'scatter = ', theta_best[2]

print 'standard deviations: ', theta_std[1], theta_std[0], theta_std[2]

## Repeat analysis for 4.5um

x = ab_df.log_P
y = ab_df.M_4p5
dy = ab_df.e_M_4p5_gks

ndim = 3  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers

# initialize walkers
starting_guesses = np.random.randn(nwalkers, ndim)
starting_guesses[:, 2] = np.random.rand(nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=[x, y, dy])
pos, prob, state = sampler.run_mcmc(starting_guesses, 200)

# Plot the three chains as above

fig, ax = mp.subplots(3, sharex=True)
for i in range(3):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
    
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

fig, ax = mp.subplots(3, sharex=True)
for i in range(3):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
    
# Use corner.py to visualize the three-dimensional posterior
corner.corner(sampler.flatchain, labels=['intercept', 'slope', 'scatter']);

chain = sampler.flatchain

mp.clf()
mp.errorbar(x, y, dy, fmt='o');

thetas = [chain[i] for i in np.random.choice(chain.shape[0], 100)]

xfit = np.arange(-1, 0, 0.1)
for i in range(100):
    theta = thetas[i]
    mp.plot(xfit, theta[0] + theta[1] * xfit,
             color='black', alpha=0.05);
    
    
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.xlabel('log P (days)')
mp.ylabel('Absolute mag [4.5]')

theta_best = chain.mean(0)
theta_std = chain.std(0)

mp.plot(xfit, theta_best[0] + theta_best[1]*xfit, 'r--')


print 'slope = ', theta_best[1], 'zeropoint = ', theta_best[0], 'scatter = ', theta_best[2]

print 'standard deviations: ', theta_std[1], theta_std[0], theta_std[2]

analysis_df

### Fit 3.6 in all variants

columns = ['slope', 'e_slope', 'zeropoint', 'e_zeropoint', 'mean_logp', 'source', 'type', 'band', 'sample', 'n_stars']

fit_df_final = pd.DataFrame(columns=columns)

p1 = np.arange(-1,0.1,0.1)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)

palette = itertools.cycle(sns.color_palette())
from matplotlib import cm

## Whole sample, no cuts, RRc fundamentalised

popt, pcov = curve_fit(free_fit, analysis_df.logP_f, analysis_df.M_3p6)

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(analysis_df.logP_f), 'source': 'FreeFit', 'type': 'Fund', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(analysis_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(analysis_df.logP_f)

col = next(palette)

axp1 = mp.subplot(311)
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -6.0)

#axp1.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp1.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp1.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='TGAS errors')
axp1.errorbar(analysis_df.logP_f, analysis_df.M_3p6, yerr = analysis_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='GKS errors')
axp1.scatter(analysis_df.ix[analysis_df.Type=='ab', 'logP_f'], analysis_df.ix[analysis_df.Type=='ab', 'M_3p6'], c=analysis_df.ix[analysis_df.Type=='ab', '[Fe/H]'], cmap=cm.Spectral_r, marker='o', zorder=4, label='RRab')
axp1.scatter(analysis_df.ix[analysis_df.Type=='c', 'logP_f'], analysis_df.ix[analysis_df.Type=='c', 'M_3p6'], c=analysis_df.ix[analysis_df.Type=='c', '[Fe/H]'], cmap=cm.Spectral_r, marker='^', zorder=4, label='RRc')

xticklabels = axp1.get_xticklabels()
mp.setp(xticklabels, visible=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, -0.5))
axp1.set_yticks([0, -2, -4, -6])



## whole sample, no cuts, RRab only

axp2 = mp.subplot(312, sharex=axp1, sharey=axp1)

ab_df = analysis_df.where(analysis_df.Type=='ab').dropna(axis=0, how='all')
ab_df = ab_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, ab_df['log_P'],ab_df['M_3p6'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(ab_df['log_P']), 'source': 'FreeFit', 'type': 'ab', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(ab_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(ab_df['log_P'])

axp2.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp2.plot(p1, slope*(p1-mean)+ zp, ls='--', label="[3.6]", color=col)
axp2.errorbar(ab_df['log_P'], ab_df['M_3p6'], yerr = ab_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp2.errorbar(ab_df.log_P, ab_df.M_3p6, yerr = ab_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp2.plot(ab_df.log_P, ab_df.M_3p6, 'o', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
mp.ylabel('Absolute Magnitude 3.6 $\mu$m')

xticklabels = axp2.get_xticklabels()
mp.setp(xticklabels, visible=False)
axp2.set_yticks([0, -2, -4, -6])

## whole sample, no cuts, RRc only

axp3 = mp.subplot(313, sharex=axp1, sharey=axp1)

c_df = analysis_df.where(analysis_df.Type=='c').dropna(axis=0, how='all')
c_df = c_df.reset_index(drop=True)

popt, pcov = curve_fit(free_fit, c_df['log_P'],c_df['M_3p6'])

fit_df_final = fit_df_final.append({'slope': popt[0], 'e_slope': pcov[0][0], 'zeropoint': popt[1], 'e_zeropoint': pcov[1][1], 'mean_logp' : np.mean(c_df['log_P']), 'source': 'FreeFit', 'type': 'c', 'band': '3p6', 'sample' : 'complete', 'n_stars': len(c_df['M_3p6']!=np.nan)}, ignore_index=True)

slope = popt[0]
e_slope = pcov[0][0]
zp = popt[1]
e_zp = pcov[1][1]
mean = np.mean(c_df['log_P'])


axp3.fill_between(p1, ((slope+(2*e_slope))*(p1-mean) + zp - 2*e_zp), ((slope-(2*e_slope))*(p1-mean) + zp + 2*e_zp), alpha = 0.3, color=col)

axp3.plot(p1, slope*(p1-mean)+ zp, ls='--', label="_nolegend_", color=col)
axp3.errorbar(c_df['log_P'], c_df['M_3p6'], yerr = c_df.e_M_3p6_tgas, ls='None',zorder=4, color='Grey', label='_nolegend_')
axp3.errorbar(c_df.log_P, c_df.M_3p6, yerr = c_df.e_M_3p6_gks, ls='None',zorder=4, color=col, label='_nolegend_')
axp3.plot(c_df.log_P, c_df.M_3p6, '^', color=col, ls='None', zorder=4, markeredgecolor='Grey', markeredgewidth=1, label='_nolegend_')
axp3.set_yticks([0, -2, -4, -6])


mp.xlabel('log P (days)')
mp.suptitle('Entire Sample of TGAS-CRRP RRL, extinction corrected')
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.savefig('crrp_tgas_3p6um_ext_corr.pdf')

## Repeat analysis for 3.6, using metallicity as an extra parameter?

x = ab_df.log_P
y = ab_df.M_3p6
z = ab_df['[Fe/H]']
dy = ab_df.e_M_3p6_gks

def log_prior(theta):
    if theta[3] <= 0 or np.any(np.abs(theta[:3]) > 1000):   ##### making limit on theta smaller so they're reasonable
        return -np.inf  # log(0)
    else:
        # Jeffreys Prior
        return -np.log(theta[3])
    
def log_likelihood(theta, x, y, z, dy):
    y_model = theta[0] + theta[1] * x + theta[2] * z
    S = dy ** 2 + theta[3] ** 2
    return -0.5 * np.sum(np.log(2 * np.pi * S) +
                         (y - y_model) ** 2 / S)

def log_posterior(theta, x, y, z, dy):
    return log_prior(theta) + log_likelihood(theta, x, y, z, dy)

ndim = 4  # number of parameters in the model
nwalkers = 100  # number of MCMC walkers

# initialize walkers
starting_guesses = np.random.randn(nwalkers, ndim)
starting_guesses[:, 3] = np.random.rand(nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=[x, y, z, dy])
pos, prob, state = sampler.run_mcmc(starting_guesses, 300)

# Plot the four chains as above

fig, ax = mp.subplots(4, sharex=True)
for i in range(4):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
  

  
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

fig, ax = mp.subplots(4, sharex=True)
for i in range(4):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
    

# Use corner.py to visualize the three-dimensional posterior
corner.corner(sampler.flatchain, labels=['intercept', 'slope', 'metallicity', 'scatter']);

chain = sampler.flatchain

mp.clf()

fig = mp.figure(figsize=(6,6))

mp.errorbar(x, y, dy, ls='None', color='Grey');
im = mp.scatter(x, y, c=z, cmap=cm.Spectral, marker='o', s=50, zorder=4)

thetas = [chain[i] for i in np.random.choice(chain.shape[0], 100)]

xfit = np.arange(-1, 0, 0.1)
zfit = np.arange(-2.5, 0, 0.25)
for i in range(100):
    theta = thetas[i]
    mp.plot(xfit, theta[0] + theta[1] * xfit + theta[2]*zfit,
             color='black', alpha=0.05);
    
    
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.xlabel('log P (days)')
mp.ylabel('Absolute mag [3.6]')

theta_best = chain.mean(0)
theta_std = chain.std(0)

mp.plot(xfit, theta_best[0] + theta_best[1]*xfit + theta_best[2]*zfit, 'k--')

cbar_ax = fig.add_axes([0.95, 0.3, 0.05, 0.5])
cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical', ticks=[-2.5, -2.0, -1.5, -1.0, -0.5, -0.0])
#im.set_clim(169.6,283.)
cb.set_label("[Fe/H]")

title_text = 'M$_{3.6}$ = ' + str(np.round(theta_best[1], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[1], decimals=3)) + ') $\log$ P + ' + str(np.round(theta_best[2], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[2], decimals=3)) + ') [Fe/H] ' + str(np.round(theta_best[0], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[0], decimals=3)) + ') , $\sigma_{int}$ = ' + str(np.round(theta_best[3], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[3], decimals=3)) + ')'

mp.suptitle(title_text)
mp.show()
print 'slope = ', theta_best[1], 'zeropoint = ', theta_best[0], 'metallicity = ', theta_best[2], 'scatter = ',  theta_best[3]

print 'standard deviations: ', theta_std[1], theta_std[0], theta_std[2], theta_std[3]

min(z), max(z)

title_text = 'M$_{3.6}$ = ' + str(np.round(theta_best[1], decimals=3)) + ' $\log$ P + ' + str(np.round(theta_best[2], decimals=3)) + ' [Fe/H] ' + str(np.round(theta_best[0], decimals=3)) + ', $\sigma_{int}$ = ' + str(np.round(theta_best[3], decimals=3))

title_text

best_df = analysis_df.where((analysis_df.e_gks/analysis_df.Plx)<0.1).dropna(axis=0, how='all')

best_df = best_df.reset_index(drop=True)

## Repeat analysis for 3.6, using metallicity as an extra parameter?

x = best_df.logP_f
y = best_df.M_3p6
z = best_df['[Fe/H]']
dy = best_df.e_M_3p6_gks

def log_prior(theta):
    if theta[3] <= 0 or np.any(np.abs(theta[:3]) > 1000):   ##### making limit on theta smaller so they're reasonable
        return -np.inf  # log(0)
    else:
        # Jeffreys Prior
        return -np.log(theta[3])
    
def log_likelihood(theta, x, y, z, dy):
    y_model = theta[0] + theta[1] * x + theta[2] * z
    S = dy ** 2 + theta[3] ** 2
    return -0.5 * np.sum(np.log(2 * np.pi * S) +
                         (y - y_model) ** 2 / S)

def log_posterior(theta, x, y, z, dy):
    return log_prior(theta) + log_likelihood(theta, x, y, z, dy)

ndim = 4  # number of parameters in the model
nwalkers = 100  # number of MCMC walkers

# initialize walkers
starting_guesses = np.random.randn(nwalkers, ndim)
starting_guesses[:, 3] = np.random.rand(nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=[x, y, z, dy])
pos, prob, state = sampler.run_mcmc(starting_guesses, 300)

# Plot the four chains as above

fig, ax = mp.subplots(4, sharex=True)
for i in range(4):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
  

  
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

fig, ax = mp.subplots(4, sharex=True)
for i in range(4):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
    

# Use corner.py to visualize the three-dimensional posterior
corner.corner(sampler.flatchain, labels=['intercept', 'slope', 'metallicity', 'scatter']);

chain = sampler.flatchain

mp.clf()

fig = mp.figure(figsize=(6,6))

mp.errorbar(x, y, dy, ls='None', color='Grey');
im = mp.scatter(x, y, c=z, cmap=cm.Spectral, marker='o', s=50, zorder=4)

thetas = [chain[i] for i in np.random.choice(chain.shape[0], 100)]

xfit = np.arange(-1, 0, 0.1)
zfit = np.arange(-2.5, 0, 0.25)
for i in range(100):
    theta = thetas[i]
    mp.plot(xfit, theta[0] + theta[1] * xfit + theta[2]*zfit,
             color='black', alpha=0.05);
    
    
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.xlabel('log P (days)')
mp.ylabel('Absolute mag [3.6]')

theta_best = chain.mean(0)
theta_std = chain.std(0)

mp.plot(xfit, theta_best[0] + theta_best[1]*xfit + theta_best[2]*zfit, 'k--')

cbar_ax = fig.add_axes([0.95, 0.3, 0.05, 0.5])
cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical', ticks=[-2.5, -2.0, -1.5, -1.0, -0.5, -0.0])
#im.set_clim(169.6,283.)
cb.set_label("[Fe/H]")

title_text = 'M$_{3.6}$ = ' + str(np.round(theta_best[1], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[1], decimals=3)) + ') $\log$ P + ' + str(np.round(theta_best[2], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[2], decimals=3)) + ') [Fe/H] ' + str(np.round(theta_best[0], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[0], decimals=3)) + ') , $\sigma_{int}$ = ' + str(np.round(theta_best[3], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[3], decimals=3)) + ')'

mp.suptitle(title_text)
mp.show()
print 'slope = ', theta_best[1], 'zeropoint = ', theta_best[0], 'metallicity = ', theta_best[2], 'scatter = ',  theta_best[3]

print 'standard deviations: ', theta_std[1], theta_std[0], theta_std[2], theta_std[3]

Image('neeley_m4_pls.png')

## Repeat analysis for 3.6, using metallicity as an extra parameter?

x = ab_df.log_P
y = ab_df.M_3p6
z = ab_df['[Fe/H]']
dy = ab_df.e_M_3p6_gks
dz = np.ones(len(x))*0.15 ### assuming an uncertainty of 0.15 dex on the metallicities

### theta[0] = zero point
## theta[1]  = metallicity term 
## theta[2] = intrinsic scatter



def log_prior(theta):
    if theta[2] <= 0 or np.any(np.abs(theta[:2]) > 1000):   ##### making limit on theta smaller so they're reasonable
        return -np.inf  # log(0)
    else:
        # Jeffreys Prior
        return -np.log(theta[2])
    
def log_likelihood(theta, x, y, z, dy, dz):
    y_model = theta[0] + theta[1] * z - 2.370 * x 
    S = dy ** 2 + theta[2] ** 2 + dz**2 ## adding an extra term for the uncertainty on the metallicities
    return -0.5 * np.sum(np.log(2 * np.pi * S) +
                         (y - y_model) ** 2 / S)

def log_posterior(theta, x, y, z, dy, dz):
    return log_prior(theta) + log_likelihood(theta, x, y, z, dy, dz)

ndim = 3  # number of parameters in the model
nwalkers = 100  # number of MCMC walkers

# initialize walkers
starting_guesses = np.random.randn(nwalkers, ndim)
starting_guesses[:, 2] = np.random.rand(nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=[x, y, z, dy, dz])
pos, prob, state = sampler.run_mcmc(starting_guesses, 300)

# Plot the four chains as above

fig, ax = mp.subplots(3, sharex=True)
for i in range(3):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
  

sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

fig, ax = mp.subplots(3, sharex=True)
for i in range(3):
    ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);

# Use corner.py to visualize the three-dimensional posterior
corner.corner(sampler.flatchain, labels=['intercept', 'metallicity', 'scatter']);

chain = sampler.flatchain

mp.clf()

fig = mp.figure(figsize=(6,6))

mp.errorbar(x, y, dy, ls='None', color='Grey');
im = mp.scatter(x, y, c=z, cmap=cm.Spectral, marker='o', s=50, zorder=4)

thetas = [chain[i] for i in np.random.choice(chain.shape[0], 100)]

xfit = np.arange(-1, 0, 0.1)
zfit = np.arange(-2.5, 0, 0.25)
for i in range(100):
    theta = thetas[i]
    mp.plot(xfit, theta[0] -2.370 * xfit + theta[1]*zfit,
             color='black', alpha=0.05);
    
    
mp.xlim(-0.6, -0.1)
mp.ylim(1.0, -2.5)

mp.xlabel('log P (days)')
mp.ylabel('Absolute mag [3.6]')

theta_best = chain.mean(0)
theta_std = chain.std(0)

mp.plot(xfit, theta_best[0] -2.370*xfit + theta_best[1]*zfit, 'k--')

cbar_ax = fig.add_axes([0.95, 0.3, 0.05, 0.5])
cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical', ticks=[-2.5, -2.0, -1.5, -1.0, -0.5, -0.0])
#im.set_clim(169.6,283.)
cb.set_label("[Fe/H]")

title_text = 'M$_{3.6} = -2.370 \log$ P + ' + str(np.round(theta_best[1], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[1], decimals=3)) + ') [Fe/H] ' + str(np.round(theta_best[0], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[0], decimals=3)) + ') , $\sigma_{int}$ = ' + str(np.round(theta_best[2], decimals=3)) + '($\pm$ ' + str(np.round(theta_std[2], decimals=3)) + ')'

mp.suptitle(title_text)
mp.show()
print 'zeropoint = ', theta_best[0], 'metallicity = ', theta_best[1], 'scatter = ',  theta_best[2]

print 'standard deviations: ', theta_std[0], theta_std[1], theta_std[2]

len(x)




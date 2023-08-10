get_ipython().magic('matplotlib inline')

#import notebook
#from notebook.nbextensions import enable_nbextension
#enable_nbextension('notebook', 'usability/codefolding/main')
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
import markdown
#import scipy.stats as stats
import sys
sys.path.append('/Users/vs/Dropbox/Python')
sys.path.append('/Users/vs/Dropbox/Python/gloess/')
import shutil
import glob
import re
import os
import gloess_fits as gf
import linecache
import numpy.ma as ma
import matplotlib.gridspec as gridspec


#import reddening_laws as red
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

### List of targets with periods
gal_df = pd.read_csv('/Users/vs/Dropbox/CHP/Galactic_Cepheids/Galactic_Cepheids_List', delimiter='&', header=None, names=('Cepheid', 'logP'))
#gal_df = pd.read_csv('/Users/vs/Dropbox/CHP/Galactic_Cepheids/to_refit', delimiter='&', header=None, names=('Cepheid', 'logP'))

#gal_df = pd.read_csv('/Users/vs/Dropbox/CHP/Galactic_Cepheids/Galactic_Cepheids_averages', delim_whitespace=True, header=0)

### Set working directory

os.chdir('/Users/vs/Dropbox/CHP/Galactic_Cepheids/')

### Don't need to strip whitespace from name (should probably fix naming of files in the directories)
### Change names to all caps to match with filenames

### Converting names to string so I can reformat them properly
gal_df['Cepheid'] = gal_df['Cepheid'].astype('str') 

gal_df

mag_columns = []
err_columns = []
cols = ['mjd']
bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '3p6', '4p5', '5p8', '8p0']
for wlen in np.arange(len(bands)):
    mag_name = ('mag_' + str(bands[wlen]))
    err_name = ('err_' + str(bands[wlen]))
    mag_columns.append(mag_name)
    err_columns.append(err_name)
    cols.append(mag_name)
    cols.append(err_name)
    
cols.append('Reference')
#cols

### Going to copy the gloess input files to here, do some editing on them, (periods, cleaning the data) then run gloess
### Setting comment symbol = '-' to remove lines that start with '-'

def gloess_setup(cepheid):
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

    cap_cepheid = str.upper(cepheid)
    orig_file = '/Users/vs/Dropbox/All_Cepheids_ever/MilkyWay/cepheids/' + cap_cepheid
    cepID = re.sub(' ', '', cepheid)
    new_file  = cepID + '.gloess_in'
    old_file = os.path.exists(orig_file)
    is_there = os.path.exists(new_file)

    if (old_file == False):
        if (is_there == False):
            period = 10**(avs_df.logP[avs_df['cepID']==cepID].values)
            smooth = '0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10'
            with open(new_file, 'w') as f:
                f.write("{0:s}\n".format(cepheid))
                f.write("{0:s}\n".format(period))
                f.write("0\n")
                f.write("{0:s}\n".format(smooth))

    if (is_there == False):
        shutil.copy(orig_file, new_file)
    linecache.clearcache()
    smooth_line = linecache.getline(new_file, 4).strip()
    smooth_line = re.sub("[\[\]\'\",]"," ", smooth_line)
    smooth = smooth_line.split()
    df = pd.read_csv(new_file, header=None, skiprows=4, names=(cols), comment='-', delim_whitespace=True)
    return(cepID, df, smooth)

def all_the_gloess(row):
    cepID, df, sm_params = gloess_setup(row['Cepheid'])
    #print sm_params
    name = row.Cepheid
    period = 10**(row.logP)
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
        ## Removing photographic data of unknown photometric system
        df.loc[df['Reference']=='1958BAN....14...81W', mag] = np.nan
        df.loc[df['Reference']=='1982PASP...94..182C', mag] = np.nan
        df.loc[df['Reference']=='1978AJ.....83.1397G', mag] = np.nan
        df.loc[df['Reference']=='1951ApJ...113..367E', mag] = np.nan
        df.loc[df['Reference']=='1961ApJS....6..253I', mag] = np.nan
        df.loc[df['Reference']=='1957MNRAS.117..406E', mag] = np.nan
        df.loc[df['Reference']=='1973AJ.....78..618W', mag] = np.nan
        df.loc[df['Reference']=='1968CoLPL..7....57W', mag] = np.nan
        df.loc[df['Reference']=='1985ApJ...295..507G', mag] = np.nan


        ## Can't locate this reference in ADS
        df.loc[df['Reference']=='1977MSuAW..70.....S', mag] = np.nan
        df.loc[df['Reference']=='10', mag] = np.nan
        df.loc[df['Reference']=='990', mag] = np.nan

        ## Removing data taken in the Johnson system
        if (bands[band] == 'R' or bands[band] == 'I'):
            df.loc[df['Reference']=='1984ApJS...55..389M', mag] = np.nan
            df.loc[df['Reference']=='1997PASP..109..645B', mag] = np.nan
            df.loc[df['Reference']=='1998MNRAS.297..825K', mag] = np.nan
            df.loc[df['Reference']=='1968CoLPL...7...57W', mag] = np.nan
            df.loc[df['Reference']=='1977ApJS...34....1E', mag] = np.nan
            df.loc[df['Reference']=='1986PZ.....22..369B', mag] = np.nan
            df.loc[df['Reference']=='1987PZ.....22..530B', mag] = np.nan
            df.loc[df['Reference']=='1992A&AT....2....1B', mag] = np.nan
            df.loc[df['Reference']=='1992A&AT....2...31B', mag] = np.nan
            df.loc[df['Reference']=='1992A&AT....2...43B', mag] = np.nan
            df.loc[df['Reference']=='1992A&AT....2..107B', mag] = np.nan
            df.loc[df['Reference']=='1992A&AT....2..157B', mag] = np.nan
            df.loc[df['Reference']=='1992A+AT....2....1B', mag] = np.nan
            df.loc[df['Reference']=='1992A+AT....2...31B', mag] = np.nan
            df.loc[df['Reference']=='1992A+AT....2...43B', mag] = np.nan
            df.loc[df['Reference']=='1992A+AT....2..107B', mag] = np.nan
            df.loc[df['Reference']=='1992A+AT....2..157B', mag] = np.nan
            df.loc[df['Reference']=='1992PAZh...18..325B', mag] = np.nan
            df.loc[df['Reference']=='1993PAZh...19..210B', mag] = np.nan
            df.loc[df['Reference']=='1994A+AT....5..317B', mag] = np.nan  
            df.loc[df['Reference']=='1994A&AT....5..317B', mag] = np.nan  
            df.loc[df['Reference']=='1994IBVS.3991....1B', mag] = np.nan    
            df.loc[df['Reference']=='1995IBVS.4141....1B', mag] = np.nan   
            df.loc[df['Reference']=='1998A+AT...16..291B', mag] = np.nan
            df.loc[df['Reference']=='1971ApJ...165..335S', mag] = np.nan       
            df.loc[df['Reference']=='1979PASP...91...67F', mag] = np.nan
            df.loc[df['Reference']=='1981ApJS...47..315G', mag] = np.nan
            df.loc[df['Reference']=='1995PAZh...21..803B', mag] = np.nan


        ### Removing non-IRAC data from the 3.6um columns
        if (bands[band] == '3p6'):
            df.loc[df['Reference']=='1992A&AS...93...93L', mag] = np.nan
            df.loc[df['Reference']=='1968CoLPL..7....57W', mag] = np.nan
            
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
        
        gal_df.ix[gal_df.Cepheid==name, mag]=ave
        gal_df.ix[gal_df.Cepheid==name, err]=sdev/(np.sqrt(len(data1)))
        gal_df.ix[gal_df.Cepheid==name, amplitude]=amp
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

### Only run if not reading previous output

av_cols = []
bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '3p6', '4p5', '5p8', '8p0']
for wlen in np.arange(len(bands)):
    mag_name = ('mag_' + str(bands[wlen]))
    err_name = ('err_' + str(bands[wlen]))
    amp_name = ('amp_' + str(bands[wlen]))
    av_cols.append(mag_name)
    av_cols.append(err_name)
    av_cols.append(amp_name)
av_cols

for columns in np.arange(len(av_cols)):
    gal_df[av_cols[columns]] = np.nan
    

#### Sort out some poorly formatted names
gal_df.Cepheid = gal_df.Cepheid.str.strip()
gal_df.ix[10, 'Cepheid'] = 'Delta Cep'
gal_df.ix[21, 'Cepheid'] = 'Eta Aql'
gal_df.ix[28, 'Cepheid'] = 'Beta Dor'
gal_df.ix[29, 'Cepheid'] = 'Zeta Gem'

### Not yet implimented
#ceps_to_do = pd.read_csv('/Users/vs/Dropbox/CHP/Galactic_Cepheids/to_refit', delimiter='&', header=None, names=('Cepheid', 'logP'))
#### Sort out some poorly formatted names
#ceps_to_do.Cepheid = ceps_to_do.Cepheid.str.strip()

#ceps_to_do['Cepheid'] = ceps_to_do['Cepheid'].map(lambda x: re.sub('[\$\\\~]', '', x))
#ceps_to_do['caps'] = ceps_to_do['Cepheid'].map(lambda x: str.upper(str.upper(x)))
#gal_df['caps'] = gal_df['Cepheid'].map(lambda x: str.upper(str.upper(x)))
#ceps_to_do['Refit'] = True

#ceps_to_do

#do_these_df = pd.merge(gal_df, ceps_to_do, how='outer', on=['caps', 'caps'])

#do_these_df = do_these_df.drop(['Cepheid_y', 'logP_y', 'caps'], 1)
#do_these_df = do_these_df.rename(columns={'Cepheid_x':'Cepheid', 'logP_x': 'logP'})
#do_these_df

gal_df

gal_df.apply(lambda line: all_the_gloess(line), axis=1)

gal_df

gal_df.to_csv('Galactic_Cepheids_Averages', index=False, header=True, sep=' ', float_format='%4.3f', na_rep= 99.99)

period = 3.7281
df['phase'] = (df['MJD'] / period) - np.floor(df['MJD'] / period)
fit_mags = (df[mag].ix[df[mag]<99]).values
fit_errs = (df[err].ix[df[mag]<99]).values
fit_phases = (df['phase'].ix[df[mag]<99]).values
fit_len = len(fit_mags)
data1, x, y, yerr, xphase = gf.fit_one_band(fit_mags, fit_errs, fit_phases, fit_len,0.1)
ave, adev, sdev, var, skew, kurtosis, amp = gf.moment(data1[200:300],100)

mag = 'mag_3p6'
err = 'err_3p6'

ave, adev, sdev, var, skew, kurtosis, amp = gf.moment(data1[200:300],100)

pullmags = df[mag]
pullphase = df['phase']

plotmag = pullmags[pullmags<50]
plotphase = pullphase[pullmags<50]

plotmag = np.concatenate((plotmag,plotmag,plotmag,plotmag,plotmag))
plotphase = np.concatenate((plotphase,(plotphase+1.0),(plotphase+2.0),(plotphase+3.0),(plotphase+4.0)))
print len(plotphase), len(plotmag)


mp.plot(x,data1,'k-', zorder=4)
mp.plot(plotphase, plotmag,'ro',ls='None')
mp.xlim(1,2.5)

data1

fit_errs

os.getcwd()

new_file  = 'AQPup' + '.gloess_in'
linecache.clearcache()
smooth_line = linecache.getline(new_file, 4).strip()
smooth_line = re.sub("[\[\]\'\",]"," ", smooth_line)
smooth = smooth_line.split()

smooth

new_file

gal_df

np.nansum(df.Reference)

cepID, df, sm_params = gloess_setup('RT Aur')
mag = 'mag_B'
err = 'err_B'

df.loc[df[err]==9.99, mag].where(df[mag]==0)




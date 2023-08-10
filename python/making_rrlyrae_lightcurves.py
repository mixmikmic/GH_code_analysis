get_ipython().magic('matplotlib inline')
#import notebook
#from notebook.nbextensions import enable_nbextension
#enable_nbextension('notebook', 'usability/codefolding/main')
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns
import pandas as pd
import sys
#sys.path.append('/Users/vs/Dropbox/Python')
from astropy.time import Time
import glob
import itertools
import gloess_fits as gf
import os
import re
from astropy.time import Time
import make_lightcurves_from_idl as idl_lc



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

### Read in the RRL info file

directory = "/Users/vs/Dropbox/CRRP/RR_Lyrae_lightcurves/S19p2_reduction"
os.chdir(directory)

info_df = pd.read_csv('rrl_periods', delim_whitespace=True, header=None, names=['Name', 'Period', 'Type'])
info_df['Name_Lower'] = map(str.lower, info_df.Name)
info_df

unique_names = info_df.Name.unique()
len(unique_names)

### Get an array containing the file names

filenames = glob.glob('*_phot')
mashed_names = []
for name in np.arange(len(filenames)):
    mashed_names.append(re.sub('_phot', '', filenames[name]))
    mashed_names[name] = re.sub('_', '', mashed_names[name])
filenames_df = pd.DataFrame(filenames)
mashednames_df = pd.DataFrame(map(str.lower, mashed_names))
names_df = pd.concat([filenames_df, mashednames_df], 1)
names_df.columns = ['Filename', 'Name_Lower']

rrl_df = info_df.merge(names_df, on='Name_Lower')

rrl_df

zp36 = 280.9
zp45 = 179.7
apcor36 = 1.125
apcor45 = 1.123

for rrl in np.arange(len(rrl_df)):
    idl_lc.make_lightcurve(rrl_df.Filename[rrl], rrl_df.Period[rrl], apcor1=apcor36, apcor2=apcor45, zp1=zp36, zp2=zp45)
    

filename = rrl_df.Filename[0]

data_df = pd.read_csv(filename, delim_whitespace=True, header=None, names=('Channel', 'MJD', 'flux', 'eflux', 'bigflux'))

data_df

channel1_df = data_df[data_df.Channel==1]

sorted_ch1_df = channel1_df.sort_values(by='MJD')

sorted_ch1_df = sorted_ch1_df.reset_index(drop=True)
sorted_ch1_df

diffs = sorted_ch1_df.MJD[10:15]-sorted_ch1_df.MJD[10]

diffs

ratio1 = diffs/diffs[1]
ratio2 = diffs/diffs[2]
ratio3 = diffs/diffs[3]
ratio4 = diffs/diffs[4]
ratio5 = diffs/diffs[5]

np.floor(ratio1), np.floor(ratio2), np.floor(ratio3), np.floor(ratio4), np.floor(ratio5)

sorted_ch1_df['timebin'] = 0

sorted_ch1_df

np.arange(0, len(sorted_ch1_df), 5)

thistimebin = 0
obs=0
binwidth = 5
### imitating a do-while loop here
### run while datapoints remaining

datarem=True
med_diff = 0
#while (obs < (len(sorted_ch1_df))-1):
#for obs in np.arange(len(sorted_ch1_df)):
while datarem == True:
    old_diff = med_diff
    print 'start of loop' + str(obs)
    if (obs + binwidth > len(sorted_ch1_df)):
        diffs = sorted_ch1_df.MJD[obs:]-sorted_ch1_df.MJD[obs]
        diffs = diffs.reset_index(drop=True)
    else:
        diffs = sorted_ch1_df.MJD[obs:obs+binwidth]-sorted_ch1_df.MJD[obs]
        diffs = diffs.reset_index(drop=True)
    if len(diffs) > 1:
        ratio = diffs / diffs[1]
        for dither in np.arange(len(ratio)):
            if (np.floor(ratio[dither]) in np.arange(binwidth)):
                print obs, thistimebin, ratio[dither]
                sorted_ch1_df.ix[obs, 'timebin'] =  thistimebin
                obs = obs + 1
            else:
                break
    if(obs < len(sorted_ch1_df)): 
        datarem = True
    elif(obs == len(sorted_ch1_df)): 
        diffs = sorted_ch1_df.MJD[obs-1:]-sorted_ch1_df.MJD[obs-1]
        diffs = diffs.reset_index(drop=True)
        ratio = diffs / diffs[1]
        if (np.floor(ratio[1]) == 1.0):
            print obs, thistimebin, ratio[1]
            sorted_ch1_df.ix[obs, 'timebin'] =  thistimebin
        else:
            print obs, thistimebin, ratio[1]
            sorted_ch1_df.ix[obs, 'timebin'] =  thistimebin+1
            datarem = False
    else:
        datarem = False
    thistimebin = thistimebin + 1
    med_diff = np.median(diffs)
    print old_diff, med_diff
    if ((med_diff > 10*old_diff) and obs > 5):
        obs = obs - binwidth
        binwidth = binwidth - 1
        med_diff = old_diff
    if ((med_diff < 10*old_diff) and binwidth < 5):
        binwidth = 5
    if(binwidth == 0):
        break

sorted_ch1_df

sorted_ch1_df.timebin

binwidth=5
np.arange(binwidth)



if (np.floor(ratio1[0]) == 0): ### First dither in set
    sorted_ch1_df.ix[sorted_ch1_df.timebin==obs, 'timebin'] = thistimebin
    obs = obs + 1
    if (np.floor(ratio1[1]) == 1): ### Second dither
        sorted_ch1_df.ix[sorted_ch1_df.timebin==obs, 'timebin'] = thistimebin
        obs = obs + 1
        if (np.floor(ratio1[2]) == 2): ### third dither
            sorted_ch1_df.ix[sorted_ch1_df.timebin==obs, 'timebin'] = thistimebin
            obs = obs + 1
            if (np.floor(ratio1[3]) == 3): ### fourth dither
                sorted_ch1_df.ix[sorted_ch1_df.timebin==obs, 'timebin'] = thistimebin
                obs = obs + 1
                if (np.floor(ratio1[4]) == 4): ### final possible dither
                    sorted_ch1_df.ix[sorted_ch1_df.timebin==obs, 'timebin'] = thistimebin
                    obs = obs + 1
print 'end of loop' + str(obs)
thistimebin = thistimebin + 1





get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
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
import linecache


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

### Set working directory

os.chdir('/Users/vs/Dropbox/CHP/Galactic_Cepheids/')

colspecs = [(0,12), (13,24), (23,29), (30,35), (36,41), (42,47)]
mtr_df = pd.read_fwf('apj447280t3_mrt.txt', colspecs=colspecs, skiprows=20, header=None, names=('Cepheid', 'MJD', 'mag_3p6', 'err_3p6', 'mag_4p5', 'err_4p5') )

avs_df = pd.read_csv('apj447280t4_ascii.txt', skiprows=5, delim_whitespace=True, header=None, names=('Cep1', 'Cep2', 'logP', 'mag_3p6', 'err_3p6', 'mag_4p5', 'err_4p5', 'color', 'err_color'), skipfooter=3)

#df.Year.str.cat(df.Quarter)
avs_df['cepID'] = avs_df.Cep1.str.cat(avs_df.Cep2)
avs_df = avs_df.drop(['Cep1', 'Cep2'], 1)

avs_df['cepID'].unique()

mtr_df

### reformat shitty names
mtr_df['Cepheid'].unique()

mtr_df.Cepheid.ix[mtr_df['Cepheid'] == '{beta} Dor'] = 'beta Dor'
mtr_df.Cepheid.ix[mtr_df['Cepheid'] == '{delta} Cep'] = 'delta Cep'
mtr_df.Cepheid.ix[mtr_df['Cepheid'] == '{eta} Aql'] = 'eta Aql'
mtr_df.Cepheid.ix[mtr_df['Cepheid'] == '{zeta} Gem'] = 'zeta Gem'
mtr_df.Cepheid.ix[mtr_df['Cepheid'] == '{ell} Car'] = 'l Car'

avs_df.cepID.ix[avs_df['cepID'] == 'ellCar'] = 'lCar'

unique_names = mtr_df['Cepheid'].unique()

unique_names

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
    if (old_file == False):
        period = 10**(avs_df.logP[avs_df['cepID']==cepID].values)
        smooth = '0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10'
        with open(new_file, 'w') as f:
            f.write("{0:s}\n".format(cepheid))
            f.write("{0:s}\n".format(period))
            f.write("0\n")
            f.write("{0:s}\n".format(smooth))

    is_there = os.path.exists(new_file)
    if (is_there == False):
        shutil.copy(orig_file, new_file)
    linecache.clearcache()
    smooth_line = linecache.getline(new_file, 4)
    smooth = smooth_line.split()
    df = pd.read_csv(new_file, header=None, skiprows=4, names=(cols), comment='-', delim_whitespace=True)
    return(cepID, df, smooth)

def add_to_file(cepheid):
    ### select rows from df
    new_df = mtr_df[mtr_df['Cepheid']==cepheid]
    new_df = new_df.drop(['Cepheid'],1)
    cepID, orig_df, smooth = gloess_setup(cepheid)
    complete_df = pd.concat([orig_df, new_df], ignore_index=True)
    outfile = cepID + '.gloess_in'
    period_line = linecache.getline(outfile, 3)
    period = period_line.split()
    
    printcols = ['MJD']
    bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '3p6', '4p5', '5p8', '8p0']
    for wlen in np.arange(len(bands)):
        mag_name = ('mag_' + str(bands[wlen]))
        err_name = ('err_' + str(bands[wlen]))
        printcols.append(mag_name)
        printcols.append(err_name)
    printcols.append('Reference')
    
    with open(outfile, 'w') as f:
        f.write("{0:s}\n".format(cepheid))
        f.write("{0:s}\n".format(period))
        f.write("{0:d}\n".format(len(complete_df)))
        f.write("{0:s}\n".format(smooth))
    complete_df.to_csv(outfile, na_rep= 99.99, float_format='%3.4f', header=None, index=False, mode='a', sep=' ', columns=printcols)
    return(complete_df)

#gal_df.apply(lambda line: all_the_gloess(line), axis=1)
for cepheids in np.arange(len(unique_names)):
    add_to_file(unique_names[cepheids])

rtaur_df = add_to_file('RT Aur')

cepheid = 'beta Dor'
new_df = mtr_df[mtr_df['Cepheid']==cepheid]

avs_df.logP[avs_df['cepID']=='betaDor'].values

unique_avs_names

rtaur_df

cols = ['MJD']
bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '3p6', '4p5', '5p8', '8p0']
for wlen in np.arange(len(bands)):
    mag_name = ('mag_' + str(bands[wlen]))
    err_name = ('err_' + str(bands[wlen]))
    cols.append(mag_name)
    cols.append(err_name)
cols




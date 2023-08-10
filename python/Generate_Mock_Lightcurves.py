get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload #Use this to reload modules if they are changed on disk while the notebook is running')
from __future__ import division
from snmachine import sndata, snfeatures, snclassifier, tsne_plot
import numpy as np
import matplotlib.pyplot as plt
import time, os, pywt,subprocess
from sklearn.decomposition import PCA
from astropy.table import Table,join,vstack
from astropy.io import fits
import sklearn.metrics 
import sncosmo
get_ipython().magic('matplotlib nbagg')
from astropy.table import Column

dataset='spcc'

# WARNING...
#Multinest uses a hardcoded character limit for the output file names. I believe it's a limit of 100 characters
#so avoid making this file path to lengthy if using nested sampling or multinest output file names will be truncated

#Change outdir to somewhere on your computer if you like
outdir=os.path.join('output_%s_no_z' %dataset,'')
out_features=os.path.join(outdir,'features') #Where we save the extracted features to
out_class=os.path.join(outdir,'classifications') #Where we save the classification probabilities and ROC curves
out_int=os.path.join(outdir,'int') #Any intermediate files (such as multinest chains or GP fits)

subprocess.call(['mkdir',outdir])
subprocess.call(['mkdir',out_features])
subprocess.call(['mkdir',out_class])
subprocess.call(['mkdir',out_int])

#Data root
rt=os.path.join('SPCC_SUBSET','')

dat=sndata.Dataset(rt)

read_from_file=False #True #We can use this flag to quickly rerun from saved features
run_name=os.path.join(out_features,'%s_all' %dataset)

dat.object_names[0]
types=dat.get_types()
types['Type'][np.floor(types['Type']/10)==2]=2
types['Type'][np.floor(types['Type']/10)==3]=3

#test_table = dat.data['DES_SN001695.DAT'] #get_lightcurve

#dat.plot_all()
#dat.data[dat.object_names[0]]



realspertype=667

type_II = []
for i in range(len(dat.data['DES_SN002457.DAT']['flux'])):
    type_II.append(np.random.normal(dat.data['DES_SN002457.DAT']['flux'][i], dat.data['DES_SN002457.DAT']['flux_error'][i], realspertype))

type_Ia = []
for i in range(len(dat.data['DES_SN002542.DAT']['flux'])):
    type_Ia.append(np.random.normal(dat.data['DES_SN002542.DAT']['flux'][i], dat.data['DES_SN002542.DAT']['flux_error'][i], realspertype))

type_Ibc = []
for i in range(len(dat.data['DES_SN005399.DAT']['flux'])):
    type_Ibc.append(np.random.normal(dat.data['DES_SN005399.DAT']['flux'][i], dat.data['DES_SN005399.DAT']['flux_error'][i], realspertype))

type_Ia = np.array(type_Ia)
type_Ibc = np.array(type_Ibc)
type_II = np.array(type_II)

#test_table.replace_column('flux', type_II[:,0])

#test_table.write('testing.dat',format = 'ascii')

#dat.data['DES_SN001695.DAT']
#print type_Ia

#isinstance(dat.object_names, np.ndarray)

#test_table_II = dat.data['DES_SN002457.DAT']

#dir(dat)

#print data[0:10]
#print data[5].split()
def make_new_data(filename, table, output, sntype):
    with open(filename) as f:
        data = f.read().split("\n")
    filters = data[5].split()
    survey = data[0].split()
    stuffRA = data[6].split()
    stuffDec = data[7].split()
    MWEBV = data[11].split()
    if sntype==1:
        typestring='SN Type = Ia , MODEL = mlcs2k2.SNchallenge'
    elif sntype==2:
        typestring='SN Type = II , MODEL = SDSS-017564'
    elif sntype==3:
        typestring='SN Type = Ic , MODEL = SDSS-014475'
    else:
        typestring='SOMETHING WENT HORRIBLY WRONG'
    table.meta = {survey[0][:-1]: survey[1], stuffRA[0][:-1]: stuffRA[1], stuffDec[0][:-1]: stuffDec[1],filters[0][:-1]: filters[1],
                 MWEBV[0][:-1]: MWEBV[1], 'SNTYPE': -9, 'SIM_COMMENT': typestring  }
    #table.rename_column('mjd', 'MJD')
    #table.rename_column('filter ', 'FLT')
    #table.rename_column('flux', 'FLUXCAL')
    #table.rename_column('flux_error', 'FLUXCALERR')
    sncosmo.write_lc(table, 'new_mocks/nodesz_%s.dat'%output,pedantic=True, format = 'snana')

filename_II = 'SPCC_SUBSET/DES_SN002457.DAT'
filename_Ia = 'SPCC_SUBSET/DES_SN002542.DAT'
filename_Ibc = 'SPCC_SUBSET/DES_SN005399.DAT'

for i in range(realspertype):
    test_table_II = dat.data['DES_SN002457.DAT'].copy()
    test_table_Ia = dat.data['DES_SN002542.DAT'].copy()
    test_table_Ibc = dat.data['DES_SN005399.DAT'].copy()
    #print test_table_II

    #test_table_Ia.replace_column('mjd', test_table_Ia['mjd']+56178)
    #test_table_II.replace_column('mjd', test_table_II['mjd']+56178)
    #test_table_Ibc.replace_column('mjd', test_table_Ibc['mjd']+56178)

    test_table_II.rename_column('flux', 'FLUXCAL')
    test_table_II.rename_column('flux_error', 'FLUXCALERR')
    
    test_table_Ibc.rename_column('flux', 'FLUXCAL')
    test_table_Ibc.rename_column('flux_error', 'FLUXCALERR')
        
    test_table_Ia.rename_column('flux', 'FLUXCAL')
    test_table_Ia.rename_column('flux_error', 'FLUXCALERR')

    objs=[]
    new_data={}

    col_Ia = Table.Column(name='field',data=np.zeros(len(test_table_Ia)) )
    test_table_Ia.add_column(col_Ia, index = 2)
    col_Ibc = Table.Column(name='field',data=np.zeros(len(test_table_Ibc)) )
    test_table_Ibc.add_column(col_Ibc, index = 2)
    col_II = Table.Column(name='field',data=np.zeros(len(test_table_II)) )
    test_table_II.add_column(col_II, index = 2)


    test_table_Ia.replace_column('FLUXCAL', type_Ia[:,i])
    test_table_Ibc.replace_column('FLUXCAL', type_Ibc[:,i])
    test_table_II.replace_column('FLUXCAL', type_II[:,i])
    
    #test_table_Ia.replace_column('FLUXCALERR', test_table_Ia['FLUXCALERR'] + test_table_Ia['FLUXCALERR']*0.02)
    #test_table_Ibc.replace_column('FLUXCALERR', test_table_Ibc['FLUXCALERR'] + test_table_Ibc['FLUXCALERR']*0.02)
    #test_table_II.replace_column('FLUXCALERR', test_table_II['FLUXCALERR'] + test_table_II['FLUXCALERR']*0.02)
    
    wh_desr_II = np.where(test_table_II['filter']=='desz')
    wh_desr_Ia = np.where(test_table_Ia['filter']=='desz')
    wh_desr_Ibc = np.where(test_table_Ibc['filter']=='desz')
    
    test_table_II.remove_rows(wh_desr_II)
    test_table_Ia.remove_rows(wh_desr_Ia)
    test_table_Ibc.remove_rows(wh_desr_Ibc)
    
    make_new_data(filename_Ibc, test_table_Ibc, 'Ibc_%s'%i, 3)
    make_new_data(filename_II, test_table_II, 'II_%s'%i, 2)
    make_new_data(filename_Ia, test_table_Ia, 'Ia_%s'%i, 1)
    
    
    #objs.append((str)(i))
    #new_data[(str)(i)]=test_table_II.copy()  #new_light_curve_Ia
    #new_data[(str)(i+realspertype)]=test_table_Ia.copy()#new_light_curve_II
    #new_data[(str)(i+realspertype)]=test_table_Ibc.copy()#new_light_curve_Ibc
    
#sncosmo.read_lc(new_data)
#new_data = Counter(new_data_Ia) + Counter(new_data_II) + Counter(new_data_Ibc)

#for i in range(3*realspertype):    
#    objs.append((str)(i))
    
#dat.object_names=np.asarray(objs)
#dat.data=new_data

#print test_table_Ia
#test_table_Ia.replace_column('FLUXCALERR', test_table_Ia['FLUXCALERR']*5.)
#print test_table_Ia
#test_table_II = dat.data['DES_SN002457.DAT']
#wh_desr = np.where(test_table_II['filter']=='desr')
#test_table_II.remove_rows(wh_desr)
#print test_table_II

rt1=os.path.join('new_mocks','')
dat1=sndata.Dataset(rt1)

for obj in dat1.object_names:
    for i in range(len(dat1.data[obj])):
        dat1.data[obj]['filter'][i]=dat1.data[obj]['filter'][i][3:7]

#types

#For now we restrict ourselves to three supernova types: Ia (1), II (2) and Ibc (3)
types=dat1.get_types()
types['Type'][np.floor(types['Type']/10)==2]=2
types['Type'][np.floor(types['Type']/10)==3]=3

#dat1.plot_all()
#dat1.get_types()



mod1Feats=snfeatures.ParametricFeatures('karpenka',sampler='leastsq')

get_ipython().run_cell_magic('capture', '--no-stdout', "if read_from_file:\n    mod1_features=Table.read('%s_karpenka.dat' %run_name, format='ascii')\n    blah=mod1_features['Object'].astype(str)\n    mod1_features.replace_column('Object', blah)\nelse:\n    mod1_features=mod1Feats.extract_features(dat1,nprocesses=4,chain_directory=out_int)\n    mod1_features.write('%s_karpenka.dat' %run_name, format='ascii')")

#Unfortunately, sometimes the fitting methods return NaN for some parameters for these models.
for c in mod1_features.colnames[1:]:
    mod1_features[c][np.isnan(mod1_features[c])]=0
    
#print mod1_features[0], len(mod1_features)

mod1Feats.fit_sn
dat1.set_model(mod1Feats.fit_sn,mod1_features)

dat1.plot_all()



plt.figure()
tsne_plot.plot(mod1_features,join(mod1_features,types)['Type'])

plt.figure()
clss=snclassifier.run_pipeline(mod1_features,types,output_name=os.path.join(out_class,'model2'),
                          classifiers=['nb','svm','boost_dt'], nprocesses=4)

np.sort(dat.object_names)

dat.set_model(waveFeats.fit_sn,wave_features,PCA_vec,PCA_mean,0,dat.get_max_length(),dat.filter_set)

#dat.plot_all()

#wave_features

#types

#join(wave_features, types)

#plt.figure()
#tsne_plot.plot(wave_features,join(wave_features,types)['Type'])

#This code takes the fitted parameters and generates the model light curve for plotting purposes.
#dat.set_model(salt2Feats.fit_sn,salt2_features)

#dat.plot_all()

#dat.data[dat.object_names[5]]






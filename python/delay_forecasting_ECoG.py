import os
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.signal as sig
import math as math
import random 
import statsmodels.tsa.stattools as sm
from scipy import integrate
import dynamical.nonlinear as nl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-colorblind')
plt.rcParams['image.cmap'] = 'RdBu'

channel = 100
session = 1
data_path = '/Users/rgao/Documents/Data/NeuroTycho/Propofol/20120730PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/'

matfile = io.loadmat(data_path +'Session' +str(session) + '/ECoG_ch' + str(channel) +'.mat', squeeze_me=True)
timefile = io.loadmat(data_path + 'Session' +str(session) + '/Condition.mat', squeeze_me=True)
ecog = matfile['ECoGData_ch'+str(channel)]
data = ecog[timefile['ConditionIndex'][0]:timefile['ConditionIndex'][1]]

#reload(nl)
X_train=data[740000:750000]
X_test=data[25000:26000]

pred, val = nl.predict_at(X_train, X_test, dim=3, future=4, nn=3)
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(X_train, 'k')
plt.legend(['Training'])
plt.subplot(2,1,2)
plt.plot(val[:])
plt.plot(pred[:])
plt.legend(['Actual', 'Predicted'])
np.corrcoef(pred,val)

data_path = '/Users/rgao/Documents/Data/NeuroTycho/Propofol/20120730PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/'
result_path = '/Users/rgao/Documents/Data/NeuroTycho/Propofol/20120730PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/forecasting/'

# data division
channels = np.append(np.arange(1,63), np.arange(64,129))
#15k points at 500Hz (downsampled) is 30seconds
train_len = 2000 #train on 2k data points
test_len = 500 #test on 500 data points
pad_len = 2000 #grab testing data 500 points after training data
skip_len = 5000 #skip the first 10s of data

#analysis params
max_dim = 8
max_future = 25
max_nn = 20
tau = 10

data_conds = [[0,2],[0,1,3]]
N_chks = 12

#loop through channels
for channel in channels:
    #aggregated result for channel
    rho = np.zeros((N_chks*5,max_dim,max_future,max_nn))
    rmse = np.zeros((N_chks*5,max_dim,max_future,max_nn))
    seg_cnt = 0
    print 'Chan: ', channel
    #loop through sessions
    for session in range(1,3):
        #print 'Session:', session
        timefile = io.loadmat(data_path + 'Session' +str(session) + '/Condition.mat', squeeze_me=True)
        matfile = io.loadmat(data_path +'Session' +str(session) + '/ECoG_ch' + str(channel) +'.mat', squeeze_me=True)
        ecog = matfile['ECoGData_ch'+str(channel)]
        for cond in data_conds[session-1]:
            cond_idx = timefile['ConditionIndex'][cond:cond+2]
            data = sig.decimate(ecog[cond_idx[0]:cond_idx[1]], 2, zero_phase=True)
            seg_inds = skip_len+np.arange(0,N_chks,dtype='int')*len(data)/N_chks
            for seg in seg_inds:
                train_data = data[seg:seg+train_len]
                test_data = data[seg+train_len+pad_len:seg+train_len+pad_len+test_len]
                rho[seg_cnt,:,:,:], rmse[seg_cnt,:,:,:] = nl.delay_embed_forecast(train_data, test_data, tau, max_dim, max_future, max_nn)
                seg_cnt+=1
        
    sp.io.savemat(result_path+'forecast_chan'+str(channel)+'.mat', {'rmse':rmse, 'rho':rho})                

data_path = '/Users/rgao/Documents/Data/NeuroTycho/Propofol/20120730PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/'
result_path = '/Users/rgao/Documents/Data/NeuroTycho/Propofol/20120730PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/forecasting_diff/'

# data division
channels = np.append(np.arange(1,63), np.arange(64,129))
#15k points at 500Hz (downsampled) is 30seconds
train_len = 2000 #train on 2k data points
test_len = 500 #test on 500 data points
pad_len = 2000 #grab testing data 500 points after training data
skip_len = 5000 #skip the first 10s of data

#analysis params
max_dim = 8
max_future = 25
max_nn = 20
tau = 10

data_conds = [[0,2],[0,1,3]]
N_chks = 12

#loop through channels
for channel in channels:
    #aggregated result for channel
    rho = np.zeros((N_chks*5,max_dim,max_future,max_nn))
    rmse = np.zeros((N_chks*5,max_dim,max_future,max_nn))
    seg_cnt = 0
    print 'Chan: ', channel
    #loop through sessions
    for session in range(1,3):
        #print 'Session:', session
        timefile = io.loadmat(data_path + 'Session' +str(session) + '/Condition.mat', squeeze_me=True)
        matfile = io.loadmat(data_path +'Session' +str(session) + '/ECoG_ch' + str(channel) +'.mat', squeeze_me=True)
        ecog = matfile['ECoGData_ch'+str(channel)]
        for cond in data_conds[session-1]:
            cond_idx = timefile['ConditionIndex'][cond:cond+2]
            # first difference
            data = np.diff(sig.decimate(ecog[cond_idx[0]:cond_idx[1]], 2, zero_phase=True))
            seg_inds = skip_len+np.arange(0,N_chks,dtype='int')*len(data)/N_chks
            for seg in seg_inds:
                train_data = data[seg:seg+train_len]
                test_data = data[seg+train_len+pad_len:seg+train_len+pad_len+test_len]
                rho[seg_cnt,:,:,:], rmse[seg_cnt,:,:,:] = nl.delay_embed_forecast(train_data, test_data, tau, max_dim, max_future, max_nn)
                seg_cnt+=1
        
    sp.io.savemat(result_path+'forecast_chan'+str(channel)+'.mat', {'rmse':rmse, 'rho':rho})                




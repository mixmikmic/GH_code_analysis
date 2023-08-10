import os
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.signal as sig
import math as math
import random
import dynamical.nonlinear as nl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-colorblind')
plt.rcParams['image.cmap'] = 'RdBu'

def load_res_chan(res_path, chan,load_nnR2=False):
    """
    returns MI, tau, pffn & attr_dim for one channel
    1-indexed (channel 1 is actually channel1 in ecog)
    if load_Rsq is True, load also del_R and attr_size
    """
    
    res_chan = sp.io.loadmat(result_path +'forecast_chan' + str(chan) + '.mat', struct_as_record=False, squeeze_me=True)['results']
    MI = np.array([seg.MI for seg in res_chan])
    tau = np.array([seg.tau for seg in res_chan])
    pffn = np.array([seg.pffn for seg in res_chan])
    attr_dim = np.array([seg.attr_dim for seg in res_chan])
    if load_nnR2:
        del_R = np.array([seg.del_R for seg in res_chan])
        attr_size = np.array([seg.attr_size for seg in res_chan])
        return MI, tau, pffn, attr_dim, del_R, attr_size
    else:
        return MI, tau, pffn, attr_dim
    
def ctx_viz(ctx_file, data='none', chans='none', ms=20.):
    """
    plots the cortex image, optionally scale colors by 
    a 1-D data vector
    """
    ctx_mat = sp.io.loadmat(ctx_file, squeeze_me=True)
    if data is not 'none' and chans is 'none':
        #need to fill in channel numbers
        chans = np.arange(len(data))
        
    plt.imshow(ctx_mat['I'])
    if data is 'none':
#        plt.plot(ctx_mat['X'],ctx_mat['Y'], 'ko', ms=ms)
        plt.scatter(ctx_mat['X'],ctx_mat['Y'], marker='o', c='w', s=ms) 
    else:
        plt.scatter(ctx_mat['X'][chans],ctx_mat['Y'][chans], marker='o', s=ms, c=data, cmap='Blues') 
        cbar = plt.colorbar(fraction=0.05)
        cbar.set_ticks([min(data),max(data)])
    
    plt.box('off')    
    plt.xlim([50, 950])
    plt.ylim([1200, 40])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()

exp = 'forecasting_diff/'
result_path = '/Users/rgao/Documents/Data/NeuroTycho/Propofol/20120730PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/'+exp

chan = 1
res_chan = sp.io.loadmat(result_path +'forecast_chan' + str(chan) + '.mat', struct_as_record=False, squeeze_me=True)
n_segs=res_chan['rmse'].shape[0]
#60, 8, 25, 20
dim=3
future=20
nn=10
channels = np.append(np.arange(1,63), np.arange(64,129))
rmse = np.zeros((len(channels),n_segs))
rho = np.zeros((len(channels),n_segs))
for idx, chan in enumerate(channels):
    print chan,
    res_chan = sp.io.loadmat(result_path +'forecast_chan' + str(chan) + '.mat', struct_as_record=False, squeeze_me=True)
#     rho[idx,:] = res_chan['rho'][:,dim,future,nn]
#     rmse[idx,:] = res_chan['rmse'][:,dim,future,nn]
    rho[idx,:] = np.mean(res_chan['rho'][:,dim,1:5,nn],axis=1)
    rmse[idx,:] = np.mean(res_chan['rmse'][:,dim,1:5,nn],axis=1)

plotting = 'rho'
if plotting is 'rmse':
    plot_data = rmse
    ti = 'RMSE'
elif plotting is 'rho':
    plot_data = rho
    ti = 'Corr Coef'

cond_labels = ['Eyes Open', 'Eyes Close', 'Induction', 'Anesthetized', 'Recover']
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(plot_data, interpolation='none', aspect='auto', origin='lower', cmap='Blues')
plt.colorbar()
XL = plt.xlim()
YL = plt.ylim()
plt.xticks(range(6,60,12),cond_labels, rotation=-45)
plt.yticks([])
plt.ylabel('Channels')
plt.plot([np.arange(12,60,12)-0.5]*2,plt.ylim(), 'k', lw=4)
plt.xlim(XL)
plt.ylim(YL)
plt.title(ti)

plt.subplot(1,2,2)
X_ = np.mean(plot_data,axis=0)
XS_ = np.std(plot_data,axis=0)
plt.fill_between(range(60), X_-XS_, X_+XS_, alpha=0.5)
plt.plot(range(60),X_, 'k.-')
YL = plt.ylim()
plt.plot([np.arange(12,60,12)-0.5]*2,YL, 'k--', lw=1)
plt.ylim(YL)
plt.xticks(range(6,60,12),cond_labels, rotation=-45);
plt.ylabel(ti)

ctx_file = '/Users/rgao/Documents/Data/NeuroTycho/Propofol/GridLocations/20110621KTMD_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_2Dimg/ChibiMap.mat'
plt.figure(figsize=(4,4))
#data = np.array([ch[15] for ch in tau])
data = np.mean(AD[:,1:6],axis=1)-np.mean(AD[:,6:12],axis=1)
data = np.mean(AD[:,6:12],axis=1)-np.mean(AD[:,18:24],axis=1)
ctx_viz(ctx_file, data=data, chans = channels-1, ms=40.)
plt.title('Dimension Difference (EyesClosed-Anesthesized)')
#ctx_viz(ctx_file, ms=40.)



print res_chan['rho'].shape
rho_ = res_chan['rho']
rmse_ = res_chan['rmse']
plot_data = np.mean(rmse_[25,:,:,:],axis=-1)
#plot_data = rmse_[25,5,:,:].T

plt.figure(figsize=(4,4))
plt.plot(plot_data);
plt.xlabel('# Future Steps')
plt.ylabel('RMSE')
plt.title('Average over all #NN, different dim')

plot_data = rmse_[40,3,:,:].T
plt.figure(figsize=(4,4))
plt.plot(plot_data);
plt.xlabel('# of Neighbors')
plt.ylabel('RMSE')




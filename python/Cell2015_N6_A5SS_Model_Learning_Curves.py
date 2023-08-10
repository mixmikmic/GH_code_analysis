import pandas as pd
import numpy as np
import scipy
import scipy.sparse
import scipy.stats
import os
import scipy.io as sio
import dnatools
from MLR import MLR
get_ipython().magic('matplotlib inline')
from pylab import *

# Plotting Params:
rc('mathtext', default='regular')
fsize=14

resultsdir = '../results/N6_A5SS_Model_Learning_Curves/'
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
figdir = '../figures/N6_A5SS_Model_Learning_Curves/'
if not os.path.exists(figdir):
    os.makedirs(figdir)
    
#Choose if you want to actually save the plots:
SAVEFIGS = True

data = sio.loadmat('../data/Reads.mat')
# A5SS
A5SS_data = data['A5SS']
A5SS_reads = np.array(A5SS_data.sum(1)).flatten()
A5SS_data = np.array(A5SS_data.todense())
# Get minigenes with reads
A5SS_nn = find(A5SS_data.sum(axis=1))
A5SS_reads = A5SS_reads[A5SS_nn]
A5SS_data = A5SS_data[A5SS_nn]
A5SS_data = A5SS_data/A5SS_data.sum(axis=1)[:,newaxis]
A5SS_seqs = pd.read_csv('../data/A5SS_Seqs.csv',index_col=0).Seq[A5SS_nn]

R1 = A5SS_seqs.str.slice(7,32)
R2 = A5SS_seqs.str.slice(50,75)
X = {}
for mer_len in range(3,8):
    X_r1 = dnatools.make_mer_matrix_no_pos(A5SS_seqs.str.slice(7-mer_len+1,32+mer_len-1),mer_len)
    X_r2 = dnatools.make_mer_matrix_no_pos(A5SS_seqs.str.slice(50-mer_len+1,75+mer_len-1),mer_len)
    X[mer_len] = scipy.sparse.csr_matrix(scipy.sparse.hstack((X_r1,X_r2)))

Y = scipy.matrix(np.array((1-A5SS_data[:,0],A5SS_data[:,0])).T)

if False:
    inds = range(len(A5SS_seqs))
    shuffle(inds)
    train_set = inds[:int(len(inds)*0.9)]
    test_set = inds[int(len(inds)*0.9):]
else:
    train_set = np.loadtxt(resultsdir+'training_inds').astype(int)
    test_set = np.loadtxt(resultsdir+'test_inds').astype(int)

data_sizes = np.int64(10**arange(2,5.26,0.25))
lambdas = 10**arange(-1,-9,-1.)

models = {}
for L in lambdas:
    models[L] = {}
    print '-----------------Lambda:',L
    for mer_len in range(3,8):
        models[L][mer_len] = {}
        print '-----------------mer_len:',mer_len
        sys.stdout.flush()
        print '-----------------Data Size:',
        for data_size in data_sizes:
            print data_size,
            models[L][mer_len][data_size] = MLR(verbose=False)
            models[L][mer_len][data_size].fit(X[mer_len][train_set[:data_size]],
                                              Y[train_set[:data_size]],
                                              reg_type='L1',
                                              reg_lambda=L,
                                              maxit=5000)

model_preds = {}
for L in lambdas:
    model_preds[L] = {}
    print '-----------------Lambda:',L
    for mer_len in range(3,8):
        model_preds[L][mer_len] = {}
        print '-----------------mer_len:',mer_len
        for data_size in data_sizes:
            print data_size,
            model_preds[L][mer_len][data_size] = models[L][mer_len][data_size].predict(X[mer_len][test_set])
        print ''

R2s = {}
for L in lambdas:
    R2s[L] = {}
    for mer_len in range(3,8):
        R2s[L][mer_len] = {}
        print '-----------------mer_len:',mer_len
        for data_size in data_sizes:
            R2s[L][mer_len][data_size] = scipy.stats.pearsonr(model_preds[L][mer_len][data_size][:,1],np.array(Y)[test_set,1])[0]**2

R2s = pd.Panel(R2s)

R2s.to_pickle(resultsdir+'Subsampling_R2.panel')

#R2s = pd.read_pickle(resultsdir+'Subsampling_R2.panel')

R2_maxes = R2s.apply(max,axis=0).iloc[:14]
fig = figure(figsize=(9,4))
ax = fig.add_subplot(111)
markers = ['o','s','v','D','p']
c = 0
for col in R2_maxes.columns:
    R2_maxes[col].plot(label=str(col)+'-mer',marker=markers[c])
    c+=1
#R2s.apply(max,axis=0).iloc[:14].plot(ax=ax,marker='o')
ax.set_xscale('log')
leg = legend([str(i)+'-mers' for i in range(3,8)],bbox_to_anchor=(1.25,1),numpoints=1,fontsize=fsize)
leg.get_frame().set_alpha(0)
leg.set_title('Features')
ax.set_xlabel('Number of Training Points',fontsize=fsize)
ax.set_ylabel('$R^2$',fontsize=fsize)
setp(leg.get_title(),fontsize=fsize)
ax.tick_params(labelsize=fsize)
ax.set_xlim(90,200000)
ax.set_title('A5SS Library Learning Curve ($SD_1$)',fontsize=fsize)
if SAVEFIGS:
    figname = 'A5SS_Learning_Curves'
    fig.savefig(figdir+figname+'.png',bbox_inches='tight', dpi = 300)
    fig.savefig(figdir+figname+'.pdf',bbox_inches='tight', dpi = 300)
    fig.savefig(figdir+figname+'.eps',bbox_inches='tight', dpi = 300)


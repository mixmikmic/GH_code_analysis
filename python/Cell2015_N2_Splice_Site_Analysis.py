import pandas as pd
import numpy as np
import scipy
import scipy.sparse
import scipy.stats
import os
import scipy.io as sio
import dnatools
from plot_tools import simpleaxis, plot_splicing_histogram
import re
get_ipython().magic('matplotlib inline')
from pylab import *

# Plotting Params:
rc('mathtext', default='regular')
fsize=14

resultsdir = '../results/N2_Splice_Site_Analysis/'
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
figdir = '../figures/N2_Splice_Site_Analysis/'
if not os.path.exists(figdir):
    os.makedirs(figdir)
    
#Choose if you want to actually save the plots:
SAVEFIGS = True

data = sio.loadmat('../data/Reads.mat')

# A5SS
A5SS_data = data['A5SS']
A5SS_data = np.array(A5SS_data.todense())
# Get minigenes with reads
A5SS_nn = find(A5SS_data.sum(axis=1))
A5SS_data = A5SS_data[A5SS_nn]
A5SS_data = A5SS_data/A5SS_data.sum(axis=1)[:,newaxis]
A5SS_seqs = pd.read_csv('../data/A5SS_Seqs.csv',index_col=0).Seq[A5SS_nn]

# A3SS
A3SS_data = data['A3SS']
# Only look at SA_1 usage:
A3SS_data = np.array(A3SS_data[:,235].todense()).reshape(-1)/np.array(A3SS_data.sum(axis=1),dtype=np.float64).reshape(-1)
# Get minigenes with reads
A3SS_nn = find(pd.notnull(A3SS_data))
A3SS_data = A3SS_data[A3SS_nn]
A3SS_seqs = pd.read_csv('../data/A3SS_Seqs.csv',index_col=0).Seq[A3SS_nn]

base_freq_all = {}
for b in dnatools.bases:
    base_freq_all[b] = pd.Series(A5SS_seqs).str.slice(7,32).str.count(b).sum() + pd.Series(A5SS_seqs).str.slice(50,75).str.count(b).sum()
base_freq_all = pd.Series(base_freq_all)

sd_base_freqs = {}
for b in range(9):
    sd_base_freqs[b] = pd.Series(A5SS_seqs[find(A5SS_data[:,11]>0.1)]).groupby(pd.Series(A5SS_seqs[find(A5SS_data[:,11]>0.1)]).str.slice(7+b,7+1+b)).size()
    for i in range(12,32-6)+range(54,75-6):
        sd_base_freqs[b] += pd.Series(A5SS_seqs[find(A5SS_data[:,i]>0.1)]).groupby(pd.Series(A5SS_seqs[find(A5SS_data[:,i]>0.1)]).str.slice(i-4+b,i-3+b)).size()
    print b,
sd_base_freqs = pd.DataFrame(sd_base_freqs)

sd_to_random_ratio = (sd_base_freqs.fillna(0).T/base_freq_all).T

sd_to_random_ratio

normalized_base_ratio = (sd_to_random_ratio/sd_to_random_ratio.T.sum(axis=1))

normalized_base_ratio

base_prob_dict = normalized_base_ratio.to_dict()
sampled_bases = {}
for pos in range(9):
    sampled_bases[pos] = []
    for b in 'ACGT':
        sampled_bases[pos] += [b]*int(round(base_prob_dict[pos][b]*1000000))
    sampled_bases[pos] = np.array(sampled_bases[pos])
def make_sd():
    sd = ''
    for i in range(9):
        sd += np.random.choice(sampled_bases[i])
    return sd

f = open(resultsdir+'Generated_SD.txt','w')
for i in range(10000):
    f.write(make_sd()+'\n')
f.close()

weblogo=matplotlib.image.imread(figdir+'Library_SD_Normalized_Frequencies.png')
imshow(weblogo)

#Get each randomized region
r1 = pd.Series(A3SS_seqs).str.slice(0,25)
r2 = pd.Series(A3SS_seqs).str.slice(-25)
Y = data['A3SS'][A3SS_nn]

base_freq_all = {}
for b in dnatools.bases:
    base_freq_all[b] = r1.str.count(b).sum() + r2.str.count(b).sum()
base_freq_all = pd.Series(base_freq_all)

#Find all new SA in the two randomized regions . Then calculate the frequency of each base at each position
sa_base_freqs = {}
for p in range(15,26):
    inds = find(np.array(Y[:,189+24+25+p].todense())>0)
    for b in range(0,min(p+5,25)):
        try:
            sa_base_freqs[b-p] = sa_base_freqs[b-p].add(r2[inds].groupby(r2[inds].str.slice(b,b+1)).size(),fill_value=0)
        except:
            sa_base_freqs[b-p] = r2[inds].groupby(r2[inds].str.slice(b,b+1)).size()
for p in range(15,26):
    inds = find(np.array(Y[:,189+p].todense())>0)
    for b in range(0,min(p+5,25)):
        try:
            sa_base_freqs[b-p] = sa_base_freqs[b-p].add(r2[inds].groupby(r1[inds].str.slice(b,b+1)).size(),fill_value=0)
        except:
            sa_base_freqs[b-p] = r2[inds].groupby(r1[inds].str.slice(b,b+1)).size()
sa_base_freqs = pd.DataFrame(sa_base_freqs)

sa_to_random_ratio = (sa_base_freqs.fillna(0).T/base_freq_all).T
normalized_base_ratio = (sa_to_random_ratio/sa_to_random_ratio.T.sum(axis=1))

base_prob_dict = normalized_base_ratio.to_dict()
sampled_bases = {}
for pos in range(-25,5):
    sampled_bases[pos] = []
    for b in 'ACGT':
        sampled_bases[pos] += [b]*int(round(base_prob_dict[pos][b]*1000000))
    sampled_bases[pos] = np.array(sampled_bases[pos])
def make_sa():
    sa = ''
    for i in range(-25,5):
        sa += np.random.choice(sampled_bases[i])
    return sa

f = open(resultsdir+'Generated_SA.txt','w')
for i in range(10000):
    f.write(make_sa()+'\n')
f.close()

weblogo=matplotlib.image.imread(figdir+'Library_SA_Normalized_Frequencies.png')
imshow(weblogo)


fig = figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.tick_params(labelsize=fsize)
simpleaxis(ax)
reads = log10(A5SS_data.mean(axis=0))
reads[0:10]=nan
reads[29:53]=nan
reads[72:] = nan
axis([0,80,-3.5,-1.9])
ax.set_yticks(arange(-3.5,-1.99,0.5))
ax.set_yticklabels(['$10^{-3.5}$','$10^{-3}$','$10^{-2.5}$','$10^{-2}$'],ha='right')
ax.set_xticks(arange(0,100,25))
y = (np.concatenate((reads[10:29],reads[53:72])))
A = np.vstack([np.array(range(10,29)+range(53,72)), np.ones(38)]).T
m,c = np.linalg.lstsq(A,y)[0]
pred = arange(80)*m+c
plot(arange(80),pred,'gray',linewidth=2)
plot(reads,'o',markersize=4,color='b')

ax.set_ylabel('Mean $SD_{NEW}$ Usage',fontsize=fsize)
ax.set_xlabel('$SD_{NEW}$ Position\n(Relative to $SD_1$)',fontsize=fsize)

p,r = scipy.stats.pearsonr(np.concatenate((reads[10:29],reads[53:72])),np.concatenate((pred[10:29],pred[53:72])))
ax.text(79,-2.25,'$\it{p}$-pearson=%.2f' %p,ha='right',fontsize=fsize)
ax.text(79,-2.5,'$\it{P}$=%.1e' %r,ha='right',fontsize=fsize)

if SAVEFIGS:
  filename = 'Splice_Counts_vs_Distance'
  fig.savefig(figdir+filename+'.png', bbox_inches='tight',dpi=300)
  fig.savefig(figdir+filename+'.eps', bbox_inches='tight',dpi=200)
  fig.savefig(figdir+filename+'.pdf', bbox_inches='tight',dpi=200)


fig = figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.tick_params(labelsize=fsize)
simpleaxis(ax)

tb_start_x = 110
tb_start_y = -3.25

start = 10
end = 29
reads = log10(A5SS_data.mean(axis=0))
reads[0:start]=nan
reads[end:]=nan
axis([0,80,-3.5,-1.9])
ax.set_yticks(arange(-3.5,-1.99,0.5))
ax.set_yticklabels(['$10^{-3.5}$','$10^{-3}$','$10^{-2.5}$','$10^{-2}$'],ha='right')
ax.set_xticks(arange(0,100,25))
y = reads[start:end]
A = np.vstack([range(start,end), np.ones(19)]).T
m,c = np.linalg.lstsq(A,y)[0]
pred = arange(start,end)*m+c
plot(arange(start,end),pred,'red',linewidth=2)
plot(reads,'o',markersize=4,color='b')
r,p = scipy.stats.pearsonr(reads[start:end],pred)
ax.text(tb_start_x,tb_start_y,'Intercept' %r,ha='center',fontsize=fsize)
ax.text(tb_start_x,tb_start_y+0.25,'Slope' %r,ha='center',fontsize=fsize)
ax.text(tb_start_x,tb_start_y+0.5,'$\it{P}$' %r,ha='center',fontsize=fsize)
ax.text(tb_start_x,tb_start_y+0.75,'$\it{p}$-pearson' ,ha='center',fontsize=fsize)
ax.text(tb_start_x+40,tb_start_y+1,'Region 1 Fit',ha='center',fontsize=fsize)
ax.text(tb_start_x+40,tb_start_y,'%0.4f' %c,ha='center',fontsize=fsize)
ax.text(tb_start_x+40,tb_start_y+0.25,'%0.4f' %m,ha='center',fontsize=fsize)
ax.text(tb_start_x+40,tb_start_y+0.75,'%0.4f' %r,ha='center',fontsize=fsize)
ax.text(tb_start_x+40,tb_start_y+0.5,'%0.4f' %p,ha='center',fontsize=fsize)

start = 53
end = 72
reads = log10(A5SS_data.mean(axis=0))
reads[0:start]=nan
reads[end:]=nan
axis([0,80,-3.5,-1.9])
ax.set_yticks(arange(-3.5,-1.99,0.5))
ax.set_yticklabels(['$10^{-3.5}$','$10^{-3}$','$10^{-2.5}$','$10^{-2}$'],ha='right')
ax.set_xticks(arange(0,100,25))
y = reads[start:end]
A = np.vstack([range(start,end), np.ones(19)]).T
m,c = np.linalg.lstsq(A,y)[0]
pred = arange(start,end)*m+c
plot(arange(start,end),pred,'red',linewidth=2)
plot(reads,'o',markersize=4,color='b')
r,p = scipy.stats.pearsonr(reads[start:end],pred)
ax.text(tb_start_x+80,tb_start_y+1,'Region 2 Fit',ha='center',fontsize=fsize)
ax.text(tb_start_x+80,tb_start_y,'%0.4f' %c,ha='center',fontsize=fsize)
ax.text(tb_start_x+80,tb_start_y+0.25,'%0.4f' %m,ha='center',fontsize=fsize)
ax.text(tb_start_x+80,tb_start_y+0.75,'%0.4f' %r,ha='center',fontsize=fsize)
ax.text(tb_start_x+80,tb_start_y+0.5,'%0.4f' %p,ha='center',fontsize=fsize)

reads = log10(A5SS_data.mean(axis=0))
reads[0:10]=nan
reads[29:53]=nan
reads[72:] = nan
axis([0,80,-3.5,-1.9])
ax.set_yticks(arange(-3.5,-1.99,0.5))
ax.set_yticklabels(['$10^{-3.5}$','$10^{-3}$','$10^{-2.5}$','$10^{-2}$'],ha='right')
ax.set_xticks(arange(0,100,25))
y = (np.concatenate((reads[10:29],reads[53:72])))
A = np.vstack([np.array(range(10,29)+range(53,72)), np.ones(38)]).T
m,c = np.linalg.lstsq(A,y)[0]
pred = arange(80)*m+c
plot(arange(80),pred,'gray',linewidth=2)
plot(reads,'o',markersize=4,color='b')
r,p = scipy.stats.pearsonr(np.concatenate((reads[10:29],reads[53:72])),
                         np.concatenate((pred[10:29],pred[53:72])))

ax.text(tb_start_x+120,tb_start_y+1,'Combined Fit',ha='center',fontsize=fsize)
ax.text(tb_start_x+120,tb_start_y,'%0.4f' %c,ha='center',fontsize=fsize)
ax.text(tb_start_x+120,tb_start_y+0.25,'%0.4f' %m,ha='center',fontsize=fsize)
ax.text(tb_start_x+120,tb_start_y+0.75,'%0.4f' %r,ha='center',fontsize=fsize)
ax.text(tb_start_x+120,tb_start_y+0.5,'%1.2e' %p,ha='center',fontsize=fsize)

ax.set_ylabel('Mean $SD_{NEW}$ Usage',fontsize=fsize)
ax.set_xlabel('$SD_{NEW}$ Position\n(Relative to $SD_1$)',fontsize=fsize)

if SAVEFIGS:
  filename = 'Splice_Counts_vs_Distance_Each_Region'
  fig.savefig(figdir+filename+'.png', bbox_inches='tight',dpi=300)
  fig.savefig(figdir+filename+'.eps', bbox_inches='tight',dpi=200)
  fig.savefig(figdir+filename+'.pdf', bbox_inches='tight',dpi=200)

SD_new_reads_R1 = np.array(data['A5SS'].sum(axis=0)).flatten()[10:29].sum()
SD_new_reads_R2 = np.array(data['A5SS'].sum(axis=0)).flatten()[53:72].sum()
print SD_new_reads_R1,SD_new_reads_R2,SD_new_reads_R1/SD_new_reads_R2

branch_point_consensus = re.compile('[CT]T[AG]A[CT]')

branch_points_by_position = {}
for i in range(21):
    branch_points_by_position[i] = r1.str.slice(0+i,5+i).str.contains(branch_point_consensus).sum()
branch_points_by_position

def regex_pos(reg,cur_str):
    # Get the most 3' location of the branch point
    # Return -1 if not found
    pos = -1
    for m in reg.finditer(cur_str):
        pos = m.start()
    return pos

# Group SA_0 by BP position and get means, num introns
grouped = pd.Series(A3SS_data).groupby(r1.apply(lambda x:regex_pos(branch_point_consensus,x)).values)
bp_pos_data = grouped.aggregate({'Mean':mean,'Size':lambda x:pd.notnull(x).sum()})

fsize=18

fig = figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.plot([-1,22.5],[bp_pos_data.Mean[-1]]*2,linewidth=2,color='r',label='No Consensus BP')
bp_pos_data.Mean[1:].plot(ax=ax,marker='o',linewidth=2,label='Consensus BP')
axis([0,22.5,0,0.2])
ax.set_xticks(range(-1,22,5));
ax.set_xticklabels(range(-21-18-1,-18,5));
ax.tick_params(labelsize=fsize)
ax.set_xlabel('Branchpoint Position\n(Relative to $SA_1$)',fontsize=fsize)
ax.set_ylabel('$SA_1$ Usage (Fraction)',fontsize=fsize)
#leg = legend(['No BP'],numpoints=1,bbox_to_anchor=(0.8,1),fontsize=fsize)
#leg.get_frame().set_alpha(0)
#ax.text(1,0.01,'No Consenseus BP',fontsize=fsize)
ax.grid('off')
simpleaxis(ax)

if SAVEFIGS:
    filename = 'Branch_point_position'
    fig.savefig(figdir+filename+'.png', bbox_inches='tight',dpi=300)
    fig.savefig(figdir+filename+'.eps', bbox_inches='tight',dpi=200)
    fig.savefig(figdir+filename+'.pdf', bbox_inches='tight',dpi=200)

def regex_pos_ag(reg,cur_str):
    # Get the most 3' location of the branch point
    # Return -1 if not found
    pos = -1
    for m in reg.finditer(cur_str):
        pos = m.start()
    AG_found = cur_str[pos:].find('AG')>=0
    return AG_found

AG_inserted = r1.apply(lambda x:regex_pos_ag(branch_point_consensus,x)).values
grouped = pd.Series(A3SS_data[AG_inserted]).groupby(r1[AG_inserted].apply(lambda x:regex_pos(branch_point_consensus,x)).values)
bp_pos_data_AG = grouped.aggregate({'Mean':mean,'Size':lambda x:pd.notnull(x).sum()})
grouped = pd.Series(A3SS_data[~AG_inserted]).groupby(r1[~AG_inserted].apply(lambda x:regex_pos(branch_point_consensus,x)).values)
bp_pos_data_noAG = grouped.aggregate({'Mean':mean,'Size':lambda x:pd.notnull(x).sum()})

fig = figure(figsize=(4,4))
ax = fig.add_subplot(111)
bp_pos_data_AG.Mean.plot(ax=ax,marker='^',linewidth=2,label='AG Inserted Between BP and SA',color='g')
bp_pos_data_noAG.Mean[1:].plot(ax=ax,marker='o',linewidth=2,label='No AG Inserted Between BP and SA',color='b')
ax.plot([-1,22.5],[bp_pos_data.Mean[-1]]*2,linewidth=2,color='r',label='No Consensus BP')
axis([0,22.5,0,0.2])
ax.set_xticks(range(-1,22,5));
ax.set_xticklabels(range(-21-18-1,-18,5));
ax.tick_params(labelsize=fsize)
ax.set_xlabel('Branchpoint Position\n(Relative to $SA_1$)',fontsize=fsize)
ax.set_ylabel('$SA_1$ Usage (Fraction)',fontsize=fsize)
leg = legend(numpoints=1,bbox_to_anchor=(2.78,1),fontsize=fsize)
leg.get_frame().set_alpha(0)
ax.grid('off')
if SAVEFIGS:
    filename = 'Branch_point_position_AG_exclusion'
    fig.savefig(figdir+filename+'.png', bbox_inches='tight',dpi=300)
    fig.savefig(figdir+filename+'.eps', bbox_inches='tight',dpi=200)
    fig.savefig(figdir+filename+'.pdf', bbox_inches='tight',dpi=200)


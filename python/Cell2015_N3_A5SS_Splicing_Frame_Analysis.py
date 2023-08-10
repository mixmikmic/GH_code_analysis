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
fsize=14
get_ipython().magic('matplotlib inline')
from pylab import *

# Plotting Params:
rc('mathtext', default='regular')
fsize=14

resultsdir = '../results/N3_A5SS_Splicing_Frame_Analysis/'
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
figdir = '../figures/N3_A5SS_Splicing_Frame_Analysis/'
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

def in_frame_stop(seq,sd):
    """ Return the location of first stop codon before position sd
    If there is no stop codon return -1"""
    full_seq = seq
    stop_codon = -1
    for i in range(2,sd,3):
        if(full_seq[i:i+3] in ['TAA','TAG','TGA']):
            stop_codon=i
            break
    return stop_codon

stop_codon_series = []
sd_main = A5SS_data.argmax(axis=1)
for i in range(len(A5SS_seqs)):
    stop_codon_series.append(in_frame_stop(A5SS_seqs.iloc[i],sd_main[i]))
    if (i%10000)==0:
        print i,
stop_codon_series = pd.Series(stop_codon_series)

stop_codon_series[stop_codon_series==-1]=nan

df = pd.DataFrame({'Reads':A5SS_reads,
                   'Stop_Codon_Pos':stop_codon_series,
                   'SD_Main':sd_main})
df['Stop_Codon_SD_Dist'] = df.SD_Main.values-df.Stop_Codon_Pos.values

fig = figure(figsize=(8,3))

no_stop = df[pd.isnull(df.Stop_Codon_Pos)].groupby('SD_Main').Reads.aggregate({'Median':median,
                                                                        'Size':size})
ax = fig.add_subplot(111)
# Only plot SD positions with more than 10 minigenes using it as a majority.
no_stop[no_stop.Size>10].Median[0:-1:3].plot(marker='o',label='Frame 0')
no_stop[no_stop.Size>10].Median[1:-1:3].plot(marker='o',label='Frame 1')
no_stop[no_stop.Size>10].Median[2:-1:3].plot(marker='o',label='Frame 2')
ax.tick_params(labelsize=fsize)
ax.set_xlabel('Position of Majority SD',fontsize=fsize)
ax.set_ylabel('Median Reads Per Minigene',fontsize=fsize)
ax.set_title('No In-Frame Stop Codon',fontsize=fsize)
ax.axis([0,80,20,65])
leg = ax.legend(bbox_to_anchor=(1.3,1),numpoints=1,title='Spliced Frame:',fontsize=fsize)
setp(leg.get_title(),fontsize=fsize)
leg.get_frame().set_alpha(0)
if SAVEFIGS:
    figname = 'Reads_by_SD_frame_nostop'
    fig.savefig(figdir+figname+'.png',bbox_inches='tight', dpi = 300)
    fig.savefig(figdir+figname+'.pdf',bbox_inches='tight', dpi = 300)
    fig.savefig(figdir+figname+'.eps',bbox_inches='tight', dpi = 300)

with_stop = df[(df.Stop_Codon_Pos>=7) & (df.Stop_Codon_Pos<20)].groupby('SD_Main').Reads.aggregate({'Median':median,
                                                                        'Size':size})
fig = figure(figsize=(8,3))
ax = fig.add_subplot(111)
with_stop[with_stop.Size>10].Median[0:-1:3].plot(marker='o',label='Frame 0')
with_stop[with_stop.Size>10].Median[1:-1:3].plot(marker='o',label='Frame 1')
with_stop[with_stop.Size>10].Median[2:-1:3].plot(marker='o',label='Frame 2')
ax.tick_params(labelsize=fsize)
ax.set_xlabel('Position of Majority SD',fontsize=fsize)
ax.set_ylabel('Median Reads Per Minigene',fontsize=fsize)
ax.axis([0,80,20,65])
ax.arrow(54, 57, 26-2, 0, head_width=2, head_length=2,linewidth=2, fc='k', ec='k')
ax.text(60,58,'NMD',fontsize=fsize)
ax.arrow(9, 35, 12, 0, head_width=2, head_length=0,linewidth=2, fc='r', ec='r')
ax.arrow(9+12, 35, -12, 0, head_width=2, head_length=0,linewidth=2, fc='r', ec='r')
ax.text(12,34,'Stop\nCodons',fontsize=fsize,va='top')

ax.set_title('In-Frame Stop Codon 9-18nt After $SD_1$',fontsize=fsize)
leg = ax.legend(bbox_to_anchor=(1.3,1),numpoints=1,title='Spliced Frame:',fontsize=fsize)
setp(leg.get_title(),fontsize=fsize)
leg.get_frame().set_alpha(0)
if True:
    figname = 'Reads_by_SD_frame_withstop'
    fig.savefig(figdir+figname+'.png',bbox_inches='tight', dpi = 300)
    fig.savefig(figdir+figname+'.pdf',bbox_inches='tight', dpi = 300)
    fig.savefig(figdir+figname+'.eps',bbox_inches='tight', dpi = 300)


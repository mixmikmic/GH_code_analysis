from ephys import viz, core, clust, events, spiketrains
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import neuraltda.topology2 as tp2
import glob
import os
from importlib import reload
get_ipython().magic('matplotlib inline')

bp = '/mnt/cube/btheilma/emily/B604/klusta/phy103116/Pen01_Lft_AP2300_ML1250__Site15_Z2780__B604_cat_P01_S15_1/'
bp = '/home/brad/emily/P01S13/'

viz.plot_all_clusters(bp, )

reload(viz)
clusters = core.load_clusters(bp)
fs = core.load_fs(bp)
cluster_group = ['Good', 'MUA']
clulist = clusters[clusters.quality.isin(cluster_group)]['cluster'].unique()
plt.figure()
for clu in clulist:
    viz.plot_spike_shape(bp, clu)

spwidths = []
for clu in clulist:
    spwidths.append(clust.get_width(bp, clu))
    
plt.plot(len(spwidths)*[1], spwidths, '.')
plt.plot([0.94, 1.06], 2*[0.000230], '--')

plt.hist(spwidths, bins=200)

clulist = clusters['cluster'].unique()
wide = []
narrow = []
thresh = 0.000540
for clu in clulist:
    sw = clust.get_width(bp, clu)
    if sw >= thresh:
        wide.append(clu)
    else:
        narrow.append(clu)

cluster_group = ['Good', 'MUA']
c2 = clusters[clusters.quality.isin(cluster_group)]['cluster'].unique()

clusters

bp = '/mnt/cube/btheilma/emily/B604/klusta/phy103116/Pen01_Lft_AP2300_ML1250__Site15_Z2780__B604_cat_P01_S15_1/'
blockPath = bp
spikes = core.load_spikes(bp)
clusters = core.load_clusters(bp)
fs = core.load_fs(bp)
trials = events.load_trials(bp)

trials

import tqdm
good_mua_clusters = clusters[clusters['quality'].isin(['Good', 'MUA'])]
good_mua_clusters_list = good_mua_clusters['cluster'].values


trial_store = []

for cluster in good_mua_clusters_list:
    print(cluster)

    
    
    act_store = []
    for trial in tqdm.tqdm(trials.iterrows()):
        rec = trial[1]['recording']
        
        stim_start = trial[1]['time_samples']
        stim_end = trial[1]['stimulus_end']
        prestim_spiketrain = spiketrains.get_spiketrain(rec, stim_start, cluster, spikes, [-2, 0], fs)
        stim_spiketrain = spiketrains.get_spiketrain(rec, stim_start, cluster, spikes, [0, float(stim_end)/fs], fs)
        prestim_inds = np.nonzero(prestim_spiketrain)[0]
        prestim_isis = prestim_inds[1:] - prestim_inds[:-1]
        stim_inds = np.nonzero(stim_spiketrain)[0]
        stim_isis = stim_inds[1:] - stim_inds[:-1]
        prestim_mean = np.mean(prestim_isis)
        stim_mean = np.mean(stim_isis)
        activity = (stim_mean - prestim_mean) / prestim_mean
        act_store.append(activity)
    trial_store.append(act_store)
    
trial_store = np.array(trial_store)

import pickle

with open('/home/brad/auditory.pkl', 'wb') as f:
    pickle.dump(trial_store, f)

trial_mean = np.nanmean(trial_store, axis=1)
trial_mean

window = [-2, 0]
samps = stim_start
bds = [w*fs+samps for w in window]
clu = cluster

window_mask = (
        (spikes['time_samples']>bds[0])
        & (spikes['time_samples']<=bds[1])
        )
    
perievent_spikes = spikes[window_mask]
    
mask = (
        (perievent_spikes['recording']==rec)
        & (perievent_spikes['cluster']==clu)
        )
t = (perievent_spikes['time_samples'][mask].values.astype(np.float_) - samps) / fs

t1 = perievent_spikes['recording']==rec
t2 = (perievent_spikes['cluster']==clu)

blockPath = '/home/brad/emily/P01S15/'
winSize = 10.0 #ms
thresh = 13.0
povers = 0.5
cluster_group = ['Good', 'MUA']
widenarrow_threshold = 0.000230 # sw threshold in seconds
stimsegmentInfo = [0, 0] # Sample/Distractor Period
prestimsegmentInfo = [-2000, 0]
spikes = core.load_spikes(blockPath)
trials = events.load_trials(blockPath)
fs = core.load_fs(blockPath)
povers = 0.0

# Get wide/narrow clusters
clusters = core.load_clusters(blockPath)
clusters_list = clusters[clusters.quality.isin(cluster_group)]['cluster'].unique()

# Bin and compute SCG
bfdict = tp2.do_dag_bin_lazy(blockPath, spikes, trials, clusters, fs, winSize,
                                    stimsegmentInfo, cluster_group=['Good', 'MUA'],
                                    dt_overlap=povers*winSize, comment='stimulus_period')
bdf_stim = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]

bfdict = tp2.do_dag_bin_lazy(blockPath, spikes, trials, clusters, fs, winSize,
                                    prestimsegmentInfo, cluster_group=['Good', 'MUA'],
                                    dt_overlap=povers*winSize, comment='prestimulus_period')
bdf_prestim = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]

import h5py as h5

thresh = 1.0
stimdiffs = {}
stimabv = {}
with h5.File(bdf_stim) as stimfile:
    stims = stimfile.keys()
    with h5.File(bdf_prestim) as prestimfile:
        for stim in stimfile.keys():
            print(stim)
            stim_data = stimfile[stim]
            stim_acty = np.mean(stim_data['pop_tens'], axis=1)
            
            prestim_data = prestimfile[stim]
            prestim_acty = np.mean(prestim_data['pop_tens'], axis = 1)
            
            pdiff = np.divide(stim_acty - prestim_acty, prestim_acty)*100.0
            #pdiff = stim_acty - prestim_acty
            abvthresh = 1.0*(stim_acty > thresh*prestim_acty)
            stimdiffs[stim] = pdiff
            stimabv[stim] = abvthresh

celldata = []

for stim in stims:
    avgactydiff = np.nanmean(stimdiffs[stim], axis=1)
    celldata.append(avgactydiff)
    avgactyabv = np.mean(stimabv[stim], axis=1)
    print(stim, avgactydiff)
    print(stim, avgactyabv)

celldata = np.array(celldata)

for cell in range(74):
    plt.figure()
    plt.plot(celldata[:, cell])
    plt.xticks(range(10), stims, rotation='vertical')
    plt.title('Cell {}'.format(cell))
    plt.ylim([-30, 100])
    plt.savefig('/home/brad/cell_{}_auditory.png'.format(cell))




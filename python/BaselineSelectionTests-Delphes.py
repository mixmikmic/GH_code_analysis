from __future__ import print_function
import functools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ROOT;
import sys
sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')
import root_numpy as rnp
get_ipython().magic('matplotlib notebook')

#filename = '/Users/sfarrell/Atlas/xaod/mc15_13TeV.361023.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3W.merge.DAOD_EXOT3.e3668_s2576_s2132_r7728_r7676_p2613/DAOD_EXOT3.08204445._000002.pool.root.1'
#signal
#filename = '/global/projecta/projectdirs/atlas/atlaslocalgroupdisk/rucio/mc15_13TeV/76/71/DAOD_EXOT3.08629754._000001.pool.root.1'
#bg
#filename = '/global/projecta/projectdirs/atlas/atlaslocalgroupdisk/rucio/mc15_13TeV/ca/31/DAOD_EXOT3.08910637._000016.pool.root.1'
#Delphes
#filename = '/global/project/projectdirs/mpccc/wbhimji/Delphes-3.4.0/RPVSusy_1400_850_100_XXXX-10k-9-1-28.root'
filename = '/global/project/projectdirs/mpccc/wbhimji/Delphes-3.4.0/RPV10_1400_850-10k-1-1-1.root'
#Delphes Bg
#filename = '/global/project/projectdirs/mpccc/wbhimji/Delphes-3.3.2/Data/QCDBkg_200_2500-30k-2-22.root'
#bg_files = filename

#bg_files = [line.rstrip() for line in open('/global/project/projectdirs/das/wbhimji/RPVSusyJetLearn/atlas_dl/config/mc15_13TeV.361004.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4.merge.DAOD_EXOT3.e3569_s2576_s2132_r7772_r7676_p2688-FileList.txt')]
#bg_files = [line.rstrip() for line in open('/project/projectdirs/das/wbhimji/RPVSusyJetLearn//atlas_dl/config/Delphes_QCDBkg_200_2500.txt')]
bg_files = [line.rstrip() for line in open('/project/projectdirs/dasrepo/atlas_rpv_susy/DelphesData/Delphes_QCDBkg_800_2500.txt')]
#sig_files = [line.rstrip() for line in open('/global/project/projectdirs/das/wbhimji/RPVSusyJetLearn/atlas_dl/config/mc15_13TeV.403568.MadGraphPythia8EvtGen_A14NNPDF23LO_GG_RPV10_1400_850.merge.DAOD_EXOT3.e5079_a766_a821_r7676_p2669-FileList.txt')]

# Branch names to read in and rename for convenience
#branchMap = {
#    'CaloCalTopoClustersAuxDyn.calEta' : 'ClusEta',
#    'CaloCalTopoClustersAuxDyn.calPhi' : 'ClusPhi',
#    'CaloCalTopoClustersAuxDyn.calE' : 'ClusE',
#    'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.pt' : 'FatJetPt',
#    'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.eta' : 'FatJetEta',
#    'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.phi' : 'FatJetPhi',
#    'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.m' : 'FatJetM',
#}

branchMap = {
    'Tower.Eta' : 'ClusEta',
    'Tower.Phi' : 'ClusPhi',
    'Tower.E' : 'ClusE',
    'FatJet.PT' : 'FatJetPt',
    'FatJet.Eta' : 'FatJetEta',
    'FatJet.Phi' : 'FatJetPhi',
    'FatJet.Mass' : 'FatJetM',
}

treename = 'Delphes'

entries = rnp.root2array(filename, treename=treename,
                         branches=branchMap.keys(),warn_missing_tree=True,
                          start=0, stop=1000000)
entries.dtype.names = branchMap.values()
print('Entries:', entries.size)
entries.dtype

# Multiple ways to dump variables for a specific event.
# I'm actually surprised these both work.
print(entries[0]['FatJetPt'])
print(entries['FatJetPt'][0]*1000)
print(entries['FatJetM'][0])

entries['FatJetPt']=entries['FatJetPt']*1000
entries['FatJetM']=entries['FatJetM']*1000

bgdf = pd.DataFrame.from_records(entries)

# Perform object selections on one event
event = entries[3]
event['FatJetPt'] > 300000

# Select fatjets with pt > 200 GeV for all events in one go
f = np.vectorize(lambda jetPts: jetPts > 200000, otypes=[np.ndarray])
selectedJets = f(entries['FatJetPt'])
print(selectedJets)

# Select events with at least 2 selected jets
countSelected = np.vectorize(sum)
numJets = countSelected(selectedJets)
selectedEvents = numJets >= 2
print(numJets)
print(selectedEvents)

sys.path.append('/project/projectdirs/das/wbhimji/RPVSusyJetLearn/atlas_dl_submitter/atlas_dl/scripts/')
from physics_selections import (select_fatjets, is_baseline_event,
                                sum_fatjet_mass, is_signal_region_event)

vec_select_fatjets = np.vectorize(select_fatjets, otypes=[np.ndarray])
vec_select_baseline_events = np.vectorize(is_baseline_event)
selectedFatJets = vec_select_fatjets(entries['FatJetPt'], entries['FatJetEta'])
baselineEvents = vec_select_baseline_events(entries['FatJetPt'], selectedFatJets)
print('Baseline selected events: %d / %d' % (np.sum(baselineEvents), entries.size))

# Calculate the summed jet mass for all events
summedMasses = np.vectorize(sum_fatjet_mass)(entries['FatJetM'], selectedFatJets)
print(summedMasses[baselineEvents])

vec_select_sr_events = np.vectorize(is_signal_region_event)
signalEvents = vec_select_sr_events(summedMasses, entries['FatJetPt'], entries['FatJetEta'],
                                    selectedFatJets, baselineEvents)
signalEntries = entries[signalEvents]
numSignalEvents = np.sum(signalEvents)
print('Signal events: %d / %d' % (numSignalEvents, entries.size))

def get_hist2d(event):
    """Convert event into the calo-cluster image"""
    return np.histogram2d(event['ClusEta'], event['ClusPhi'],
                          bins=(50, 50), weights=event['ClusE'],
                          range=[[-2.5, 2.5], [-3.15, 3.15]])[0]

def plot_calo_image(h2d):
    """Plot a calo-image on the current axes"""
    plt.imshow(np.log10(h2d).T, #extent=[-2.,2.,-3.14, 3.14],
               extent=[-2.5, 2.5, -3.15, 3.15],
               interpolation='none', aspect='auto', origin='low')
    plt.colorbar(label='Cluster energy [Log(MeV)]')
    plt.xlabel('eta')
    plt.ylabel('phi')

def plot_jets(jetEtas, jetPhis, jetRadius=1):
    """Plot jet circles on the current axes"""
    for eta, phi in zip(jetEtas, jetPhis):
        circle = plt.Circle((eta, phi), radius=jetRadius, facecolor='none')
        plt.gcf().gca().add_artist(circle)

# Pick out a sample of signal region events.
# The indexing is now starting to get very confusing.
numSample = 4
sampleIdxs = np.random.choice(np.arange(numSignalEvents), numSample, replace=False)
sampleEntries = signalEntries[sampleIdxs]
sampleFatJets = selectedFatJets[signalEvents][sampleIdxs] # are we lost yet?
assert(sampleEntries.size == sampleFatJets.size)

# Get the quantities to plot
hists = [get_hist2d(ev) for ev in sampleEntries]
jetEtas = [etas[jets] for (etas, jets) in zip(sampleEntries['FatJetEta'], sampleFatJets)]
jetPhis = [phis[jets] for (phis, jets) in zip(sampleEntries['FatJetPhi'], sampleFatJets)]

# Draw the calo images and draw the selected fat jets as circles
plt.figure(figsize=(12, 10))
plt.subplot(221)
plot_calo_image(hists[0])
plot_jets(jetEtas[0], jetPhis[0])

plt.subplot(222)
plot_calo_image(hists[1])
plot_jets(jetEtas[1], jetPhis[1])

plt.subplot(223)
plot_calo_image(hists[2])
plot_jets(jetEtas[2], jetPhis[2])

plt.subplot(224)
plot_calo_image(hists[3])
plot_jets(jetEtas[3], jetPhis[3])

num_jets = np.vectorize(lambda x: np.sum(x))

from physics_selections import fatjet_deta12

def fatJetDelta(entriesEta,selectedFatJets):
    if sum(selectedFatJets) < 4:
        return -99
    else:
        return fatjet_deta12(entriesEta,selectedFatJets)

fatjetDelta = np.vectorize(fatJetDelta)(entries['FatJetEta'],selectedFatJets)

maxpts = np.vectorize(lambda x: np.max(x))

def fatJetSumMass(entriesMass,selectedFatJets):
    if sum(selectedFatJets) < 4:
        return -99
    else:
        return  sum_fatjet_mass(entriesMass,selectedFatJets)
vec_fatJetSumMass = np.vectorize(fatJetSumMass)(entries['FatJetM'],selectedFatJets)

get_ipython().magic('matplotlib inline')
plt.figure(figsize=(12, 10))
plt.subplot(221)
#np.histogram(jetEtas)
plt.hist(fatjetDelta[fatjetDelta>0],bins=50, normed=True)

plt.subplot(222)
plt.hist(summedMasses[baselineEvents],bins=50, normed=True)

plt.subplot(223)
plt.hist(vec_fatJetSumMass[vec_fatJetSumMass>0],bins=50, normed=True)
#plt.hist(num_jets(selectedFatJets))

plt.subplot(224)
plt.hist(maxpts(entries['FatJetPt']),bins=50, histtype='step')

#hist_handle = plt.figure()
#import pickle as pl
#pl.dump(hist_handle,file('Delphes-Hists.pickle','w'))

sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')
from rootpy.plotting import Hist
from root_numpy import fill_hist
from rootpy.io import root_open, DoesNotExist
f = root_open('DelphesHistos-PileUp-Mu20_subtracted-TopDecay-HighGran.root','RECREATE')

hDeltaEta = Hist(100, 0, 3.2, name='DeltaEta')
fill_hist(hDeltaEta,fatjetDelta[fatjetDelta>0])
hSumJetMass = Hist(100, 0, 2000, name='SumJetMass')
fill_hist(hSumJetMass,summedMasses[baselineEvents]/1000)
hSumJetMassSelected = Hist(100, 0, 2000, name='SumJetMassSelected')
fill_hist(hSumJetMassSelected,vec_fatJetSumMass[vec_fatJetSumMass>0]/1000)
hMaxJetPt = Hist(100, 0, 2000, name='MaxJetPt')
fill_hist(hMaxJetPt,maxpts(entries['FatJetPt'])/1000)
hNumJets = Hist(100, 0, 8, name='NumJets')
fill_hist(hNumJets,num_jets(selectedFatJets))

hDeltaEta.Write()
hSumJetMass.Write()
hMaxJetPt.Write()
hNumJets.Write()
hSumJetMassSelected.Write()
f.Close()

plot_calo_image(hists[3])
plot_jets(jetEtas[3], jetPhis[3])




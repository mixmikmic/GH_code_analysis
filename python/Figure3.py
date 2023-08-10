import matplotlib
get_ipython().magic('matplotlib inline')
import sys
from __future__ import division
from collections import defaultdict
from pylab import *
from itertools import izip
import colorbrewer as cb
import os
from scripts import utils as sutils
import cPickle as pickle
import colorbrewer as cb
from matplotlib import pyplot as plt
import pandas as pnd

PLOTPATH = 'plots/fig3/'

data_anno = pnd.read_csv('../data/fig3/ratios_anno.txt', sep='\t')
data_rna = pnd.read_csv('../data/fig3/ratios_rna.txt', sep='\t')

names_rna = ['Lizard', 'Chicken', 'Opossum', 'ensembl_mouse', 'ensembl_mouse_12tis8rep', 'Chimp', 'ensembl_human', 'refseq_mouse', 'refseq_human']
data_rna_sorted = data_rna.reindex([list(data_rna.species).index(a) for a in names_rna]).iloc[:-2, :]
data_rna.reindex([list(data_rna.species).index(a) for a in names_rna])

names_anno = ['Lizard', 'Chicken', 'Opossum', 'ensembl_Mouse', 'ensembl_Mouse', 'Chimp', 'ensembl_Human', 'refseq_Mouse', 'refseq_Human']
data_anno_sorted = data_anno.reindex([list(data_anno.species).index(a) for a in names_anno]).iloc[:-2, :]
data_anno.reindex([list(data_anno.species).index(a) for a in names_anno])

fig, ax1 = plt.subplots(figsize=[7, 4])

color_ns = '#9e9e9e' #'#%02x%02x%02x' % cb.Greys[6][1]
color_ns_rna = '#FF0000'#'#%02x%02x%02x' % cb.Greys[6][3]

names_anno = ['Lizard', 'Chicken', 'Opossum', 'Mouse', 'Mouse*', 'Chimp', 'Human']

anno_rna_ns   = data_rna_sorted.num_complex
anno_ns   = data_anno_sorted.num_complex

ind = np.arange(len(anno_ns))    # the x locations for the groups
width = .4       # the width of the bars: can also be len(x) sequence

p1 = ax1.bar(ind, anno_ns, width, color='#D1D1D1', edgecolor=color_ns, linewidth=1) # , edgecolor = "none"
p2 = ax1.bar(ind+width, anno_rna_ns,  width, color='#FF6666', edgecolor=color_ns_rna, linewidth=1)

ax1.set_ylabel('Number of Complex LSVs', size=16)
ax1.set_ylim(0, max(anno_rna_ns)+5000)
plt.title('LSVs complexity by species', size=20)
plt.xticks(ind+width, names_anno, size=12)
plt.legend( (p1[0], p2[0]), ('Only DB', 'DB + RNA'), loc=2)

# plt.savefig("%s/figure3A_top.pdf"% PLOTPATH, width=300, height=300, dpi=200)
# plt.clf()
plt.show()

fig, ax1 = plt.subplots(figsize=[7, 4])

color_perc = '#9e9e9e' #'#%02x%02x%02x' % cb.Greys[6][1]
color_perc_rna = '#ff0000'#'#%02x%02x%02x' % cb.Greys[6][3]

names_anno_rna = ['Lizard', 'Chicken', 'Opossum', 'Mouse', 'Mouse*', 'Chimp',  'Human']
anno_rna_perc = data_rna_sorted.num_complex/data_rna_sorted.num_lsvs*100
anno_perc = data_anno_sorted.num_complex/data_anno_sorted.num_lsvs*100

ind = np.arange(len(anno_rna_perc))    # the x locations for the groups
width = .4       # the width of the bars: can also be len(x) sequence

p1 = ax1.bar(ind, anno_perc, width, color='#D1D1D1', edgecolor=color_perc, linewidth=1)
p2 = ax1.bar(ind+width, anno_rna_perc,  width, color='#FF6666', edgecolor=color_perc_rna, linewidth=1)

ax1.set_ylabel('% of Complex LSVs', size=16)
ax1.set_ylim(0,100)
plt.title('LSVs complexity by species', size=20)
plt.xticks(ind+width, names_anno_rna, size=12)
plt.legend( (p1[0], p2[0]), ('Only DB', 'DB + RNA'), loc=2 )

# plt.savefig("%s/figure3A_bottom.pdf"% PLOTPATH, width=300, height=300, dpi=200)
# plt.clf()
plt.show()

fig, ax1 = plt.subplots(figsize=[7, 4])

color_refseq = '#E35A00' #'#%02x%02x%02x' % cb.Greys[6][1]
color_ensembl = '#005DB5'#'#%02x%02x%02x' % cb.Greys[6][3]

names = ['Mouse DB', 'Mouse RNA+DB', 'Human DB', 'Human DB+RNA']
data_ensembl = [data_anno[data_anno.species == 'ensembl_Mouse'].iloc[0],
              data_rna[data_rna.species == 'ensembl_mouse'].iloc[0],
              data_anno[data_anno.species == 'ensembl_Human'].iloc[0],
              data_rna[data_rna.species == 'ensembl_human'].iloc[0]
             ]
data_refseq = [
    data_anno[data_anno.species == 'refseq_Mouse'].iloc[0],
    data_rna[data_rna.species == 'refseq_mouse'].iloc[0],
    data_anno[data_anno.species == 'refseq_Human'].iloc[0],
    data_rna[data_rna.species == 'refseq_human'].iloc[0]
]

ns_ensembl = [aa['num_complex'] for aa in data_ensembl]
ns_refseq = [aa['num_complex'] for aa in data_refseq]

ind = np.arange(len(names))    # the x locations for the groups
width = .4       # the width of the bars: can also be len(x) sequence

p1 = ax1.bar(ind, ns_refseq, width, color='#FF8F33', edgecolor=color_refseq, linewidth=1)
p2 = ax1.bar(ind+width, ns_ensembl,  width, color='#3390E8', edgecolor=color_ensembl, linewidth=1)

ax1.set_ylabel('Number of complex LSVs', size=16)
ax1.set_ylim(0,max(anno_rna_ns)+5000)
plt.title('Effect of annotation DB \non # of complex LSVs ', size=20)
plt.xticks(ind+width, names, size=12)
plt.legend( (p1[0], p2[0]), ('Refseq', 'Ensembl'), loc=2 )

# plt.savefig("%s/figure3B_top.pdf"% PLOTPATH, width=300, height=300, dpi=200)
# plt.clf()
plt.show()

fig, ax1 = plt.subplots(figsize=[7, 4])

# color_refseq = '#%02x%02x%02x' % cb.Greys[6][1]
# color_ensembl = '#%02x%02x%02x' % cb.Greys[6][3]

names = ['Mouse DB', 'Mouse RNA+DB', 'Human DB', 'Human DB+RNA']
percs_ensembl = np.array(ns_ensembl)/np.array([aa['num_lsvs'] for aa in data_ensembl])
percs_refseq = np.array(ns_refseq)/np.array([aa['num_lsvs'] for aa in data_refseq])

ind = np.arange(len(names))    # the x locations for the groups
width = .4       # the width of the bars: can also be len(x) sequence

p1 = ax1.bar(ind, percs_refseq, width, color='#FF8F33', edgecolor=color_refseq, linewidth=1)
p2 = ax1.bar(ind+width, percs_ensembl,  width, color='#3390E8', edgecolor=color_ensembl, linewidth=1)

ax1.set_ylabel('Fraction of complex LSVs', size=16)
plt.title('Effect of annotation DB \non fraction of complex LSVs ', size=20)
plt.ylim([0,1])
plt.xticks(ind+width, names, size=12)
plt.legend( (p1[0], p2[0]), ('Refseq', 'Ensembl'), loc=2 )

# plt.savefig("%s/figure3B_bottom.pdf"% PLOTPATH, width=300, height=300, dpi=200)
# plt.clf()
plt.show()




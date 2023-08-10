import pandas as pd
import numpy as np
import scipy
import scipy.sparse
import scipy.stats
import os
import scipy.io as sio
import dnatools
from collections import Counter
get_ipython().magic('matplotlib inline')
from pylab import *
# Plotting Params:
rc('mathtext', default='regular')
fsize=14

resultsdir = '../results/N0_A3SS_Fastq_to_Splice_Reads/'
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
figdir = '../figures/N0_A3SS_Fastq_to_Splice_Reads/'
if not os.path.exists(figdir):
    os.makedirs(figdir)
    
#Choose if you want to actually save the plots:
SAVEFIGS = True

alt3SS_seq = '.........................GCTTGGATCTGATCTCAACAGGGT.........................'
alt3SS_tag = 'CATTACCTGC.........................'

f = {}
f[0] = open('../fastq/A3SS_dna_R1.fq','r')
f[1] = open('../fastq/A3SS_dna_R2.fq','r')
tags = Counter()
c = 0
p = 0
header = {}
seq = {}
strand = {}
quality ={}
tag_seqs = {}
d = 0
while True:
    for i in range(2):
        header[i] = f[i].readline()[:-1]
        seq[i] = f[i].readline()[:-1]
        strand[i] = f[i].readline()[:-1]
        quality[i] = f[i].readline()[:-1]

    cur_tag = dnatools.reverse_complement(seq[1])
    if(len(header[0])==0):
        break
        
    # Check passing reads and that the sequence after the random tag matches
    # the plasmid sequence.
    if (cur_tag[10:20]==alt3SS_tag[:10]):
        p += 1
        #Check that the non-randomized sequences match perfectly to the reference
        if(seq[0][25:25+24]==alt3SS_seq[25:-25]):
            d+=1
            try:
                tag_seqs[cur_tag]
            except:
                tag_seqs[cur_tag] = Counter()
            tag_seqs[cur_tag][seq[0]]+=1

    if(c%1000000)==0:
        print c,p,'|',
    c+=1
    
for i in range(2):
    f[i].close()

ks = tag_seqs.keys()
tag_map = {}
tag_map_counts = {}
c = 0
for k in ks:
    max_seq = max(tag_seqs[k]) # Get seq
    max_seq_counts = tag_seqs[k][max_seq]
    if(max_seq_counts>=2):
        tag_map[k] = max_seq
        tag_map_counts[k] = max_seq_counts
    if(c%100000)==0:
        print c,
    c+=1
seq_series = pd.Series(tag_map)
seq_counts = pd.Series(tag_map_counts)

seq_series = pd.Series(dict(zip(pd.Series(seq_series.index).str.slice(-20),seq_series.values )))

seq_series.name='Seq'
seq_series.index.name='Tag'
seq_series.to_csv('../data/A3SS_Seqs.csv',index_label='Tag',header=True)

tag2seq_dict = dict(zip(seq_series.index,arange(len(seq_series))))

alt3SS_full_seq = 'gtaagttatcaccttcgtggctacagagtttccttatttgtctctgttgccggcttatatggacaagcatatcacagccatttatcggagcgcctccgtacacgctattatcggacgcctcgcgagatcaatacgtataccagctgccctcgatacatgtcttggacggggtcggtgttgatatcgtatNNNNNNNNNNNNNNNNNNNNNNNNNGCTTGGATCTGATCTCAACAGGGTNNNNNNNNNNNNNNNNNNNNNNNNNatgattacacatatagacacgcgagcacccatcttttatagaatgggtagaacccgtcctaaggactcagattgagcatcgtttgcttctcgagtactacctggtacagatgtctcttcaaacaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagctaccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaaggactgatagtaaggcccattacctgcNNNNNNNNNNNNNNNNNNNNGCAGAACACAGCGGTTCGACCTGCGTGATATCTCGTATGCCGTCTTCTGCTTG'
alt3SS_full_seq = alt3SS_full_seq.upper()

c = 0
header = {}
seq = {}
strand = {}
quality ={}

tag_list = []
ss_list = []

f = {}
f[0] = open('../fastq/A3SS_rna_R1.fq','r')
f[1] = open('../fastq/A3SS_rna_R2.fq','r')

while True:
    for i in range(2):
        header[i] = f[i].readline()[:-1]
        seq[i] = f[i].readline()[:-1]
        strand[i] = f[i].readline()[:-1]
        quality[i] = f[i].readline()[:-1]
    if(len(header[i])==0):
        break
        #min_qual[i] = min(quality[i])
    tag = dnatools.reverse_complement(seq[1][:20])

    try:
        tag_ind = tag2seq_dict[tag]
    except:
        pass
    else:
        # Check if the end of the read 100-120 matches the second exon
        # of citrine. In case of mismatches, I check for matches to 3
        # different 20nt regions.
        s_start = alt3SS_full_seq.find(seq[0][100:120])-100
        if(s_start<-100):
            s_start = alt3SS_full_seq.find(seq[0][80:100])-80
            if(s_start<-80):
                s_start = alt3SS_full_seq.find(seq[0][60:80])-60
        if(s_start>=0):
            tag_list.append(tag_ind)
            ss_list.append(s_start)
    if(c%1000000)==0:
        print c,
    c+=1

for i in range(2):
    f[i].close()

splices = {'A3SS':scipy.sparse.csr_matrix((list(np.ones_like(ss_list))+[0],
                                           (tag_list+[len(seq_series)-1],ss_list+[565])),
                                          dtype=np.float64)}

sio.savemat('../data/A3SS_Reads.mat',splices)


import os
import sys
import csv
import gzip
import time
import math
import numpy
import random
import shutil
import operator
from scipy import stats
from textwrap import wrap
from collections import defaultdict

from tqdm import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.textpath as textpath
import seaborn as sns
import pandas as pd
get_ipython().magic('matplotlib inline')

colors = sns.color_palette("GnBu_d", 7)
colors.reverse()
sns.palplot(colors)

noise_analysis = dict()
data_files = [f for f in os.listdir('../data/noise_results/') if f.endswith('data.txt')]
data_files

for df in data_files:
    af = df.replace('data', 'annot')
    p_noise = float(df.split('_')[0])
        
    noisy_h2s = defaultdict(list)
    noisy_successes = dict()
    noisy_significant = dict()
    reader = csv.reader(open(os.path.join('../data/noise_results/', af)), delimiter='\t')
    reader.next()
    for pn, nfam, nsig, nsucc in reader:
        noisy_successes[int(nfam)] = int(nsucc)
        noisy_significant[int(nfam)] = int(nsig)
    
    reader = csv.reader(open(os.path.join('../data/noise_results/', df)), delimiter='\t')
    reader.next()
    for pn, nfam, h2 in reader:
        noisy_h2s[int(nfam)].append(float(h2))
    
    
#     for nfam in sorted(noisy_significant.keys()):
#         print "%10d %5d %5d %5d" % (nfam, noisy_significant[nfam], noisy_successes[nfam], len(noisy_h2s[nfam]))
    
    noise_analysis[p_noise] = (noisy_h2s, noisy_significant, noisy_successes)

for p_noise in sorted(noise_analysis.keys()):
    noisy_h2s, noisy_significant, noisy_successes = noise_analysis[p_noise]
    
    print "P_noise = ", p_noise
    print "%5s %5s %7s %7s %7s %7s %7s %7s" % ("NFam", "reps", "min", "2.5%", "median", "97.5%", "max", "posa")
    for key, values in sorted(noisy_h2s.items()):
        if float(noisy_successes[key]) == 0:
            continue
        posa = noisy_significant[key]/float(noisy_successes[key])
        print "%5d %5d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f" % (key, len(values), numpy.min(values), numpy.percentile(values, 2.5), numpy.median(values), numpy.percentile(values, 97.5), numpy.max(values), posa)

    sns.set_style('ticks')
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)

    fams, data = zip(*sorted(noisy_h2s.items()))
    #means = [numpy.mean(d) for d in data]
    #se95 = [1.96*numpy.std(d)/numpy.sqrt(len(d)) for d in data]
    #ci95 = zip(*[(numpy.mean(d)-numpy.percentile(d,2.5), numpy.percentile(d,97.5)-numpy.mean(d)) for d in data])
    #plt.errorbar(fams, means, ci95, fmt='ko', lw=1)
    plt.boxplot(data, labels = fams)
    plt.title(p_noise)
    plt.ylim(0,1.05)
    plt.ylabel("$h_2^o$ from EHR")
    plt.xlabel('Number of Families Sampled')
    
    plt.subplot(1,2,2)

    fsig = list()
    pfams = list()
    
    for f in fams:
        if float(noisy_successes[f]) == 0:
            continue
        fsig.append(noisy_significant[f]/float(noisy_successes[f]))
        pfams.append(f)

    plt.plot(pfams, fsig, 'ko-',lw=1)
    plt.plot([0,max(fams)],[0.9,0.9], 'r-',lw=1)
    plt.ylabel("POSA")
    plt.xlabel('Number of Families Sampled')
    
    plt.ylim(0,1.05)
    plt.xlim(0,max(fams)+10)
    #plt.xticks(fams)
    sns.despine(trim=True)

missing_analysis = dict()
data_files = [f for f in os.listdir('../data/missing_results/') if f.endswith('data.txt')]
data_files

for df in data_files:
    af = df.replace('data', 'annot')
    p_missing = float(df.split('_')[0])
        
    missing_h2s = defaultdict(list)
    missing_successes = dict()
    missing_significant = dict()
    reader = csv.reader(open(os.path.join('../data/missing_results/', af)), delimiter='\t')
    reader.next()
    for pn, nfam, nsig, nsucc in reader:
        missing_successes[int(nfam)] = int(nsucc)
        missing_significant[int(nfam)] = int(nsig)
    
    reader = csv.reader(open(os.path.join('../data/missing_results/', df)), delimiter='\t')
    reader.next()
    for pn, nfam, h2 in reader:
        missing_h2s[int(nfam)].append(float(h2))
    
    
    missing_analysis[p_missing] = (missing_h2s, missing_significant, missing_successes)

for p_missing in sorted(missing_analysis.keys()):
    missing_h2s, missing_significant, missing_successes = missing_analysis[p_missing]
    print "P_missing = ", p_missing
    print "%5s %5s %7s %7s %7s %7s %7s %7s" % ("NFam", "reps", "min", "2.5%", "median", "97.5%", "max", "posa")
    for key, values in sorted(missing_h2s.items()):
        if float(missing_successes[key]) == 0:
            continue
        posa = missing_significant[key]/float(missing_successes[key])
        print "%5d %5d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f" % (key, len(values), numpy.min(values), numpy.percentile(values, 2.5), numpy.median(values), numpy.percentile(values, 97.5), numpy.max(values), posa)

    sns.set_style('ticks')
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)

    fams, data = zip(*sorted(missing_h2s.items()))
    #means = [numpy.mean(d) for d in data]
    #se95 = [1.96*numpy.std(d)/numpy.sqrt(len(d)) for d in data]
    #ci95 = zip(*[(numpy.mean(d)-numpy.percentile(d,2.5), numpy.percentile(d,97.5)-numpy.mean(d)) for d in data])
    #plt.errorbar(fams, means, ci95, fmt='ko', lw=1)
    plt.boxplot(data, labels = fams)
    plt.title(p_missing)
    plt.ylim(0,1.05)
    plt.ylabel("$h_2^o$ from EHR")
    plt.xlabel('Number of Families Sampled')
    plt.subplot(1,2,2)

    fsig = list()
    pfams = list()
    
    for f in fams:
        if float(missing_successes[f]) == 0:
            continue
        fsig.append(missing_significant[f]/float(missing_successes[f]))
        pfams.append(f)
    
    plt.plot(pfams, fsig, 'ko-',lw=1)
    plt.plot([0,max(fams)],[0.9,0.9], 'r-',lw=1)
    plt.ylabel("POSA")
    plt.xlabel('Number of Families Sampled')
    plt.ylim(0,1.05)
    plt.xlim(0,max(fams)+10)
    #plt.xticks(fams)
    sns.despine(trim=True)

keys = sorted(noise_analysis.keys())

p_noise = list()
h2_med = list()
h2_lo = list()
h2_hi = list()

for k in keys:

    noisy_h2s, noisy_significant, noisy_successes = noise_analysis[k]
    nfam = max(noisy_h2s.keys())

    p_noise.append(k)
    h2_med.append(numpy.median(noisy_h2s[nfam]))
    h2_lo.append(numpy.percentile(noisy_h2s[nfam], 2.5))
    h2_hi.append(numpy.percentile(noisy_h2s[nfam], 97.5))

p_noise = numpy.array(p_noise)
h2_med = numpy.array(h2_med)
h2_lo = numpy.array(h2_lo)
h2_hi = numpy.array(h2_hi)

sns.set(style='ticks', font_scale=1.4)
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
for c, p, h2, l, h in zip(colors, p_noise, h2_med, h2_lo, h2_hi):
    plt.errorbar([p], [h2], yerr=([h2-l], [h-h2]), fmt='o-', lw=1, color=c)
plt.xlim(-0.05, 0.35)
plt.xticks(p_noise, p_noise*100)
plt.xlabel("% Noise Injected")
plt.ylabel("$h_2^o$ from EHR")
plt.ylim(0, 1)
sns.despine(trim=True)


keys = sorted(missing_analysis.keys())

p_missing = list()
h2_med = list()
h2_lo = list()
h2_hi = list()

for k in keys:

    missing_h2s, missing_significant, missing_successes = missing_analysis[k]
    nfam = max(missing_h2s.keys())

    p_missing.append(k)
    h2_med.append(numpy.median(missing_h2s[nfam]))
    h2_lo.append(numpy.percentile(missing_h2s[nfam], 2.5))
    h2_hi.append(numpy.percentile(missing_h2s[nfam], 97.5))

p_missing = numpy.array(p_missing)
h2_med = numpy.array(h2_med)
h2_lo = numpy.array(h2_lo)
h2_hi = numpy.array(h2_hi)

plt.subplot(1,2,2)
for c, p, h2, l, h in zip(colors, p_missing, h2_med, h2_lo, h2_hi):
    plt.errorbar([p], [h2], yerr=([h2-l], [h-h2]), fmt='o-', lw=1, color=c)
plt.xlim(-0.05, 0.35)
plt.xticks(p_missing, p_missing*100)
plt.xlabel("% Missingness Injected")
plt.ylabel("$h_2^o$ from EHR")
plt.ylim(0, 1)
sns.despine(trim=True)

plt.savefig('../results/h2o_sensitivity_analysis.pdf')




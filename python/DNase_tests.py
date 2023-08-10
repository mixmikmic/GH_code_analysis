data_dir = "../data/dnase/"

k562_peaks = {}
hepg2_peaks = {}

f = open(data_dir + "wgEncodeUWDukeDnaseK562.fdr01peaks.hg19.bed", 'r')
reads = f.readlines()
f.close()
for read in reads:
    coords = read.split()[:3]
    if coords[0][3:] in k562_peaks:
        k562_peaks[coords[0][3:]].append((int(coords[1]), int(coords[2])))
    else:
        k562_peaks[coords[0][3:]] = [(int(coords[1]), int(coords[2]))]
        
f = open(data_dir + "wgEncodeUWDukeDnaseHepG2.fdr01peaks.hg19.bed", 'r')
reads = f.readlines()
f.close()
for read in reads:
    coords = read.split()[:3]
    if coords[0][3:] in hepg2_peaks:
        hepg2_peaks[coords[0][3:]].append((int(coords[1]), int(coords[2])))
    else:
        hepg2_peaks[coords[0][3:]] = [(int(coords[1]), int(coords[2]))]

from collections import *
import math
import numpy as np
from subprocess import Popen, PIPE

letterindex = {'A': 0, 'a': 0, 'T': 1, 't': 1, 'C': 2, 'c': 2, 'G': 3, 'g': 3, 'N': -1, 'n': -1}

def bases(chrom, start, end):
    seq_count = int(math.ceil((float(end - start) / 60.0)))
    sum_seq = ""
    for i in xrange(seq_count - 1):
        p = Popen(['samtools', 'faidx', '../Genome/hg19.fa', 'chr' + str(chrom) + ':' + str(start + 1 + i * 60) + '-' + str(start + 1 + (i + 1) * 60)], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        sum_seq = sum_seq + output.split('\n')[1]
    p = Popen(['samtools', 'faidx', '../Genome/hg19.fa', 'chr' + str(chrom) + ':' + str(start + 1 + (seq_count - 1) * 60) + '-' + str(end)], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    sum_seq = sum_seq + output.split('\n')[1]
    return sum_seq

data_dir = "../data/dnase/"
hepg2seqs = []
f = open(data_dir + "HepG2seqs.txt")
reads = f.readlines()
f.close()
for read in reads:
    if read[0] != '>':
        read = read[:-1]
        seql = len(read)
        hepg2seqs.append(read[seql/2 - 72 : seql/2 + 73].upper())

k562seqs = []
f = open(data_dir + "K562seqs.txt")
reads = f.readlines()
f.close()
for read in reads:
    if read[0] != '>':
        read = read[:-1]
        seql = len(read)
        k562seqs.append(read[seql/2 - 72 : seql/2 + 73].upper())

hepg2seqs_spec = []
f = open("HepG2seqs_spec.txt")
reads = f.readlines()
f.close()
for read in reads:
    if read[0] != '>':
        read = read[:-1]
        seql = len(read)
        hepg2seqs_spec.append(read[seql/2 - 72 : seql/2 + 73].upper())

k562seqs_spec = []
f = open("K562seqs_spec.txt")
reads = f.readlines()
f.close()
for read in reads:
    if read[0] != '>':
        read = read[:-1]
        seql = len(read)
        k562seqs_spec.append(read[seql/2 - 72 : seql/2 + 73].upper())

from dragonn.models import SequenceDNN_Regression
from collections import *
import math
import numpy as np

model = SequenceDNN_Regression.load('models/models/100n1_100n2_8w1_15w2.arch.json', 'models/models/100n1_100n2_8w1_15w2.weights.h5')

print "startk562"
k562outs = model.predict(model_input(k562seqs_spec))
print "starthepg2"
hepg2outs = model.predict(model_input(hepg2seqs_spec))

letterindex = {'A': 0, 'a': 0, 'T': 1, 't': 1, 'C': 2, 'c': 2, 'G': 3, 'g': 3, 'N': -1, 'n': -1}

def model_input(seqs):
    mi = np.zeros((len(seqs), 1, 4, len(seqs[0])))
    for j in xrange(len(seqs)):
        for i in xrange(len(seqs[0])):
            mi[j][0][letterindex[seqs[j][i]]][i] = 1
    return mi

[sum(hepg2outs[:,i])/float(len(hepg2seqs)) for i in xrange(4)]

[sum(k562outs[:,i])/float(len(k562seqs)) for i in xrange(4)]

[sum(hepg2outs[:,i] > 0)/float(len(hepg2seqs)) for i in xrange(4)]

[sum(k562outs[:,i] > 0)/float(len(k562seqs)) for i in xrange(4)]

print sum([int(hepg2outs[i, 0] > hepg2outs[i, 1]) for i in xrange(len(hepg2outs))]), len(hepg2outs)
print sum([int(hepg2outs[i, 2] > hepg2outs[i, 3]) for i in xrange(len(hepg2outs))]), len(hepg2outs)

hdiffs0 = [hepg2outs[i, 0] - hepg2outs[i, 1] for i in xrange(len(hepg2outs))]

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(hdiffs0, bins=100)
plt.show()



print "startk562"
k562specouts = model.predict(model_input(k562seqs_spec))
print "starthepg2"
hepg2specouts = model.predict(model_input(hepg2seqs_spec))

np.asarray([sum(hepg2specouts[:,i])/float(len(hepg2seqs_spec)) for i in xrange(4)]) - np.asarray([sum(k562specouts[:,i])/float(len(k562seqs_spec)) for i in xrange(4)])

print sum([int(hepg2specouts[i, 0] > hepg2specouts[i, 1]) for i in xrange(len(hepg2specouts))]), len(hepg2specouts)
print sum([int(hepg2specouts[i, 2] > hepg2specouts[i, 3]) for i in xrange(len(hepg2specouts))]), len(hepg2specouts)

sum([int(k562specouts[i, 0] < k562specouts[i, 1]) for i in xrange(len(k562specouts))]), len(k562specouts)

np.asarray([sum(hepg2outs[:,i])/float(len(hepg2seqs)) for i in xrange(4)]) - np.asarray([sum(k562outs[:,i])/float(len(k562seqs)) for i in xrange(4)])

hepg2seqs_bg = []
f = open(data_dir + "HepG2seqs_bg.txt")
reads = f.readlines()
f.close()
for read in reads:
    if read[0] != '>':
        read = read[:-1]
        seql = len(read)
        hepg2seqs_bg.append(read[seql/2 - 72 : seql/2 + 73].upper())

k562seqs_bg = []
f = open(data_dir + "K562seqs_bg.txt")
reads = f.readlines()
f.close()
for read in reads:
    if read[0] != '>':
        read = read[:-1]
        seql = len(read)
        k562seqs_bg.append(read[seql/2 - 72 : seql/2 + 73].upper())

print "startk562"
k562bgouts = model.predict(model_input(k562seqs_bg))
print "starthepg2"
hepg2bgouts = model.predict(model_input(hepg2seqs_bg))

np.asarray([sum(hepg2outs[:,i])/float(len(hepg2seqs)) for i in xrange(4)]) - np.asarray([sum(hepg2bgouts[:,i])/float(len(hepg2seqs_bg)) for i in xrange(4)])

np.asarray([sum(k562outs[:,i])/float(len(k562seqs)) for i in xrange(4)]) - np.asarray([sum(k562bgouts[:,i])/float(len(k562seqs_bg)) for i in xrange(4)])

hepg2seqs_bg_bad = []
f = open(data_dir + "HepG2seqs_bg_bad.txt")
reads = f.readlines()
f.close()
for read in reads:
    if read[0] != '>':
        read = read[:-1]
        seql = len(read)
        hepg2seqs_bg_bad.append(read[seql/2 - 72 : seql/2 + 73].upper())

k562seqs_bg_bad = []
f = open(data_dir + "K562seqs_bg_bad.txt")
reads = f.readlines()
f.close()
for read in reads:
    if read[0] != '>':
        read = read[:-1]
        seql = len(read)
        k562seqs_bg_bad.append(read[seql/2 - 72 : seql/2 + 73].upper())

print "startk562"
k562bgbadouts = model.predict(model_input(k562seqs_bg_bad))
print "starthepg2"
hepg2bgbadouts = model.predict(model_input(hepg2seqs_bg_bad))

np.asarray([sum(hepg2outs[:,i])/float(len(hepg2seqs)) for i in xrange(4)]) - np.asarray([sum(hepg2bgbadouts[:,i])/float(len(hepg2seqs_bg_bad)) for i in xrange(4)])

means = np.asarray([-0.17303934, -0.24127505, -0.12425002, -0.06486262])

counter = 0

for out in hepg2specouts:
    rescale = np.asarray(out) - means
    if rescale[0] > rescale[1] and rescale[2] > rescale[3]:
        counter += 1
    if rescale[0] > rescale[1] and rescale[2] < rescale[3]:
        counter += 0.5
    if rescale[0] < rescale[1] and rescale[2] > rescale[3]:
        counter += 0.5

counter

len(hepg2specouts)

k562outs.shape

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

a = 2

from scipy.stats import linregress
linregress(k562outs[:, 0], k562outs[:, 1])

plt.scatter(k562outs[:,0], k562outs[:,1], s=1.0, lw=0)
plt.show()
linregress(k562outs[:, 0], k562outs[:, 1])

plt.scatter(k562outs[:,2], k562outs[:,3], s=1.0, lw=0)
plt.show()
linregress(k562outs[:, 2], k562outs[:, 3])

plt.scatter(hepg2outs[:,0], hepg2outs[:,1], s=1.0, lw=0)
plt.show()
linregress(hepg2outs[:, 0], hepg2outs[:, 1])

plt.scatter(hepg2outs[:,2], hepg2outs[:,3], s=1.0, lw=0)
plt.show()
linregress(hepg2outs[:, 2], hepg2outs[:, 3])




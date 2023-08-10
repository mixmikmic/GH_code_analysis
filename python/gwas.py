import numpy as np

from dragonn.models import SequenceDNN_Regression

model = SequenceDNN_Regression.load('models/models/100n1_100n2_8w1_15w2.arch.json', 'models/models/100n1_100n2_8w1_15w2.weights.h5')

letterindex = {'A': 0, 'a': 0, 'T': 1, 't': 1, 'C': 2, 'c': 2, 'G': 3, 'g': 3, 'N': -1, 'n': -1}

def model_input(seqs):
    mi = np.zeros((len(seqs), 1, 4, len(seqs[0])))
    for j in xrange(len(seqs)):
        for i in xrange(len(seqs[0])):
            mi[j][0][letterindex[seqs[j][i]]][i] = 1
    return mi

lkref, lkalt = [], []

lkf = open("lks.txt", 'r').readlines()
for line in lkf:
    if len(line.split()) == 2:
        lkref.append(line.split()[0])
        lkalt.append(line.split()[1])
del lkref[-1]
del lkalt[-1]
add = [('GAAGTCATCTCTGCACCCAACACACAGGGCCAATCAGCAGGTGTGGCAGGTGTTCCGTCTTCCTGTTGGAGCGATCTTCTAGGTCCTAGTTCCATCTCCAGCATGGCGTGTGACTCTGAGAAGGTCATCAACCAAGCTCAGTTTTAG',
  'GAAGTCATCTCTGCACCCAACACACAGGGCCAATCAGCAGGTGTGGCAGGTGTTCCGTCTTCCTGTTGGAGCTATCTTCTAGGTCCTAGTTCCATCTCCAGCATGGCGTGTGACTCTGAGAAGGTCATCAACCAAGCTCAGTTTTAG'),
 ('AATTTCTTGATTTGGGACCAGTGTAATTGACTAAATATTGTATTTTATTGTTGGCCTTCAGGCATTCAGGAGGCCATGACTACTACAGGGAAAATTCTGGGATCATAAAATTTAGCCAAGCATGAAACTAAAAACTGGGAACATGAG',
  'AATTTCTTGATTTGGGACCAGTGTAATTGACTAAATATTGTATTTTATTGTTGGCCTTCAGGCATTCAGGAGACCATGACTACTACAGGGAAAATTCTGGGATCATAAAATTTAGCCAAGCATGAAACTAAAAACTGGGAACATGAG'),
 ('CCTTTGCACCAATCTAATAATATAAATACAGAAAGTTCACTGTCCAATTACAATGCAGCATGGTGAGCGCATCCTAGGATGAATGGAAGTGCTATGGGCTTGCCGGTGAAATATTTAATCCTCATACCTCTGCACAGATGTTTTACT',
  'CCTTTGCACCAATCTAATAATATAAATACAGAAAGTTCACTGTCCAATTACAATGCAGCATGGTGAGCGCATTCTAGGATGAATGGAAGTGCTATGGGCTTGCCGGTGAAATATTTAATCCTCATACCTCTGCACAGATGTTTTACT'),
 ('AAATTTAATTTGTCTAATCCATAGTTTAACTAGAGTTTCCACTCAAACCAAAAGTAAATCTTACCTAGAAAATAATGTACCTGGCCGGGCACAGTGGCTCTTGCCTGTAATCCCAGCACTTTGGGAGGCCAAGGCGGATGGATTACG',
  'AAATTTAATTTGTCTAATCCATAGTTTAACTAGAGTTTCCACTCAAACCAAAAGTAAATCTTACCTAGAAAAGAATGTACCTGGCCGGGCACAGTGGCTCTTGCCTGTAATCCCAGCACTTTGGGAGGCCAAGGCGGATGGATTACG'),
 ('CTGGATTTAAACCTTTGCTCCTCCATTTACTAATTATGACTTGACCAAATGACTTAGCCTCTTTAAACTTTGGTTTCTCATCCATCTGACAAATAGTTATAATAATGGTATCCACACTTCAGTATTGTTGAGAGGGTTAAATGAAAT',
  'CTGGATTTAAACCTTTGCTCCTCCATTTACTAATTATGACTTGACCAAATGACTTAGCCTCTTTAAACTTTGATTTCTCATCCATCTGACAAATAGTTATAATAATGGTATCCACACTTCAGTATTGTTGAGAGGGTTAAATGAAAT'),
 ('TTTATTCATTAGTCTTCCTCCTCACATATCACTTGCTTTCTTTTCTGAGTCAGTTTAACTTCGAGCCTATAGTTTTTTTCTTTTCTTCCTTCTCCCCATTGCACCAAAATAATGTGAAGAAAAAGACAAGTCAGAATTTCTGTCCCC',
  'TTTATTCATTAGTCTTCCTCCTCACATATCACTTGCTTTCTTTTCTGAGTCAGTTTAACTTCGAGCCTATAGCTTTTTTCTTTTCTTCCTTCTCCCCATTGCACCAAAATAATGTGAAGAAAAAGACAAGTCAGAATTTCTGTCCCC'),
 ('CCTACAATAGCTTGTCTTGAAGCCAGACCTCAGCCCAATAGTCCAGTATATAAAAACCCATGAATATGTAAAGTAGACCTACTGAAGAAGAGGAAAACCAAATTACTCCAGAAAGGCACCAGTTTTCTCCTTCATACTCATGTTCAA',
  'CCTACAATAGCTTGTCTTGAAGCCAGACCTCAGCCCAATAGTCCAGTATATAAAAACCCATGAATATGTAAATTAGACCTACTGAAGAAGAGGAAAACCAAATTACTCCAGAAAGGCACCAGTTTTCTCCTTCATACTCATGTTCAA'),
 ('CAATTTTGTGGGGCAAAGTTGGCAGATCCCAGCTTTAATTTCTCTTTCATGTTTTCATAGCATCTTGAAATGGCTTTTAAGCTTCTATTTTTTTTTCCAATTCATCCTTTGGCAGGAGGACCATAACCCTTATAATCATGGACAGGC',
  'CAATTTTGTGGGGCAAAGTTGGCAGATCCCAGCTTTAATTTCTCTTTCATGTTTTCATAGCATCTTGAAATGCCTTTTAAGCTTCTATTTTTTTTTCCAATTCATCCTTTGGCAGGAGGACCATAACCCTTATAATCATGGACAGGC'),
 ('ACACACCTGTAGTCCCAGCTACTTGGGAGGCTGAGGCAGGAGGATCACTTGAGCCCAGGAGGCTGAGGCTACTGTGAGCTGTGGTTGTGCCACTGTACTACAGCCTCGGTGACAGGGTGAGACCCTGTGTCTAAAACAAAACAAAAC',
  'ACACACCTGTAGTCCCAGCTACTTGGGAGGCTGAGGCAGGAGGATCACTTGAGCCCAGGAGGCTGAGGCTACAGTGAGCTGTGGTTGTGCCACTGTACTACAGCCTCGGTGACAGGGTGAGACCCTGTGTCTAAAACAAAACAAAAC'),
 ('TGAGGGTCACGTGATGGGGACAAAGGGAGACAGGAACAACAGGTTCAGATCAGGACCTCCAGATCTGGTTAGACAGGACCTCCCAGGGTAGCCAAGGAGTCTGGGTTCGATTTTATTTTGCAGGGAGCGTTTTATGCTAGAGTGATG',
  'TGAGGGTCACGTGATGGGGACAAAGGGAGACAGGAACAACAGGTTCAGATCAGGACCTCCAGATCTGGTTAGGCAGGACCTCCCAGGGTAGCCAAGGAGTCTGGGTTCGATTTTATTTTGCAGGGAGCGTTTTATGCTAGAGTGATG'),
 ('AGAGACAAGAACCCAGCAGTGTGTGCAGAGTCAAGGCCGTTCAGGGAGCTCTGCAAATGCCCGTTCCACGCTAGCTAAAATGCACGGTTCCCCTCTCCCCGGAAGAAAAGGCAGCAGCGTGGGTTTTTTGTTTTTTTTTCTTTTCTC',
  'AGAGACAAGAACCCAGCAGTGTGTGCAGAGTCAAGGCCGTTCAGGGAGCTCTGCAAATGCCCGTTCCACGCTGGCTAAAATGCACGGTTCCCCTCTCCCCGGAAGAAAAGGCAGCAGCGTGGGTTTTTTGTTTTTTTTTCTTTTCTC'),
 ('GGCACTGTGCCCCAGAATGGCCCCACCCATGCAGGAGGGCACCTTCCCTCCTCACTGTGCTGGCTTCTTTTAACAAATAGCCCTATCTATGAGCTCCCTTGTTCCTTCCAACCATTCTAAAGATAGGGCAGGCACCAAGGCCAGGGG',
  'GGCACTGTGCCCCAGAATGGCCCCACCCATGCAGGAGGGCACCTTCCCTCCTCACTGTGCTGGCTTCTTTTATCAAATAGCCCTATCTATGAGCTCCCTTGTTCCTTCCAACCATTCTAAAGATAGGGCAGGCACCAAGGCCAGGGG'),
 ('GCCTTGGGTCCATGGGGAGAGCTTGGCTCAGGAAGCCGGTGCTGCCTCTATCACCTCTGAGTTTCCAAGCCATACTCTCTCACCCCATCCCAGCTTGGGAAACAGCCAGCATCTTGCTGGCCTCTTGGTGGAGTGTTGCTGATCCAT',
  'GCCTTGGGTCCATGGGGAGAGCTTGGCTCAGGAAGCCGGTGCTGCCTCTATCACCTCTGAGTTTCCAAGCCACACTCTCTCACCCCATCCCAGCTTGGGAAACAGCCAGCATCTTGCTGGCCTCTTGGTGGAGTGTTGCTGATCCAT'),
 ('GATTTCCTGTGCACACTAAAGTTTGAGAAGCACAAGATTAGGAGTCAAAATCCCTGAGTTCCTGGTCCAGCTATTGCTATCAAATTGCTGTGTGGTTTTGCAGGACTTGCTTGACCTCTCTGGGTGCTAGAATTTCTTCCTCTGGGC',
  'GATTTCCTGTGCACACTAAAGTTTGAGAAGCACAAGATTAGGAGTCAAAATCCCTGAGTTCCTGGTCCAGCTGTTGCTATCAAATTGCTGTGTGGTTTTGCAGGACTTGCTTGACCTCTCTGGGTGCTAGAATTTCTTCCTCTGGGC')]
for pair in add:
    lkref.append(pair[0])
    lkalt.append(pair[1])

lvref, lvalt = [], []

lvf = open("lvs.txt", 'r').readlines()
for line in lvf:
    if len(line.split()) == 2:
        lvref.append(line.split()[0])
        lvalt.append(line.split()[1])
add = ('AATGATATGAAATGAAATCAATGTACATAAAAAATAGTCCCTAGTAATAGAAATCAGAAAGTAGTTGCCTAGGGGAAGGAGACATTGACTGGAAAGAAAAAAATGAAGGATTATTCTGGGGCGATGGAAATATTGTATCTCTTTTTG',
 'AATGATATGAAATGAAATCAATGTACATAAAAAATAGTCCCTAGTAATAGAAATCAGAAAGTAGTTGCCTAGAGGAAGGAGACATTGACTGGAAAGAAAAAAATGAAGGATTATTCTGGGGCGATGGAAATATTGTATCTCTTTTTG')
lvref.append(add[0])
lvalt.append(add[1])

lvref = [seq[:145] for seq in lvref]
lvalt = [seq[:145] for seq in lvalt]
lkref = [seq[:145] for seq in lkref]
lkalt = [seq[:145] for seq in lkalt]

lvrefouts.shape

lvrefouts = model.predict(model_input(lvref))
lvaltouts = model.predict(model_input(lvalt))

lkrefouts = model.predict(model_input(lkref))
lkaltouts = model.predict(model_input(lkalt))

lvdiffsM = [lvaltouts[i][0] - lvrefouts[i][0] for i in xrange(len(lvrefouts))]
lvdiffsS = [lvaltouts[i][2] - lvrefouts[i][2] for i in xrange(len(lvrefouts))]
lkdiffsM = [lkaltouts[i][1] - lkrefouts[i][1] for i in xrange(len(lkrefouts))]
lkdiffsS = [lkaltouts[i][3] - lkrefouts[i][3] for i in xrange(len(lkrefouts))]

len(lvdiffsM), len(lkdiffsM)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(lvdiffsM, bins=100)
plt.show()

plt.hist(lvdiffsS, bins=100)
plt.show()

a = plt.hist(lkdiffsM, bins=100)
plt.show()

plt.hist(lkdiffsS, bins=100)
plt.show()

from scipy.stats import spearmanr
spearmanr(lvdiffsS, lvdiffsM)

a, b = zip(*sorted(zip(lvdiffsS, range(len(lvdiffsS)))))

c, d = zip(*sorted(zip(lvdiffsM, range(len(lvdiffsM)))))

zip(b, d)

e, f = zip(*sorted(zip(lkdiffsS, range(len(lkdiffsS)))))
g, h = zip(*sorted(zip(lkdiffsM, range(len(lkdiffsM)))))
zip(f, h)

hepg2_peaks = [line[:-1].upper().replace('N', 'A') for line in open("h_bg_501.txt", 'r').readlines()[1::2]]

k562_peaks = [line[:-1].upper().replace('N', 'A') for line in open("k_bg_501.txt", 'r').readlines()[1::2]]

hepg2_peaks_input = []
for seq in hepg2_peaks:
    mid = len(seq) / 2
    hepg2_peaks_input.append(seq[mid - 72 : mid + 73])
k562_peaks_input = []
for seq in k562_peaks:
    mid = len(seq) / 2
    k562_peaks_input.append(seq[mid - 72 : mid + 73])

hepg2_peaks_input_A = [seq[:72] + 'A' + seq[73:] for seq in hepg2_peaks_input]
hepg2_peaks_input_T = [seq[:72] + 'T' + seq[73:] for seq in hepg2_peaks_input]
hepg2_peaks_input_G = [seq[:72] + 'G' + seq[73:] for seq in hepg2_peaks_input]
hepg2_peaks_input_C = [seq[:72] + 'C' + seq[73:] for seq in hepg2_peaks_input]
k562_peaks_input_A = [seq[:72] + 'A' + seq[73:] for seq in k562_peaks_input]
k562_peaks_input_T = [seq[:72] + 'T' + seq[73:] for seq in k562_peaks_input]
k562_peaks_input_G = [seq[:72] + 'G' + seq[73:] for seq in k562_peaks_input]
k562_peaks_input_C = [seq[:72] + 'C' + seq[73:] for seq in k562_peaks_input]

HA = model.predict(model_input(hepg2_peaks_input_A))
HT = model.predict(model_input(hepg2_peaks_input_T))
HC = model.predict(model_input(hepg2_peaks_input_C))
HG = model.predict(model_input(hepg2_peaks_input_G))

h_m_variation = []
for i in xrange(len(hepg2_peaks)):
    predictions = [HA[i][0], HT[i][0], HC[i][0], HG[i][0]]
    for j in xrange(len(predictions)):
        for k in xrange(j):
            h_m_variation.append(abs(predictions[j] - predictions[k]))
h_s_variation = []
for i in xrange(len(hepg2_peaks)):
    predictions = [HA[i][2], HT[i][2], HC[i][2], HG[i][2]]
    for j in xrange(len(predictions)):
        for k in xrange(j):
            h_s_variation.append(abs(predictions[j] - predictions[k]))

s_h_m = sorted(h_m_variation)
s_h_s = sorted(h_s_variation)
s_k_m = sorted(k_m_variation)
s_k_s = sorted(k_s_variation)

def rank(iterable, score):
    lo, hi = 0, len(iterable) - 1
    while lo < hi - 1:
        mid = (lo + hi) / 2
        if iterable[mid] < score:
            lo = mid
        else:
            hi = mid
    return lo

def significance_h_m(score):
    return 1.0 - float(rank(s_h_m, score)) / float(len(s_h_m))

def significance_h_s(score):
    return 1.0 - float(rank(s_h_s, score)) / float(len(s_h_s))

def significance_k_m(score):
    return 1.0 - float(rank(s_k_m, score)) / float(len(s_k_m))

def significance_k_s(score):
    return 1.0 - float(rank(s_k_s, score)) / float(len(s_k_s))

lkm = [abs(i) for i in lkdiffsM]
lvm = [abs(i) for i in lvdiffsM]
lks = [abs(i) for i in lkdiffsS]
lvs = [abs(i) for i in lvdiffsS]

sig_lv = [significance_h_m(lvm[i]) * significance_h_s(lvs[i]) for i in xrange(len(lvm))]
ssig_lv, lv_indices = zip(*sorted(zip(sig_lv, range(len(sig_lv)))))
[(ssig_lv[i] < (0.05 / (len(ssig_lv) - i))) for i in xrange(len(ssig_lv))][:10]

sig_lk = [significance_k_m(lkm[i]) * significance_k_s(lks[i]) for i in xrange(len(lkm))]
ssig_lk, lk_indices = zip(*sorted(zip(sig_lk, range(len(sig_lk)))))
[(ssig_lk[i] < (0.05 / (len(ssig_lk) - i))) for i in xrange(len(ssig_lk))][:10]

KA = model.predict(model_input(k562_peaks_input_A))
KT = model.predict(model_input(k562_peaks_input_T))
KC = model.predict(model_input(k562_peaks_input_C))
KG = model.predict(model_input(k562_peaks_input_G))

k_m_variation = []
for i in xrange(len(k562_peaks)):
    predictions = [KA[i][1], KT[i][1], KC[i][1], KG[i][1]]
    for j in xrange(len(predictions)):
        for k in xrange(j):
            k_m_variation.append(abs(predictions[j] - predictions[k]))
k_s_variation = []
for i in xrange(len(k562_peaks)):
    predictions = [KA[i][3], KT[i][3], KC[i][3], KG[i][3]]
    for j in xrange(len(predictions)):
        for k in xrange(j):
            k_s_variation.append(abs(predictions[j] - predictions[k]))

lv_indices[:4], lk_indices[:9]
ssig_lv[:4], ssig_lk[:9]

0.05 / (len(ssig_lk) - 8)

skm = sorted(lkm)
sks = sorted(lks)
svm = sorted(lvm)
svs = sorted(lvs)

x = np.linspace(0.0, 0.5, 1000)[::-1]

y0 = [(1.0 - float(rank(skm, x[i])) / len(skm)) for i in xrange(len(x))]
y1 = [significance_k_m(score) for score in x]

import matplotlib.pyplot as plt
plt.plot(x, y0)
plt.plot(x, y1)
plt.show()

y0 = [(1.0 - float(rank(svm, x[i])) / len(svm)) for i in xrange(len(x))]
y1 = [significance_h_m(score) for score in x]

import matplotlib.pyplot as plt
plt.plot(x, y0)
plt.plot(x, y1)
plt.show()

import matplotlib.pyplot as plt
plt.plot(x, y0)
plt.plot(x, y1)
plt.show()




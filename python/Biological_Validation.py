import matplotlib
matplotlib.use('pdf')
get_ipython().magic('pylab inline')

import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')

import pandas, numpy

from operator import itemgetter
from scipy.stats import spearmanr

celltypes = 'GM12878', 'K562', 'IMR90', 'NHEK', 'HMEC', 'HUVEC'
colors = ['c', 'm', 'g', 'r', 'b', '#FF6600']

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

low, high = 10, 201
p, d, idxs = numpy.array([]), numpy.array([]), []

y_pred = numpy.load('contact_maps/chr21.GM12878.y_pred.5000.npy', mmap_mode='r')
p_map = numpy.load('contact_maps/chr21.GM12878.p.5000.1000.npy', mmap_mode='r')

for i in range(low, high):
    # Identify which regions we have predictions for, as some regions are unmappable regions
    # The model will never predict either exactly 0 or exactly 1.
    idx = numpy.diag(y_pred, i) != 0
    idxs.append(idx)
    
    x = -numpy.log(numpy.diag(p_map, i)[idx])
    p = numpy.concatenate([p, x])
    d = numpy.concatenate([d, -numpy.ones_like(x)*i])
    

plt.figure( figsize=(10, 18) )
for i, (c, celltype) in enumerate(zip(colors, celltypes)):
    yp = numpy.load('contact_maps/chr21.{}.y_pred.5000.npy'.format(celltype), mmap_mode='r')
    yt = numpy.load('contact_maps/chr21.{}.y.5000.npy'.format(celltype), mmap_mode='r')
    
    y_true, y_pred = numpy.array([]), numpy.array([])
    for k, idx in zip(range(low, high), idxs):        
        y_true = numpy.concatenate([y_true, numpy.diag(yt, k)[idx]])
        y_pred = numpy.concatenate([y_pred, numpy.diag(yp, k)[idx]])

    plt.subplot(6, 2, (2*i)+1)
    plt.plot([0,1], [0, 1], c='k', alpha=0.6)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    eval_auc = numpy.around(roc_auc_score(y_true, y_pred), 4)
    plt.plot(fpr, tpr, c=c, label=eval_auc)

    fpr, tpr, _ = roc_curve(y_true, d)
    eval_auc = numpy.around(roc_auc_score(y_true, d), 4)
    plt.plot(fpr, tpr, c=c, linestyle=':', label=eval_auc )

    fpr, tpr, _ = roc_curve(y_true, p)
    eval_auc = numpy.around(roc_auc_score(y_true, p), 4)
    plt.plot(fpr, tpr, c=c, linestyle='--', label=eval_auc)

    plt.xticks([0, 0.5, 1], fontsize=10)
    plt.yticks([0, 0.5, 1], fontsize=10)
    plt.ylabel("TPR", fontsize=12)
    plt.legend(loc=4)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(False)

    plt.subplot(6, 2, (2*i)+2)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    eval_auc = numpy.around(average_precision_score(y_true, y_pred), 4)
    plt.plot(recall, precision, c=c, label=eval_auc)

    precision, recall, _ = precision_recall_curve(y_true, d)
    eval_auc = numpy.around(average_precision_score(y_true, d), 4)
    plt.plot(recall, precision, c=c, linestyle=':', label=eval_auc)

    precision, recall, _ = precision_recall_curve(y_true, p)
    eval_auc = numpy.around(average_precision_score(y_true, p), 4)
    plt.plot(recall, precision, c=c, linestyle='--', label=eval_auc)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks([0, 0.5, 1], fontsize=10)
    plt.yticks([0, 0.5, 1], fontsize=10)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc=3)
    plt.grid(False)

plt.subplot(6, 2, 1)
plt.title("ROC", fontsize=12)

plt.subplot(6, 2, 2)
plt.title("Precision-Recall", fontsize=12)

plt.subplot(6, 2, 11)
plt.xlabel("FPR", fontsize=12)
plt.ylabel("TPR", fontsize=12)

plt.subplot(6, 2, 12)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.savefig('ISMB_5kb_validation.pdf')

del precision, recall, fpr, tpr, p, d, x, y_pred, y_true

r_insulation_score, c_insulation_score, dnases = [], [], []
for celltype in celltypes:
    r_insulation_score.append(numpy.load('insulation/chr21.{}.insulation.rambutan.5000.npy'.format(celltype))[3500:9500])
    c_insulation_score.append(numpy.load('insulation/chr21.{}.insulation.5000.npy'.format(celltype))[3500:9500])
    
    d = numpy.load('dnase/chr21.{}.dnase.npy'.format(celltype))
    dnase = numpy.zeros(9626)
    for i in range(9626):
        dnase[i] = d[i*5000:(i+1)*5000].mean()
    dnase[dnase == 0] = 1
    dnase = numpy.log(dnase) / numpy.log(dnase).std()
    dnases.append(dnase[3500:9500])
    
idx = [(c != 0) & (r != 0) for c, r in zip(c_insulation_score, r_insulation_score)] 
r_insulation_score = [r_insulation_score[i][idx[i]] for i in range(6)]
c_insulation_score = [c_insulation_score[i][idx[i]] for i in range(6)]
dnases = [dnases[i][idx[i]] for i in range(6)]

r_insulation_score = [numpy.log(r_insulation_score[i]) - numpy.log(r_insulation_score[i].mean()) for i in range(6)]
c_insulation_score = [numpy.log(c_insulation_score[i]) - numpy.log(c_insulation_score[i].mean()) for i in range(6)]

zr_insulation_score = [(r_insulation_score[i] - r_insulation_score[i].mean()) / r_insulation_score[i].std() for i in range(6)]
zc_insulation_score = [(c_insulation_score[i] - c_insulation_score[i].mean()) / c_insulation_score[i].std() for i in range(6)]

for i, celltype in enumerate(celltypes):
    n = len(zr_insulation_score[i])
    
    plt.figure(figsize=(12, 4))
    plt.title(celltype, fontsize=16)
    plt.plot(dnases[i], color='r', alpha=0.3)
    plt.plot(zr_insulation_score[i], color='c', label="Rambutan")
    plt.plot(zc_insulation_score[i], color='m', label="Hi-C")
    #plt.legend(fontsize=14, loc=2)
    plt.xticks(range(0, n, 1000), range(3500*5, 5*(n+3500), 5000), fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Standardized Insulation Score", fontsize=14)
    plt.xlabel("Genomic Coordinate (kb)", fontsize=14)
    plt.savefig('figures/{}_insulation_dnase.pdf'.format(celltype))

plt.figure(figsize=(12, 8))
plt.title("Rambutan Insulation Score versus Hi-C Insulation Score", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Rambutan Insulation Score", fontsize=14)
plt.ylabel("Insulation Score", fontsize=14)

for i, color, celltype in zip(range(6), colors, celltypes):
    plt.subplot(2, 3, i+1)
    plt.grid(False)
    corr = numpy.corrcoef(r_insulation_score[i][::200], c_insulation_score[i][::200])[0, 1]
    plt.scatter(r_insulation_score[i][::200], c_insulation_score[i][::200], edgecolor=color, color=color, label="{}: {:3.3}".format(celltype, corr))
    plt.legend(fontsize=12, loc=2)

plt.subplot(2, 3, 1)
plt.ylabel("Hi-C Insulation Score", fontsize=14)

plt.subplot(2, 3, 4)
plt.ylabel("Hi-C Insulation Score", fontsize=14)

plt.subplot(2, 3, 4)
plt.xlabel("Rambutan Insulation Score", fontsize=14)
plt.ylabel("Hi-C Insulation Score", fontsize=14)

plt.subplot(2, 3, 5)
plt.xlabel("Rambutan Insulation Score", fontsize=14)

plt.subplot(2, 3, 6)
plt.xlabel("Rambutan Insulation Score", fontsize=14)
plt.savefig('figures/insulation_corr.pdf')

plt.figure(figsize=(10, 8))
plt.subplot(211)
for insulation_score, color, celltype in zip(c_insulation_score, colors, celltypes):
    plt.title("Hi-C Derived Insulation Scores", fontsize=14)
    plt.plot(insulation_score, color=color, label=celltype)
plt.xticks(range(0, 6001, 1000), range(3500*5, 9501*5, 5000), fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Insulation Score", fontsize=12)
plt.xlim(0, 6000)

plt.subplot(212)
for insulation_score, color, celltype in zip(r_insulation_score, colors, celltypes):
    plt.title("Rambutan Derived Insulation Scores", fontsize=14)
    plt.plot(insulation_score, color=color, label=celltype)
plt.legend(fontsize=12)
plt.xlim(0, 6000)
plt.xticks(range(0, 6001, 1000), range(3500*5, 9501*5, 5000), fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Genomic Coordinate (kb)", fontsize=12)
plt.ylabel("Insulation Score", fontsize=12)
plt.savefig("figures/ISMB_Supp_Insul1.pdf")

plt.figure(figsize=(8, 8))
plt.subplot(211)
plt.grid(False)
plt.title("Correlation Among Hi-C Derived Insulation Scores", fontsize=14)
plt.imshow(numpy.corrcoef([ins[:5972] for ins in c_insulation_score]), interpolation='nearest', vmin=0, vmax=1)
plt.xticks(range(6), ['']*6)
plt.yticks(range(6), celltypes, fontsize=12)
plt.colorbar()

plt.subplot(212)
plt.grid(False)
plt.title("Correlation Among Hi-C Derived Insulation Scores", fontsize=14)
plt.imshow(numpy.corrcoef([ins[:5972] for ins in r_insulation_score]), interpolation='nearest', vmin=0, vmax=1)
plt.xticks(range(6), celltypes, fontsize=12, rotation=90)
plt.yticks(range(6), celltypes, fontsize=12)
plt.colorbar()
plt.savefig("figures/ISMB_Supp_Insul2.pdf")

celltypes = 'GM12878', 'K562', 'IMR90', 'NHEK', 'HUVEC'
colors = ['c', 'm', 'g', 'r', '#FF6600']

r = [numpy.load('replication/chr21.{}.replication.npy'.format(celltype)) for celltype in celltypes]
replication = [numpy.zeros(9626) for i in range(5)]
rr = [None for i in range(5)]

for i in range(5):
    for j in range(9626):
        replication[i][j] = r[i][j*5000:(j+1)*5000].mean()

    if i == 4:
        rr[i] = replication[i][3500:9500][idx[5]]
    else:
        rr[i] = replication[i][3500:9500][idx[i]]

for i, celltype in zip(range(5), celltypes):
    plt.figure(figsize=(12, 3))
    plt.title(celltype, fontsize=16)
    plt.plot(zr_insulation_score[i], color='c', label="Rambutan")
    plt.plot(zc_insulation_score[i], color='m', label="Hi-C")
    plt.plot(rr[i] , color='b', label="Replication Timing")
    #plt.legend(fontsize=14, loc=2)
    plt.xticks(range(0, n, 1000), range(3500*5, 5*(n+3500), 5000), fontsize=12)
    plt.yticks(range(-3, 4), range(-3, 4), fontsize=14)
    plt.ylabel("Normalized Replication Timing", fontsize=14)
    plt.xlabel("Genomic Coordinate (kb)", fontsize=14)
    plt.savefig('figures/{}_replication_timing.pdf'.format(celltype))

plt.figure(figsize=(12, 8))
for i, color, celltype in zip(range(5), colors, celltypes):
    plt.subplot(2, 3, i+1)
    plt.grid(True)
    corr = numpy.corrcoef(r_insulation_score[i][::200], rr[i][::200])[0, 1]
    plt.scatter(r_insulation_score[i][::200], rr[i][::200], edgecolor=color, color=color, label="{}: {:3.3}".format(celltype, corr))
    plt.legend(fontsize=12, loc=2)

plt.subplot(2, 3, 1)
plt.ylabel("Replication Timing", fontsize=14)

plt.subplot(2, 3, 4)
plt.ylabel("Replication Timing", fontsize=14)

plt.subplot(2, 3, 4)
plt.xlabel("Rambutan Insulation Score", fontsize=14)
plt.ylabel("Replication Timing", fontsize=14)

plt.subplot(2, 3, 5)
plt.xlabel("Rambutan Insulation Score", fontsize=14)
plt.savefig('replication_corr.pdf')

celltypes = 'GM12878', 'K562', 'IMR90', 'NHEK', 'HMEC', 'HUVEC'
colors = ['c', 'm', 'g', 'r', 'b', '#FF6600']
histone_names = ['H3K36me3', 'H3K27me3', 'H3K9me3', 'H4K20me1']

plt.figure(figsize=(6, 12))

for ix, his in enumerate(histone_names):
    rvals, ivals, dvals = [], [], []
    
    for i, celltype in enumerate(celltypes):
        rambutan = numpy.load("insulation/chr21.{}.insulation.rambutan.5000.npy".format(celltype))
        insulation = numpy.load("insulation/chr21.{}.insulation.5000.npy".format(celltype))
        
        h = numpy.load('histone/chr21.{}.{}.npy'.format(celltype, his))
        histone = numpy.zeros(rambutan.shape[0])
        for j in range(rambutan.shape[0]):
            histone[j] = h[j*5000:(j+1)*5000].mean() 
        
        rambutan = rambutan[3500:9500][idx[i]]
        insulation = insulation[3500:9500][idx[i]]
        histone = histone[3500:9500][idx[i]]
        
        rvals.append(spearmanr(rambutan, histone)[0])
        ivals.append(spearmanr(insulation, histone)[0])
        dvals.append(spearmanr(dnases[i], histone)[0])
    
    plt.subplot(4, 1, ix+1)
    plt.title(his, fontsize=14)
    plt.xticks(range(6), ['']*6)
    plt.bar(numpy.arange(6)-0.3, rvals, width=0.2, facecolor='c', edgecolor='c', label="Rambutan")
    plt.bar(numpy.arange(6)-0.1, ivals, width=0.2, facecolor='m', edgecolor='m', label="Hi-C")
    plt.bar(numpy.arange(6)+0.1, dvals, width=0.2, facecolor='#FF6600', edgecolor='#FF6600', label="Dnase")
    plt.plot([-0.5, 5.5], [0, 0], color='k')
    plt.legend(fontsize=14, loc=1)
    plt.grid(False)
    plt.yticks(fontsize=14)
    plt.xlim(-0.5, 5.5)

plt.subplot(4, 1, 4)
plt.ylabel("Spearman Correlation", fontsize=14)
plt.xticks(range(6), celltypes, rotation=90, fontsize=14)
plt.savefig('figures/histones.pdf')

del dnases, c_insulation_score, r_insulation_score, zc_insulation_score, zr_insulation_score

celltypes = 'GM12878', 'K562', 'IMR90', 'NHEK', 'HUVEC'
preds, truths = [], []
for k, celltype in enumerate(celltypes):
    y = numpy.load('contact_maps/chr21.{}.p.5000.npy'.format(celltype))
    y_pred = numpy.load('contact_maps/chr21.{}.y_pred.5000.npy'.format(celltype))

    pred = []
    truth = []
    for i in range(9626):
        for j in range(i, 9626):
            if y_pred[i, j] != 0:
                pred.append([j-i, y_pred[i, j], replication[k][i], replication[k][j]])
                truth.append([j-i, y[i, j], replication[k][i], replication[k][j]])

    preds.append(numpy.array(pred))
    truths.append(numpy.array(truth))

n = 1000

fithic_high, fithic_low = [], []
rambutan_high, rambutan_low = [], []

for y, yp in zip(truths, preds):
    fh, fl, rh, rl = [], [], [], []
    for i in range(10, 201):
        x = yp[yp[:,0] == i]
        idxs = x[:,1].argsort()
        rl.append(numpy.abs(x[idxs[:-n], 3] - x[idxs[:-n], 2]).mean())
        rh.append(numpy.abs(x[idxs[-n:], 3] - x[idxs[-n:], 2]).mean())

        x = y[y[:,0] == i]
        idxs = x[:,1].argsort()
        fh.append(numpy.abs(x[idxs[:n], 3] - x[idxs[:n], 2]).mean())
        fl.append(numpy.abs(x[idxs[n:], 3] - x[idxs[n:], 2]).mean())
    
    fithic_high.append(fh)
    fithic_low.append(fl)
    rambutan_high.append(rh)
    rambutan_low.append(rl)

plt.figure(figsize=(6, 14))
plt.figure(figsize=(8, 24))
for i, celltype in enumerate(celltypes):
    plt.subplot(5, 1, i+1)
    plt.title(celltype, fontsize=16)
    plt.yticks(numpy.arange(0.0, 1, 0.2), numpy.arange(0.0, 1, 0.2), fontsize=14)
    plt.xticks(range(0, 191, 20), ['', '', '', '', ''])
    plt.ylim(0, 0.8)
    
    plt.plot(rambutan_low[i],  'b'  , label="Low Confidence Rambutan Calls")
    plt.plot(fithic_low[i],    'c--', label="Low Confidence Fit-Hi-C Calls")
    plt.plot(rambutan_high[i], 'r'  , label="High Confidence Rambutan Calls")
    plt.plot(fithic_high[i],   'm--', label="High Confidence Fit-Hi-C Calls")
    plt.legend(loc=2, fontsize=14)

plt.xlabel("Genomic Distance (kb)", fontsize=14)
plt.ylabel("Average Difference in Replication Timing", fontsize=14)
plt.xticks(range(0, 191, 20), numpy.arange(10, 201, 20)*5, fontsize=14, rotation=45)
plt.savefig('figures/replication_diff.pdf')

del fithic_high, fithic_low, fh, fl, rambutan_high, rambutan_low, rh, rl, x

celltypes = 'GM12878', 'K562', 'IMR90', 'NHEK', 'HMEC', 'HUVEC'
colors = ['c', 'm', 'g', 'r', 'b', '#FF6600']

m = 5000.
rambutans, contacts, label_list = [], [], []
for celltype in celltypes:
    data = pandas.read_csv("segway/SEGWAY_{}.bed".format(celltype), sep="\t", header=None)
    data = data[data[0] == 'chr21'][[1, 2, 3]]

    y = numpy.load('segway/chr21.{}.y_sum.5000.npy'.format(celltype))
    r = numpy.load('segway/chr21.{}.y_pred_sum.5000.npy'.format(celltype))
    
    labels = sorted(list(data[3].unique()))
    n = len(labels)
    
    mapping = {label: i for i, label in enumerate(labels)}
    rambutan = numpy.zeros((9626, n))
    contact = numpy.zeros((9626, n))
    random = numpy.zeros((9626, n))

    for _, (start, end, kind) in data.iterrows():
        k = mapping[kind]
        i, j = int(start // m), int(end // m)
        for l in range(i, j+1):
            rambutan[l, k] += r[l] if end-start > m else r[l] * (end-start) / m
            contact[l, k] += y[l] if end-start > m else y[l] * (end-start) / m
            random[l, k] += (end-start) / m


    rambutan = rambutan.sum(axis=0) / rambutan.sum()
    contact = contact.sum(axis=0) / contact.sum()
    random = random.sum(axis=0) / random.sum()
    rambutan = numpy.log(rambutan / random)
    contact = numpy.log(contact / random)
    rambutan, contact, random, labels = (numpy.array(list(x)) for x in zip(*sorted(zip(rambutan, contact, random, labels),key=itemgetter(1))))
    
    rambutans.append(rambutan)
    contacts.append(contact)
    label_list.append(labels)

for celltype, rambutan, contact, labels in zip(celltypes, rambutans, contacts, label_list):
    n = len(rambutan)
    plt.figure(figsize=(12, 4))
    plt.title("{} Contact Enrichment by Segway Anotation".format(celltype), fontsize=16)
    plt.plot([-0.5, n], [0, 0], color='k')
    plt.bar(numpy.arange(n)-0.3, rambutan, width=0.3, facecolor='c', edgecolor='c', label="Rambutan")
    plt.bar(numpy.arange(n), contact, width=0.3, facecolor='m', edgecolor='m', label="Hi-C")
    plt.xticks(range(n), labels, rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-0.5, n)
    plt.ylabel("Log Fold Enrichment", fontsize=14)
    #plt.legend(fontsize=16, loc=2)
    plt.grid(True)
    plt.savefig('figures/{}_segway.pdf'.format(celltype))

plt.figure(figsize=(18, 12))
for i, color, celltype in zip(range(6), colors, celltypes):
    plt.subplot(2, 3, i+1)
    corr = numpy.corrcoef(rambutans[i], contacts[i])[0, 1]
    plt.scatter(rambutans[i], contacts[i], color=color, s=50, linewidth=0, label="{}: {:3.3}".format(celltype, corr))
    plt.legend(fontsize=14, loc=2)

plt.subplot(2, 3, 1)
plt.ylabel("Hi-C Insulation Score", fontsize=14)

plt.subplot(2, 3, 4)
plt.ylabel("Hi-C Insulation Score", fontsize=14)

plt.subplot(2, 3, 4)
plt.xlabel("Rambutan Insulation Score", fontsize=14)
plt.ylabel("Hi-C Insulation Score", fontsize=14)

plt.subplot(2, 3, 5)
plt.xlabel("Rambutan Insulation Score", fontsize=14)

plt.subplot(2, 3, 6)
plt.xlabel("Rambutan Insulation Score", fontsize=14)
plt.savefig('figures/segway_corr.pdf')

get_ipython().magic('pylab inline')
import pandas; pandas.set_option('display.max_colwidth', 50); pandas.set_option('display.width', 120)
import seaborn; seaborn.set_style('whitegrid'); seaborn.set(font="monospace")

namemap = numpy.loadtxt('RoadmapNames.txt', delimiter=' ', dtype=str)
d = {int(e[1:]) : (name, type) for e, name, type in namemap}

plt.figure(figsize=(20, 8))
plt.ylabel("Insulation Score", fontsize=14)
plt.xlabel("Genomic Coordinate", fontsize=14)
insulation_scores = []
celltypes = []
names = []
types = []

for i in range(129):
    celltype = 'E{}'.format(str(i).zfill(3))
    
    try:
        data = numpy.load('roadmap/chr21.{}.insulation.rambutan.5000.npy'.format(celltype))
        data = data[3500:9500]
        data = data[data != 0]
        data = numpy.log(data) - numpy.log(data.mean())
        celltypes.append(i)
        names.append(d[i][0])
        types.append(d[i][1])
        insulation_scores.append(data)
        plt.plot(data)
    except:
        pass

types = numpy.array(types)
plt.show()
n = len(celltypes)

corr = numpy.corrcoef(insulation_scores)
seaborn.clustermap(corr, linewidths=.5, figsize=(18, 18), xticklabels=names, yticklabels=celltypes)

from sklearn.manifold import TSNE
labels = set(label for label in types)

clf = TSNE(metric='precomputed', method='exact', perplexity=1, learning_rate=10, n_iter=100000)
embedding = clf.fit_transform(1-corr)

seaborn.set_style('white')
colors = seaborn.color_palette('husl', len(labels))

plt.figure(figsize=(10, 10))
for i, label in enumerate(labels):
    idx = types == label
    plt.scatter(embedding[idx,0], embedding[idx, 1], color=colors[i], label=label)
plt.legend(fontsize=14)
plt.savefig('ISMB_roadmap.pdf')

for i in range(n):
    print "{:55} {:10} {}".format(names[i], types[i], embedding[i])


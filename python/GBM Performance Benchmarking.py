get_ipython().system('which python')



import gzip
from collections import OrderedDict
import itertools

def get_seqs_from_file(fasta):
    seqfile = gzip.open(fasta)
    seqs = {}
    for line in seqfile:
        if line[0] == '>':
            name = line.strip()[1:]
        else:
            seq = line.strip()
            seqs[name] = seq
    seqfile.close()
    return seqs

def get_labels_from_file(filename):
    labelfile = gzip.open(filename)
    labels = {}
    labelfile.readline()
    for line in labelfile:
        line = line.strip().split('\t')
        name = line[0]
        vals = [float(lbl) for lbl in line[13:]] # 1 for name + 12 for counts --> 13:
        labels[name] = vals
    labelfile.close()
    return labels
        

def get_seqs_labels_from_split(filename, seqs, labels):
    splitFile = gzip.open(filename)
    seqs_from_split = []
    labels_from_split = []
    for line in splitFile:
        name = line.strip()
        seqs_from_split.append(seqs[name])
        labels_from_split.append(labels[name])
    splitFile.close()
    return np.array(seqs_from_split), np.array(labels_from_split)

# Functions adapted from Joe Paggi (https://github.com/jpaggi/deepmpra/blob/master/models/kmer_model.py)

BASES = ['A', 'C', 'G', 'T']

def seqs_to_matrix(seqs):
    return np.vstack([map(lambda x: BASES.index(x), seq)
                     for seq in seqs])

def get_kmer_features(seqs, k):
    X = seqs_to_matrix(seqs)
    bases = ['00', '01', '10', '11']
    counts = []
    for seq in X:
        binary_seq = ''.join(map(lambda x: bases[x], seq))
        print binary_seq
        k_vals = np.arange(1, k+1)
        count = np.zeros(np.sum(map(lambda x: 4**x, k_vals)), dtype = np.uint8)
        count_idx = 0
        for k_val in k_vals:
            for i in range(0, len(seq) - k_val + 1):
                count[int(binary_seq[i*2:(i+k_val)*2], 2)] += 1
        counts += [count]
    return np.vstack(counts)

def get_kmer_features_strings(seqs, k):
    feature_matrix = []
    kmers = []
    for i in range(1, k+1):
        kmers += [''.join(kmer) for kmer in list(itertools.product(*[BASES for strlen in range(i)]))]
    for seq in seqs:
        k_vals = np.arange(1, k+1)
        kmer_counts = OrderedDict()
        for kmer in kmers:
            kmer_counts[kmer] = 0
        for i in range(0, len(seq)):
            for kmer_len in range(1, k+1):
                if i + kmer_len > len(seq):
                    continue
                kmer_counts[seq[i : i+kmer_len]] += 1
        feature_matrix.append([kmer_counts[kmer] for kmer in kmer_counts])
    return np.array(feature_matrix)

import time
t0 = time.time()
X_tst1 = get_kmer_features_strings(['ATTGCATG'], 6)
print "new method took %.3f" % (time.time() - t0)
print X_tst1.shape
cnt = 0
for i in range(X_tst1.shape[1]):
    if np.mean(X_tst1[:, i]) != 0:
        cnt += 1
print cnt
print np.sum(X_tst1), np.std(X_tst1)

print

t0 = time.time()
X_tst2 = get_kmer_features(['ATTGCATG'], 6)
print "old method took %.3f" % (time.time() - t0)
print X_tst2.shape
cnt = 0
for i in range(X_tst2.shape[1]):
    if np.mean(X_tst2[:, i]) != 0:
        cnt += 1
print cnt
print np.sum(X_tst2), np.std(X_tst2)

idxs = np.arange(X_tst1.shape[1])[X_tst1[0, :] != X_tst2[0, :]]
print len(idxs)
idxs = idxs[:10]
print idxs
print X_tst1[0][idxs]
print X_tst2[0][idxs]
print np.sum(X_tst1[0][0:4])
print np.sum(X_tst2[0][0:4])
print np.sum(X_tst1[0][4:20])
print np.sum(X_tst2[0][4:20])
print np.sum(X_tst1[0][20:84])
print np.sum(X_tst2[0][20:84])
print np.sum(X_tst1[0][84:340])
print np.sum(X_tst2[0][84:340])
print np.sum(X_tst1[0][340:1364])
print np.sum(X_tst2[0][340:1364])
print np.sum(X_tst1[0][1364:5460])
print np.sum(X_tst2[0][1364:5460])

def get_feature_names(seqs, k, outfile):
    return 0

int('00', 2)

seqsPath = '../features/sequences_sharpr_znormed_jul23.fa.gz'
labelsPath = '../labels/labels_sharpr_znormed_jul23.txt.gz'

trainSplitPath = '../splits/sharpr_znormed_jul23/train_split.txt.gz'
valSplitPath = '../splits/sharpr_znormed_jul23/val_split.txt.gz'
testSplitPath = '../splits/sharpr_znormed_jul23/test_split.txt.gz'

seqs = get_seqs_from_file(seqsPath)
labels = get_labels_from_file(labelsPath)

trainSeqs, trainLabels = get_seqs_labels_from_split(trainSplitPath, seqs, labels)
valSeqs, valLabels = get_seqs_labels_from_split(valSplitPath, seqs, labels)

train_idxs_without_N = [i for (i, seq) in enumerate(trainSeqs) if 'N' not in seq]
trainSeqs = trainSeqs[train_idxs_without_N]
trainLabels = trainLabels[train_idxs_without_N]

val_idxs_without_N = [i for (i, seq) in enumerate(valSeqs) if 'N' not in seq]
valSeqs = valSeqs[val_idxs_without_N]
valLabels = valLabels[val_idxs_without_N]

print trainSeqs.shape, trainLabels.shape, valSeqs.shape, valLabels.shape

import time
from avutils import util

# ntest = len(trainSeqs)
# ntest = 10
k = 6
label_idx = 2 # k562_minp_norm_avg

t0 = time.time()
# X_train = get_kmer_features(trainSeqs[:ntest], k)
# X_train = np.array([np.ravel(util.seq_to_one_hot(seq)) for seq in trainSeqs])
print X_train.shape
y_train = trainLabels[:, label_idx]#[:ntest]
# X_val = get_kmer_features(valSeqs[:ntest], k)
# X_val = np.array([np.ravel(util.seq_to_one_hot(seq)) for seq in valSeqs])
print X_val.shape
y_val = valLabels[:, label_idx]#[:ntest]
print("Creating k-mer features for train/val set took %.3f s" % (time.time() - t0))
# print("Creating one-hot features for train/val set took %.3f s" % (time.time() - t0))

# Features take forever to generate, so just save them to file

# np.savetxt(fname = '../features/xgb/upTo6Mers_train_aug1.txt',
#            X_train,
#            fmt = '%s',
#            delimiter = '\t'
#           )
# np.savetxt(fname = '../features/xgb/upTo6Mers_val_aug1.txt',
#            X_val,
#            fmt = '%s',
#            delimiter = '\t'
#           )

import pandas as pd
import time

# ntest = 10000

t0 = time.time()
X_train = pd.read_csv('../features/xgb/upTo6Mers_train_aug1.txt',
                      dtype = np.uint8,
                      delimiter = '\t',
#                       nrows = ntest,
                      header = None
                     ).values
X_val = pd.read_csv('../features/xgb/upTo6Mers_val_aug1.txt',
                    dtype = np.uint8,
                    delimiter = '\t',
#                     nrows = ntest,
                    header = None
                   ).values
print("Loading XGB k-mer features for %d datapoints took %.3f s" % (len(X_train), time.time() - t0))

from scipy.stats import logistic

sample_weights = logistic.cdf(np.reciprocal(np.abs(trainLabels[:, 0] - trainLabels[:, 1] + 0.1)))
print np.mean(sample_weights[:ntest]), np.std(sample_weights[:ntest])
# sample_weights = np.ones(len(sample_weights))

print X_train.shape # should be (lenX, sum i=1 to k of 4^i (k=6 --> 5460))
print X_val.shape
print y_train.shape
print y_val.shape
print np.max(X_train[0]) # should be >= ceil(145/4) = 37

# eval_set = [(X_val, y_val)]
eval_dmatrix = xgb.DMatrix(data = X_val,
                           label = y_val)
params = {'max_depth': 6, 
          'learning_rate': 0.2,
          'n_estimators': 250,
          'objective': 'reg:linear',
          'silent': 0,
#           'updater': 'grow_gpu',
          'random_state': 0,
          'tree_method': 'exact',
#           'gpu_id': 2
          }         

from scipy.stats import spearmanr

def spearman_eval(y_pred, y_true):
#     print y_true
    y_pred = np.array(y_pred)
    y_true = np.array(y_true.get_label())
#     print(y_pred.shape, y_true.shape)
    return ('spearman', spearmanr(y_pred, y_true)[0])

import os
import random
import string

k=6
model_path = '../model_files/xgb_kmer_sharpr_aug1/'
os.system("mkdir %s" % model_path)
random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
print("On training run %s" % random_id)
model_name = str(k) + 'merModel_record_%s_<>.model' % random_id
# model_name = 'flattenSeq_record_%s_<>.model' % random_id
record_number = 0

# ntest = 20000
lr_decay = 0.996
n_batches = 1 # to avoid GPU memory errors
ti = time.time()
t0 = time.time()
for i in range(n_batches):
    # create xgboost data DMatrix objects
    start_idx = i*len(X_train) / n_batches
    end_idx = (i+1)*len(X_train) / n_batches
    X_batch = X_train[start_idx : end_idx]
    y_batch = y_train[start_idx : end_idx]
    weights_batch = sample_weights[start_idx : end_idx]
    batch_matrix = xgb.DMatrix(data = X_batch,
                               label = y_batch,
                               weight = weights_batch)
    if i == 0:
        bst = xgb.train(params = params,
                        dtrain = batch_matrix,
                        evals = [
                                 (eval_dmatrix, 'val'),
#                                  (batch_matrix, 'train')
                                ],
                        num_boost_round = 1000,
                        feval = spearman_eval,
                        maximize = True,
                        early_stopping_rounds = 5,
                        learning_rates = lambda x, y : params['learning_rate']*(lr_decay**x)
                        )
        bst.save_model(model_path + '/intermediate_models/intermediateModel%d_%s.model' % (i, random_id))
        if n_batches > 1:
            del bst
    else:
        # learning rate decay
        params['learning_rate'] *= lr_decay
        bst = xgb.train(params = params,
                        dtrain = batch_matrix,
                        evals = [
                                 (eval_dmatrix, 'val'),
#                                  (batch_matrix, 'train')
                                ],
                        num_boost_round = 1000,
                        feval = spearman_eval,
                        maximize = True,
                        early_stopping_rounds = 5,
                        xgb_model = model_path + '/intermediate_models/intermediateModel%d_%s.model' % (i-1, random_id)
                        )
        bst.save_model(model_path + '/intermediate_models/intermediateModel%d_%s.model' % (i, random_id))
        if i != n_batches - 1:
            del bst
    print("Training model on batch %d took %.3f s" % (i+1, time.time() - t0))
    t0 = time.time()
bst.save_model(model_path + '/' + model_name.replace('<>', str(record_number)))
# model.fit(X_train[:ntest], 
#           y_train[:ntest], 
#           eval_set = eval_set, 
#           early_stopping_rounds = 8, 
#           eval_metric = 'mae',
#           sample_weight = sample_weights[:ntest])
print("Fitting model took %.3f s" % (time.time() - ti))

y_val_pred = bst.predict(eval_dmatrix)
print y_val_pred.shape
print y_val.shape
print spearmanr(y_val, y_val_pred)

print np.mean(y_train), np.std(y_train)
print np.mean(y_val_pred), np.std(y_val_pred)

from plot_functions import jointplot

jointplot(vals1 = y_val, 
          vals2 = y_val_pred,
          out_pdf = "../plots/sharpr_scatterplots/kmer-xgb/6mer_Unweighted_CpuExact.png",
          show = True,
          cor = 'spearmanr',
          square = True,
          despine = False,
          x_label = "Sharpr Z-Score, Experimental",
          y_label = "Sharpr Z-Score, Predicted",
          figsize = 6,
          ratio = 6,
          dpi = 300,
          color = 'red',
          kde = True,
          bw = 'scott'
         )

bst = xgb.Booster(model_file = '../model_files/xgb_kmer_sharpr_aug1/6merModel_record_7VCRS_0.model')
feature_importances = bst.get_fscore()

importances = np.array([(k, feature_importances[k]) for k in feature_importances])
importances = importances[np.argsort(importances[:, 1])[::-1]]
print importances[:10]




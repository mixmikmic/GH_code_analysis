import os, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition, manifold

get_ipython().magic('matplotlib notebook')

def compute_distance(x,y):
    m = np.empty([len(y),len(x)])
    for i in range(len(y)):
        m[i] = np.abs( x - y[i] ).sum(axis=1)
    m = m / np.linalg.norm(m, axis = 0)
    return np.min(m, axis = 0).sum() / len(x)

def print_percentage(n, t):
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * ((n * 20/t) + 1) , n * 100/t + 1 ))
    if n == t: sys.stdout.write('\n')
    sys.stdout.flush()
    

#feature_dir = '/fileserver/nmec-handwriting/localfeatures/first-pass/'
feature_dir = '/fileserver/nmec-handwriting/localfeatures/nmec_bw_cc_deNNiam_fiel657_min500'

files = os.listdir(feature_dir)
files.sort()

mean_feats = []
all_feats  = []
for i,filename in enumerate(files):
    path = feature_dir + "/" + filename
    #if os.path.isfile(path) and ( '004.' in filename or '007.' in filename):
    if os.path.isfile(path):
        x = np.load(path)
        mean_feats.append( x.mean(axis=0) )
        all_feats.append(x)
    print_percentage(i, len(files))
sys.stdout.write('\n')
sys.stdout.flush()
        
mean_feats = np.array(mean_feats)
all_feats  = np.array(all_feats)
print mean_feats.shape
print all_feats[0].shape

pca = decomposition.PCA(n_components=128)
train = mean_feats
train_reduced = pca.fit_transform(train)

all_reduced = np.array([pca.transform(sample) for sample in all_feats])

metric = []
for i, image in enumerate(all_reduced):
    metricline = [np.array([compute_distance(image, other) for other in all_reduced])]
    metric += metricline
    print_percentage(i, len(all_reduced))

metric = np.array(metric)
F = -metric
np.fill_diagonal(F, -sys.maxint)

soft_correct = 0
hard_correct = 0
total_num = 0

k = 10
g = 8
max_top = 3

for j, i in enumerate(F):
    
    total_num += 1
    topk = i.argsort()[-k:]
    
    if files[j][:6] in (files[index][:6] for index in topk):
        soft_correct += 1
    
    hardsample = list(files[index][3:6] for index in topk[-max_top:])
    if len(set(hardsample)) == 1 and hardsample[0] == files[j][3:6]:
        print "%s matched %s" % (files[j][3:10], hardsample)
        hard_correct += 1

print "%-30s" % ( "-" * 37 )
print "SOFT CRITERIA: Top %d\t= %f" %(k, (soft_correct + 0.0) / total_num)
print "HARD CRITERIA: Top %d\t= %f" %(max_top, (hard_correct + 0.0) / total_num)




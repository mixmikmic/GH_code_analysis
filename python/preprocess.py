get_ipython().system('wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz')
get_ipython().system('gzip -d kddcup.data.gz')

print "opening the data file"
f = open("kddcup.data", "rb").read().split("\n")
data_raw = []
labels_raw = []
n = 0
for line in f[:-2]:
    if n % 10000 == 0: 
        print "Converted " + str(n) + "lines"
    data_raw.append(line.split(",")[:-1])
    labels_raw.append(line.split(",")[-1])
    n += 1

import numpy as np
types = ["n","s","s","s","n","n","n","n","n","n","n","n",
         "n","n","n","n","n","n","n","n","n","n","n","n",
         "n","n","n","n","n","n","n","n","n","n","n","n",
         "n","n","n","n","n"]
nf = len(data_raw[0])
ns = len(data_raw)
data = np.zeros((ns,1))

print "Converting the data to binary vectors"
for i in range(nf):
    print "Converting Feature: " + str(i)
    feats = np.array([data_raw[n][i] for n in range(ns)])
    if types[i] == "n":
        x = np.array([float(x) for x in feats])
        data = np.column_stack((data, x))
    elif types[i] == "s":
        vals = np.unique(feats)
        keys = range(len(vals))
        md = dict([(v,k) for v,k in map(None, vals, keys)])
        new_feat = np.zeros((ns,len(keys)))
        for n in range(ns):
            new_feat[n,keys[md[feats[n]]]] = 1
        data = np.column_stack((data, new_feat))

print "Converting the labels"
vals = np.unique(labels_raw)
keys = range(len(vals))
md = dict([(v,k) for v,k in map(None, vals, keys)])
labels = np.zeros((ns,))
for n in range(ns):
    labels[n] = md[labels_raw[n]]
    
labels = labels.tolist()
data = data.tolist()    

print "Writing Data"
f = open('kdd99_data.txt', 'w')
for row in data: 
    s = ""
    for x in row[:-1]:
        s += str(x) + ","
    s += str(row[-1])+"\n"
    f.write(s)
f.close()

print "Writing Labels"
f = open('kdd99_labels.txt', 'w')
for row in labels: 
    f.write(str(int(row))+"\n")


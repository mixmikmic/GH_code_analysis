# initialise:

get_ipython().magic('run /Users/etc/Projects/201612-lanl-analysis/LANL-analysis/0-lanl-init.ipynb')

from collections import Counter

binsize = 3600

def counter(dat):
    return Counter([t/binsize for t in dat])

def reduce(input, r=400 * 3600/binsize):
    c = input[0]
    d = input[1]
    for x in range(r):
        if x not in d.keys():
            d[x] = 0
    return (c, [[k,v] for (k,v) in d.items() if k < r])

dns_ts_by_comp = dns                .map(lambda x: (x[1], x[0]))                .groupByKey()                .map(lambda x : (x[0], counter(list(x[1]))))                .map(reduce)

p265_ts_by_comp = p265_comps                .map(lambda x : (x[0], counter(list(x[1]))))                .map(reduce)

p265_dns_ts_join = p265_ts_by_comp.join(dns_ts_by_comp)
p265_dns_ts_join.cache()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from scipy import stats 
import numpy as np

def correlate(m1, m2, offset=0):
    z1 = [x[1] for x in m1] # P265
    z2 = [x[1] for x in m2] # DNS
    n = len(z1)
    if offset < 0:
        y1 = z1[-offset:]
        y2 = z2[:(n+offset)]
    else:
        y1 = z1[:(n-offset)]
        y2 = z2[offset:]
    return [stats.pearsonr(y1, y2)[0],
            stats.spearmanr(y1, y2)[0]]

# offset +1 will compare P265 at time 0 to DNS at time +1 hour

plt.figure(figsize=(12,6))
ax = [plt.subplot2grid((1,2), (0, 0), colspan=1),
      plt.subplot2grid((1,2), (0, 1), colspan=1)]

for i in [0,1]:
    offset = i # or add translates e.g. offset = i-1
    
    p265_dns_ts_correlation = p265_dns_ts_join                            .map(lambda x: (x[0], correlate(x[1][0], x[1][1], offset=offset)))                            .filter(lambda x: not (np.isnan(x[1][0]) or np.isnan(x[1][1])))                            .collect()
    pearson = [x[1][0] for x in p265_dns_ts_correlation]
    spearman = [x[1][1] for x in p265_dns_ts_correlation]
    ax[i].set_xlim(xmax = 1, xmin = -0.2)
    ax[i].set_ylim(ymax = 1, ymin = -0.2)
    ax[i].set_xlabel("Pearson correlation")
    ax[i].set_ylabel("Spearman correlation")
    ax[i].set_title("Offset %d hour" % offset)
    ax[i].plot([0,0], [-0.2, 1], color='grey', linestyle='-')
    ax[i].plot([-0.2, 1], [0,0], color='grey', linestyle='-')
    ax[i].scatter(pearson, spearman, s=5, alpha=0.3)

plt.show()

import json

filename = lanl_path + "computer_p265_hh_dict.json"
fp = open(filename, 'r')
computer_p265_hh_hour = json.load(fp)
fp.close()

comps = [x[0] for x in p265_dns_ts_correlation]
subset = [i for i in range(len(comps)) if computer_p265_hh_hour[comps[i]] < 300]

p265_dns_ts_correlation = p265_dns_ts_join                            .map(lambda x: (x[0], correlate(x[1][0], x[1][1], offset=1)))                            .filter(lambda x: not (np.isnan(x[1][0]) or np.isnan(x[1][1])))                            .collect()

plt.figure(figsize=(15,10))
plt.xlim(xmax = 1, xmin = -0.2)
plt.ylim(ymax = 1, ymin = -0.2)
plt.xlabel("Pearson correlation")
plt.ylabel("Spearman correlation")
plt.plot([0,0], [-0.2, 1], color='grey', linestyle='-')
plt.plot([-0.2, 1], [0,0], color='grey', linestyle='-')
plt.scatter(pearson, spearman, s=20, c='grey', alpha=0.1)
plt.scatter([pearson[i] for i in subset], 
            [spearman[i] for i in subset], 
            s=30, c='red', alpha=1.0)
plt.show()

p265_dns_ts_join.unpersist()


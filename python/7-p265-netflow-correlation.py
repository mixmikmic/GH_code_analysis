# initialise:

get_ipython().magic('run /Users/etc/Projects/201612-lanl-analysis/LANL-analysis/0-lanl-init.ipynb')

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

from math import log
from collections import Counter
from __future__ import print_function

binsize = 3600
maxhour = 400

def counter(dat, binsize=3600):
    return Counter([t/binsize for t in dat])

def reduce(input, binsize=3600, maxhour=400):
    c = input[0]
    d = input[1]
    r = maxhour * 3600/binsize
    for x in range(r):
        if x not in d.keys():
            d[x] = 0
    return [([c,k], v) for (k,v) in d.items() if k < r]

p265_comphour_activity = p265_comps                        .map(lambda x : (x[0], counter(list(x[1]), binsize=binsize)))                        .flatMap(lambda x: reduce(x, binsize=binsize, maxhour=maxhour))

# make a list of P265-heavy-hitter computer-hours (up to maxhour by construction):

busyness = 200 * binsize/3600

hh_comphours = p265_comphour_activity                .filter(lambda x: x[1] > busyness)                .map(lambda x: x[0])                .collect()
hh_comphours = set([(c,h) for [c,h] in hh_comphours])
len(hh_comphours)

def is_heavy_hitter(f):
    src_comp = f[2]
    dst_comp = f[4]
    hour = f[0]/binsize
    if ((src_comp, hour) in hh_comphours) or ((dst_comp, hour) in hh_comphours):
        return True
    return False

p265_hh_flows = flows.filter(is_heavy_hitter)
p265_hh_flows.cache()

# protocol distribution in these computer-hours:

p265_hh_protocol_hist = p265_hh_flows                        .map(lambda x: x[6])                        .countByValue()

n = sum(p265_hh_protocol_hist.values())
for p in sorted(p265_hh_protocol_hist.keys()):
    print('{0:2d}: {1:10d}  {2:2.3}%'            .format(p, p265_hh_protocol_hist[p], 100.0*p265_hh_protocol_hist[p]/n))

# computers:

computer_edgelist = p265_hh_flows.map(lambda x: (x[2], x[4])).countByValue()

src_set = set([x[0][0] for x in computer_edgelist.items()])
dst_set = set([x[0][1] for x in computer_edgelist.items()])

print("Sources:      ", len(src_set))
print("Destinations: ", len(dst_set))
print("Union:        ", len(src_set.union(dst_set)))
print("Intersection: ", len(src_set.intersection(dst_set)))










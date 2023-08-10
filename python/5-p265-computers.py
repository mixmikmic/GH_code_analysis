# initialise:

get_ipython().magic('run /Users/etc/Projects/201612-lanl-analysis/LANL-analysis/0-lanl-init.ipynb')

p265_comps = proc_p265            .filter(lambda x: x[4]=='Start')            .map(lambda x: (x[3], x[0]))            .groupByKey()

# full frequency table of P265 starts by computer and time-bin up to hour 400:

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
    return [([c,k], v) for (k,v) in d.items() if k < r]
    
p265_comps_ts_sparse = p265_comps.map(lambda x : (x[0], counter(list(x[1]))))
p265_comps_ts_full = p265_comps_ts_sparse.flatMap(reduce)

p265_comps_ts_sparse.cache()
p265_comps_ts_full.cache()

p265_comps_ts_sparse.take(3)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

busyness = 300
start_hour = 338
end_hour = 345

for hour in range(start_hour, end_hour):
    hour_set = p265_comps_ts_sparse.filter(lambda x: x[1][hour] > 0).collect()
    vol_dist = [cnt[hour] for (cmp, cnt) in hour_set]
    nr_comp = len([cnt[hour] for (cmp, cnt) in hour_set if cnt[hour] > busyness])
    plt.hist(vol_dist, bins=20)
    plt.title("Hour %d: nr computers with >%d 'starts' = %d" % (hour, busyness, nr_comp))
    plt.show()

# distribution of frequency table entries up to hour 330:

from math import log 

upper = 300

early_freq = p265_comps_ts_full            .filter(lambda x: x[0][1] <= upper)            .map(lambda x: x[1])            .countByValue()

fig = plt.figure(figsize=(20,8))
size = [100*log(1+y, 10) for y in early_freq.values()]
plt.xlim(xmin=7e-02, xmax=1e04)
plt.ylim(ymin=5e-01, ymax=1e07)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Nr P265 starts")
plt.ylabel("Frequency")
plt.title("Distribution of 'nr P265 starts' per computer-hour")
plt.scatter([x+0.1 for x in early_freq.keys()], early_freq.values(), s=size, alpha=0.3)
plt.show()

"""
Construct set of heavy hitter computers (for some busy-ness threshold) 
in each time bin, and inspect time series of the size of this set:
"""

busyness = 200 * binsize/3600

heavy_hitter_sets = p265_comps_ts_full                    .filter(lambda x: x[1] > busyness)                    .map(lambda x: (x[0][1], x[0][0]))                    .groupByKey()                    .map(lambda x: (x[0], set(x[1])))                    .collect()

x = [c[0] for c in heavy_hitter_sets]
y = [len(c[1]) for c in heavy_hitter_sets]

plt.figure(figsize=(20,8))
plt.xlabel("Hour")
plt.ylabel("Nr P265 heavy hitters")
plt.title("Time series of nr P265 heavy hitter computers")
plt.yscale("log")
plt.plot(x,y)
plt.show()

lookup = dict(heavy_hitter_sets)
heavy_hitter_cum_set = dict()

T = 400 * 3600/binsize # time range

heavy_hitter_cum_set[0] = set([])
for t in range(1,T):
    if t not in lookup.keys():
        lookup[t] = set([])
    heavy_hitter_cum_set[t] = heavy_hitter_cum_set[t-1].union(lookup[t])

x = heavy_hitter_cum_set.keys()
heavy_hitter_cum_count = [len(heavy_hitter_cum_set[t]) for t in x]

plt.figure(figsize=(20,8))
plt.xlabel("Hour")
plt.ylabel("Nr heavy hitters")
plt.title("Cumulative number of P265 heavy hitter computers")
plt.stem(x, heavy_hitter_cum_count)
plt.show()

import json

computers = procs.map(lambda x: x[3]).distinct().collect()
computers = set(computers)
computer_p265_hh_hour = dict([(c, 2000) for c in computers])

for h in heavy_hitter_cum_set.keys():
    for c in heavy_hitter_cum_set[h]:
        if computer_p265_hh_hour[c] > h:
            computer_p265_hh_hour[c] = h
            
filename = lanl_path + "computer_p265_hh_dict.json"
fp = open(filename, 'w')
json.dump(computer_p265_hh_hour, fp)
fp.close()

p265_comps_ts_sparse.unpersist()
p265_comps_ts_full.unpersist()


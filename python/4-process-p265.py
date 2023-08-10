# initialise:

get_ipython().magic('run /Users/etc/Projects/201612-lanl-analysis/LANL-analysis/0-lanl-init.ipynb')

proc_p265 = procs.filter(lambda x: x[4] == 'P265').map(lambda x: x[0:4] + [x[5]])
proc_p265.cache()

from __future__ import print_function

nr_cmps = proc_p265.map(lambda x: x[3]).distinct().count()

print("Nr P265 events:", proc_p265.count())
print("Nr computers:", nr_cmps)

from collections import Counter

binsize = 3600

def mapper(x):
    return (x[0]/binsize, x[1:5])

"""
 mapper will output collections of records like:
 (timebin, ['C1001$', 'DOM1', 'C1001', 'Start'])
"""

def reducer(dat):
    n = len(dat)
    user = dict(Counter([x[0] for x in dat]))
    domn = dict(Counter([x[1] for x in dat]))
    comp = dict(Counter([x[2] for x in dat]))
    nr_user = len(user.keys())
    nr_comp = len(comp.keys())
    nr_domn = len(domn.keys())
    cmax_prop = max(comp.values())/float(n)
    cmax = [k for (k,v) in comp.items() if v == max(comp.values())][0]
    return [n, nr_user, nr_comp, nr_domn, cmax_prop, cmax]

# map-reduce job:
proc_p265_ts = proc_p265.map(mapper)                    .groupByKey()                    .map(lambda x : (x[0], reducer(list(x[1]))))                    .collect()
            
proc_p265_ts_dict = dict(proc_p265_ts)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

time = sorted(proc_p265_ts_dict.keys())
nr_events = [proc_p265_ts_dict[t][0] for t in time]
nr_user = [proc_p265_ts_dict[t][1] for t in time]
nr_comp = [proc_p265_ts_dict[t][2] for t in time]
nr_domn = [proc_p265_ts_dict[t][3] for t in time]

plt.figure(figsize=(20,8))
ts1, = plt.semilogy(time, nr_events, label='nr events')
ts2, = plt.semilogy(time, nr_user, label='nr users')
ts3, = plt.semilogy(time, nr_comp, label='nr computers')
ts4, = plt.semilogy(time, nr_domn, label='nr domains')
plt.legend(handles=[ts1, ts2, ts3, ts4], loc='upper left')
plt.show()

proc_p265.unpersist()


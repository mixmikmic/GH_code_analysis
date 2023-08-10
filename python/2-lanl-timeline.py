lanl_path = "/Users/etc/data/LANL/"

dnsfile = lanl_path + "dns.txt"
flowfile = lanl_path + "flows.txt"
procfile = lanl_path + "proc.txt"

def flow_parser(x): return  [int(x[0]), int(x[1])]                 + [str(x[i]) for i in range(2,6)]                 + [int(x[i]) for i in range(6,9)]

def dns_parser(x): return [int(x[0])]                + [str(x[i]) for i in range(1,len(x))]

def proc_parser(x):
    n = len(x)
    return  [int(x[0])] + str(x[1]).split('@') + [str(x[i]) for i in range(2,n)]

procs = sc.textFile(procfile)    .map(lambda line: proc_parser(line.split(',')))
flows = sc.textFile(flowfile)    .map(lambda line: flow_parser(line.split(',')))
dns = sc.textFile(dnsfile)    .map(lambda line: dns_parser(line.split(',')))

proc_times = procs.map(lambda x: x[0])
flow_times = flows.map(lambda x: x[0])
dns_times = dns.map(lambda x: x[0])

# these RDDs are called at least twice in this notebook:
proc_times.cache()
flow_times.cache()
dns_times.cache()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def compute_series(binsize = 3600, timerange=1e09):

    proc_count = proc_times.filter(lambda x: x<timerange).map(lambda x: x/binsize).countByValue()
    flow_count = flow_times.filter(lambda x: x<timerange).map(lambda x: x/binsize).countByValue()
    dns_count = dns_times.filter(lambda x: x<timerange).map(lambda x: x/binsize).countByValue()
    
    return [proc_count, flow_count, dns_count]

def show_series(pc, fc, dc):
    
    plt.figure(figsize=(20,8))
    ts1, = plt.semilogy(pc.keys(), pc.values(), label='processes')
    ts2, = plt.semilogy(fc.keys(), fc.values(), label='flows')
    ts3, = plt.semilogy(dc.keys(), dc.values(), label='DNS requests')
    plt.legend(handles=[ts1, ts2, ts3], loc='lower left')
    plt.show()

[pc, fc, dc] = compute_series()
show_series(pc, fc, dc)

# the initial fortnight in 5-minute bins:

tr = 370*3600

[pc_init, fc_init, dc_init] = compute_series(binsize = 300, timerange = tr)
show_series(pc_init, fc_init, dc_init)

print("Time range:", tr/3600.0/24.0, "days")

# examine correlation coefficients for lagging up to one week:

from scipy import stats

x = pc_init.values()
y = fc_init.values()
z = dc_init.values()

f = stats.spearmanr  # monotonic correlation
#f = stats.pearsonr  # linear correlation

n = min(len(x), len(y), len(z))
base = range(12*24*7)

corr_xy = [f(x[:(n - offset)],y[offset:n])[0] for offset in base]
corr_yz = [f(y[:(n - offset)],z[offset:n])[0] for offset in base]
corr_zx = [f(z[:(n - offset)],x[offset:n])[0] for offset in base]

plt.figure(figsize=(20,8))
zero = [0 for i in base]
plt.plot(base, zero, c='grey')
ts1, = plt.plot(corr_xy, label='process vs flows')
ts2, = plt.plot(corr_yz, label='flows vs DNS requests')
ts3, = plt.plot(corr_zx, label='DNS requests vs processes')
plt.title("Lagged correlation")
plt.legend(handles=[ts1, ts2, ts3], loc='lower right')
plt.show()

# increase resolution and zoom in plus/minus 6-hour lags:

binsize = 300
base = range(-6 * 3600/binsize, 6 * 3600/binsize)

[pc_init, fc_init, dc_init] = compute_series(binsize=binsize, timerange = tr)
x = pc_init.values()
y = fc_init.values()
z = dc_init.values()

corr_xy = []
corr_yz = []
corr_zx = []

for offset in base:
    """
    to allow negative as well as positive offsets
    """
    r = [i for i in zip(range(0, n-offset), range(offset,n)) if i[0]<n and i[1]>=0]
    corr_xy += [f([x[i[0]] for i in r], [y[i[1]] for i in r])[0]]
    corr_yz += [f([y[i[0]] for i in r], [z[i[1]] for i in r])[0]]
    corr_zx += [f([z[i[0]] for i in r], [x[i[1]] for i in r])[0]]

xy_shift = -20 * 60.0/binsize
yz_shift = 80 * 60.0/binsize
zx_shift = -60 * 60.0/binsize

plt.figure(figsize=(20,8))
zero = [0 for i in base]
ymin = min(corr_xy + corr_yz + corr_zx)
ymax = max(corr_xy + corr_yz + corr_zx)
plt.plot(base, zero, c='grey')
plt.plot([xy_shift, xy_shift], [ymin, ymax], color='blue', linestyle='dashed')
plt.plot([yz_shift, yz_shift], [ymin, ymax], color='green', linestyle='dashed')
plt.plot([zx_shift, zx_shift], [ymin, ymax], color='red', linestyle='dashed')
ts1, = plt.plot(base, corr_xy, label='process vs flows')
ts2, = plt.plot(base, corr_yz, label='flows vs DNS requests')
ts3, = plt.plot(base, corr_zx, label='DNS requests vs processes')
plt.title("Lagged correlation")
plt.legend(handles=[ts1, ts2, ts3], loc='upper right')
plt.show()

proc_times.unpersist()
flow_times.unpersist()
dns_times.unpersist()


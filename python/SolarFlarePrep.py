# plotting
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import seaborn as sns

# numerics
import numpy as np

# FITS reading
from astropy.io import fits

# power spectral stuff
from stingray import Lightcurve, Powerspectrum

# directory with the data
datadir = "/Users/danielahuppenkothen/work/data/solarflares/bn121022782/current/"

# trigger ID
burstid = "121022782"

# detectors
det = [0, 1, 3, 5]

# construct file name
f = "glg_ctime_n" + str(det[0]) + "_bn" + str(burstid) + "_v00.pha"

hdulist = fits.open(datadir+f)

hdulist.info()

hdulist[0].header

hdulist[1].data

data = hdulist[2].data

data

counts = data.field("COUNTS")
time = data.field("TIME")
time = time - time[0]

counts = np.sum(counts[:,:2], axis=1)

dt = np.diff(time)

plt.figure(figsize=(10, 6))
plt.plot(time, counts)
plt.plot(time[:-1], dt*10000)

def load_data(datadir, burstid, detectors):
    
    counts_all = []
    for d in detectors:
        
        # construct file name
        f = "glg_ctime_n" + str(d) + "_bn" + str(burstid) + "_v00.pha"

        # open data file
        hdulist = fits.open(datadir+f)
        
        data = hdulist[2].data
        counts = data.field("COUNTS")
        time = data.field("TIME")
        time = time - time[0]
        dt = np.diff(time)
        
        hdulist.close()
                
        counts_all.append(counts[:,1])
        
    counts_all = np.array(counts_all)
    counts_all = np.sum(counts_all, axis=0)
    
    return time, dt, counts_all

time, dt, counts = load_data(datadir, burstid, detectors=det)

plt.figure(figsize=(12, 6))
plt.plot(time-time[0], counts)

start_ind = time.searchsorted(670)

start_ind

time = time[start_ind:]
counts = counts[start_ind:]
dt = dt[start_ind:]

np.where(dt < 0.1)[0]

dt_min_ind = np.where(dt < 0.1)[0][1]
dt_max_ind = np.where(dt < 0.1)[0][-2]

len(time[dt_max_ind:])

lc1 = Lightcurve(time[:dt_min_ind-1], counts[:dt_min_ind-1])
lc2 = Lightcurve(time[dt_min_ind+1:dt_max_ind-5], counts[dt_min_ind+1:dt_max_ind-5])
lc3 = Lightcurve(time[dt_max_ind+3:-15], counts[dt_max_ind+3:-15])

plt.figure(figsize=(12,6))
plt.plot(lc1.time, lc1.countrate)
plt.plot(lc2.time, lc2.countrate)
plt.plot(lc3.time, lc3.countrate)

lc1bin = lc1.rebin(1.0)
lc2bin = lc2.rebin(1.0)
lc3bin = lc3.rebin(1.0)

plt.figure(figsize=(12,6))
plt.plot(lc1bin.time, lc1bin.counts)
#plt.plot(lc2.time, lc2.countrate)
plt.plot(lc2bin.time, lc2bin.countrate)
plt.plot(lc3bin.time, lc3bin.counts)

lc = lc1bin.join(lc2bin)

lc = lc.join(lc3bin)

plt.figure()
plt.plot(lc.time, lc.counts)

lc.gti = np.array([[lc.gti[0,0], lc.gti[-1,-1]]])

ps = Powerspectrum(lc)

plt.figure()
plt.loglog(ps.freq, ps.power)

np.savetxt("121022782_ctime_lc.txt", np.array([lc.time, lc.counts]).T)

# directory with the data
datadir = "/Users/danielahuppenkothen/work/data/solarflares/bn120704187/current/"

# trigger ID
burstid = "120704187"

# detectors
det = [5, 3, 1, 4]

time, dt, counts = load_data(datadir, burstid, detectors=det)

time = time - time[0]

plt.figure(figsize=(12,6))
plt.plot(time, counts)

start_ind = time.searchsorted(300)
time = time[start_ind:]
counts = counts[start_ind:]
dt = dt[start_ind:]

dt_min_ind = np.where(dt < 0.1)[0][1]
dt_max_ind = np.where(dt < 0.1)[0][-1]

lc1 = Lightcurve(time[:dt_min_ind-3], counts[:dt_min_ind-3])
lc2 = Lightcurve(time[dt_min_ind:dt_max_ind-5], counts[dt_min_ind:dt_max_ind-5])
lc3 = Lightcurve(time[dt_max_ind+1:], counts[dt_max_ind+1:])

lc1.counts[-10:]


plt.figure(figsize=(12,6))
plt.plot(lc1.time, lc1.countrate)
plt.plot(lc2.time, lc2.countrate)
plt.plot(lc3.time, lc3.countrate)

lc1bin = lc1.rebin(1.0)
lc2bin = lc2.rebin(1.0)
lc3bin = lc3.rebin(1.0)

plt.figure(figsize=(12,6))
plt.plot(lc1bin.time, lc1bin.counts)
#plt.plot(lc2.time, lc2.countrate)
plt.plot(lc2bin.time, lc2bin.countrate)
plt.plot(lc3bin.time, lc3bin.counts)

lc = lc1bin.join(lc2bin)
lc = lc.join(lc3bin)

lc.gti = np.array([[lc.gti[0,0], lc.gti[-1,-1]]])

ps = Powerspectrum(lc)

plt.figure()
plt.loglog(ps.freq, ps.power, linestyle="steps-mid")

np.savetxt("120704187_ctime_lc.txt", np.array([lc.time, lc.counts]).T)


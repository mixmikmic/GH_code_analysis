# Import modules.
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
from lsst.sims.maf.metricBundles import MetricBundle, MetricBundleGroup, makeBundlesDictFromList

sqliteDir = '/Users/yoachim/Scratch/Opsim_sqlites/'
#runName = 'enigma_1189'
runName = 'ops2_1094'
#runName = 'enigma_1258'
opsdb = db.OpsimDatabase(sqliteDir+runName + '_sqlite.db')

fields = opsdb.fetchFieldsFromFieldTable()

sql = 'filter = "g" or filter="r" or filter="i" or filter="z"'
data = opsdb.fetchMetricData(['night','fieldID'], sql)

data['night']

nightBins = np.arange(data['night'].min(), data['night'].max() +2) -0.5
fIDBins = np.arange(data['fieldID'].min(), data['fieldID'].max() +2) -0.5

H,xe,ye = np.histogram2d(data['night'], data['fieldID'], bins=[nightBins,fIDBins])

plt.imshow(H, vmin=1, vmax=5, cmap='gray_r')
plt.xlabel('night')
plt.ylabel('fieldID')

bins = np.arange(0.5, H.max()+1.5)

bins = np.arange(0.5, H.max()+2)
finalHist, finalBins, ack = plt.hist(H.ravel(), bins)
plt.xlim([0,10])
plt.xlabel('N visits per field per night')
plt.ylabel('# of instances')

print data.size
print np.sum(H)
print np.sum(finalHist)

x = np.arange(1,finalHist.size+1)
plt.plot(x, finalHist*x, 'ko')
plt.xlim([0,10])
plt.ylabel('Number of visits in')
plt.xlabel('N visits per field per night')
print np.sum(finalHist*x)

# Same, now normalized
plt.plot(x, finalHist*x/np.sum(H), 'ko')
plt.xlim([0,10])
plt.ylabel('Number of visits in')
plt.xlabel('N visits per field per night')






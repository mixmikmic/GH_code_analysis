import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
plt.style.use('ggplot')
import joblib
import glob
import os
import SNclass
reload(SNclass);

import george

george.__version__

dirname = os.environ['SNPhotCC_dir']
print(dirname)

N = 300
for n, f in enumerate(sorted(glob.glob(dirname + '/DES_*.DAT.gz'))[:N]):
    obs, metadata = SNclass.SNPhotCC_Parser(f)
    #print n, snid, sn_type, sim_type, sim_z, obs.shape
    if n == 130:  # stop at a nice big lightcurve
    #if n == 0:
        break

print f
obs.head()

metadata

#reload(SNclass)
snc = SNclass.SNclass(os.path.join(dirname, 'DES_SN005754.DAT.gz'))
print snc.metadata['sim_type'], snc.metadata['hostz'], snc.metadata['filename']
snc.plot()

snc.plot(normalized=True)

#reload(SNclass)
get_ipython().magic('timeit SNclass.SNPhot_fitter(obs)')

import cPickle
tmp = cPickle.dumps(snc)  # approximate size of objects
len(tmp) * 30000 / 1024 / 1024   # what if we were to read them all in and fit them all? (in megabytes)

def progress(x, ind, every):
    if (ind % every) == 0:
        print ind, x
    return x

reload(SNclass)
N = 3000
lcdata = [SNclass.SNclass(progress(f, i, 300))             for i,f in enumerate(sorted(glob.glob('SIMGEN_PUBLIC_DES/DES_*.DAT.gz'))[:N])]

reload(SNclass)
f = dirname +'/DES_SN109810.DAT.gz'
snc = SNclass.SNclass(f)
print snc.metadata['sim_type'], snc.metadata['hostz'], snc.metadata['filename']
snc.plot()

reload(SNclass)
snc = SNclass.SNclass(f)
snc.normalize()
snc.plot(normalized=True)


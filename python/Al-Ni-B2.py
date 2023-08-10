import datetime
import time
import os
current_time = datetime.datetime.now()
print(current_time)
timestamp = str(time.mktime(current_time.timetuple()))
os.makedirs('output-results/{0}/{1}/'.format(PHASE, timestamp))

SYSTEM = 'Al-Ni'
PHASE = 'BCC_B2'

get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('bmh')
from pycalphad.fitting import build_pymc_model, plot_results
from pycalphad import Database
import pymc
import numpy as np

import glob
input_database = Database(SYSTEM+'-start.tdb')
dataset_names = sorted(glob.glob('input-json/'+PHASE+'/*.json'))

LB2ALVA_A = pymc.Normal('LB2ALVA_A', mu=0, tau=1./200000**2)
B2ALVA_A = pymc.Normal('B2ALVA_A', mu=0, tau=1./100000**2)
B2ALVA_B = pymc.Normal('B2ALVA_B', mu=0, tau=1./30**2)
LB2NIVA_A = pymc.Normal('LB2NIVA_A', mu=0, tau=1./100000**2)
LB2NIVA_B = pymc.Normal('LB2NIVA_B', mu=0, tau=1./30**2)
B2NIVA_A = pymc.Normal('B2NIVA_A', mu=0, tau=1./100000**2)
B2NIVA_B = pymc.Normal('B2NIVA_B', mu=0, tau=1./30**2)
LB2ALNI_A = pymc.Normal('LB2ALNI_A', mu=0, tau=1./100000**2)
LB2ALNI_B = pymc.Normal('LB2ALNI_B', mu=0, tau=1./30**2)
B2ALNI_A = pymc.Normal('B2ALNI_A', mu=0, tau=1./100000**2)
B2ALNI_B = pymc.Normal('B2ALNI_B', mu=0, tau=1./30**2)
params = [LB2ALVA_A, B2ALVA_A, B2ALVA_B, LB2NIVA_A,
          LB2NIVA_B, B2NIVA_A, B2NIVA_B, LB2ALNI_A,
          LB2ALNI_B, B2ALNI_A, B2ALNI_B]
get_ipython().magic('time mod, datasets = build_pymc_model(input_database, dataset_names, params)')

MDL = pymc.MCMC(mod, db='hdf5', dbname='output-traces/'+SYSTEM+'-'+PHASE+'-'+timestamp+'.hdf5',
                dbcomplevel=4, dbcomplib='bzip2')

get_ipython().magic('time MDL.sample(iter=20000, burn=15000, burn_till_tuned=False, thin=2, progress_bar=True)')

import pymc
try:
    desired_timestamp = timestamp
except:
    desired_timestamp = '1448917146.0'
timestamp = desired_timestamp
desired_system = 'Al-Ni'
desired_phase = 'BCC_B2'
db = pymc.database.hdf5.load('output-traces/'+desired_system+'-'+desired_phase+'-'+desired_timestamp+'.hdf5')
MDL = pymc.MCMC(mod, db=db)

pymc.Matplot.autocorrelation(MDL, format='eps', path='output-results/{0}/{1}/'.format(PHASE, timestamp))

pymc.Matplot.summary_plot(params, format='eps', name=SYSTEM+'-'+PHASE+'-parameters',
                          path='output-results/{0}/{1}/'.format(PHASE, timestamp))

from corner import corner
from itertools import chain
data = np.vstack(chain([i.trace() for i in params])).T
figure = corner(data, labels=[str(i) for i in params],
                quantiles=[0.16, 0.5, 0.84],
                truths=[150000, 10000, -1, -64024.38, 26.49419, 162397.3, -27.40575,
                        -64024.38, 26.49419, 162397.3, -27.40575, -52440.88, 11.30117,
                        -152397.3, 26.40575],
                show_titles=True, title_args={"fontsize": 40})
figure.savefig('output-results/{0}/{1}/{0}-{2}-corner.pdf'.format(PHASE, timestamp, SYSTEM))
figure.show()

get_ipython().magic('matplotlib inline')
#figures = list(plot_results(datasets, params))
#figures[0].savefig('output-results/{0}/{1}/{0}-{2}-heatcapacity.eps'.format(PHASE, timestamp, SYSTEM))

MDL.write_csv('output-results/{0}/{1}/{0}-{2}-param-summary.csv'.format(PHASE, timestamp, SYSTEM))
MDL.summary()




timestamp = '1448906809.0'
SYSTEM = 'Al-Ni'
PHASE = 'FCC_L12'
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('bmh')
from pycalphad.fitting import build_pymc_model, plot_results
from pycalphad import Database
import tables
import numpy as np
from collections import OrderedDict

data = tables.open_file('output-traces/'+SYSTEM+'-'+PHASE+'-'+timestamp+'.hdf5', mode='r')
data_dict = OrderedDict()
parnames = [x for x in data.root.chain0.PyMCsamples.colnames if not x.startswith('Metropolis') and x != 'deviance']
for param in sorted(parnames):
    data_dict[param] = np.asarray(data.root.chain0.PyMCsamples.read(field=param), dtype='float')

def autocorr(x):
    x = (x - x.mean()) / np.linalg.norm(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]
for param, trace in data_dict.items():
    figure = plt.figure()
    figure.gca().plot(autocorr(trace))
    figure.gca().set_title(param+' Autocorrelation')
plt.show()

from corner import corner
from itertools import chain
data = np.vstack(chain([i for i in data_dict.values()])).T
figure = corner(data, labels=list(data_dict.keys()),
                quantiles=[0.16, 0.5, 0.84],
                truths=[-1622407.75, 16.212965, 73417.798, -34.914168, 33471.014, -9.8373558,
                        -30758.01, 10.25267, 0.52, -1112, 1745, -22212.8931, 4.39570389],
                show_titles=True, title_args={"fontsize": 40}, rasterized=True)
figure.savefig('output-results/{0}/{1}/{0}-{2}-corner.pdf'.format(PHASE, timestamp, SYSTEM))
figure.show()

get_ipython().magic('matplotlib inline')
from pycalphad.fitting import plot_results, setup_dataset, Dataset
import glob
input_database = Database(SYSTEM+'-start.tdb')
dataset_names = sorted(glob.glob('input-json/'+PHASE+'/*.json'))
datasets = []
for fname in dataset_names:
    with open(fname) as file_:
        datasets.append(Dataset(*setup_dataset(file_, input_database, parnames, mode='numpy')))
figures = list(plot_results(input_database, datasets, data_dict,
                            databases={'Dupin 2001 (CALPHAD)': Database('Al-Ni-Dupin-2001.tdb')}))
for idx, fig in enumerate(figures):
    fig.savefig('output-results/{0}/{1}/{0}-{2}-Figure{3}.eps'.format(PHASE, timestamp, SYSTEM, idx+1))

from pymc.database.base import batchsd
from pymc import utils
import csv
with open('output-results/{0}/{1}/{0}-{2}-param-summary.csv'.format(PHASE, timestamp, SYSTEM), 'w') as csvfile:
    fieldnames = ['Parameter', 'Mean', 'SD', 'Lower 95% HPD', 'Upper 95% HPD',
                  'MC error', 'q2.5', 'q25', 'q50', 'q75', 'q97.5']
    writer = csv.DictWriter(csvfile, fieldnames)
    writer.writeheader()
    for parname, data in data_dict.items():
        qxx = utils.quantiles(data, qlist=(2.5, 25, 50, 75, 97.5))
        q2d5, q25, q50, q75, q975 = qxx[2.5], qxx[25], qxx[50], qxx[75], qxx[97.5]
        lower_hpd, upper_hpd = utils.hpd(data, 0.05)
        row = {
                'Parameter': parname,
                'Mean': data.mean(0),
                'SD': data.std(0),
                'Lower 95% HPD': lower_hpd,
                'Upper 95% HPD': upper_hpd,
                'MC error': batchsd(data, min(len(data), 100)),
                'q2.5': q2d5, 'q25': q25, 'q50': q50, 'q75': q75, 'q97.5': q975
              }
        writer.writerow(row)






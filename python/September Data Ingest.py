import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('nbagg')
from matplotlib import style
style.use('ggplot')
import matplotlib.pyplot as plt
from pandeia.engine.perform_calculation import perform_calculation
from pandeia.engine.io_utils import read_json, write_json
from pandeia.engine.utils import merge_data
from astropy.io import fits
from astropy.table import Table
from copy import deepcopy

def strip_meta(d):
    if isinstance(d, dict):
        if 'meta' in d:
            meta_data = d.pop('meta')
        for k in d.keys():
            strip_meta(d[k])

get_ipython().magic('cd /Users/pickering/STScI/smite/pandeia_data/devtools/delivered')

data = read_json("jwst_nircam_configuration.json")
strip_meta(data)
data

data.keys()

for k, v in data.items():
    print(k)

os.environ['pandeia_refdata']

"september"[0:-1]

miri_filters = [
        "f1065c",
        "f1140c",
        "f1550c",
        "f2300c",
        "f560w",
        "f770w",
        "f1000w",
        "f1130w",
        "f1280w",
        "f1500w",
        "f1800w",
        "f2100w",
        "f2550w"
]
niriss_filters = [
        "f090w",
        "f115w",
        "f140m",
        "f150w",
        "f158m",
        "f200w",
        "f277w",
        "f356w",
        "f380m",
        "f430m",
        "f444w",
        "f480m"
    ]
nirspec_filters = [
        "f070lp",
        "f100lp",
        "f170lp",
        "f290lp",
        "f110w",
        "f140x",
        "clear"
    ]

filter_config = {}
for f in nirspec_filters:
    filter_config[f] = {}
    filter_config[f]["display_string"] = f.upper()

filter_config

write_json(filter_config, "blah.json")

get_ipython().magic('pinfo filter_config.update')

for i in filter_config:
    print(i)

data = Table.read("../../jwst/niriss/wavepix/jwst_niriss_soss-256-ord1_trace.fits")
#old_data = Table.read("../../../master/pandeia_data/jwst/nircam/dispersion/jwst_nircam_disp_20151210124744.fits")
data2 = Table.read("../../jwst/niriss/wavepix/jwst_niriss_soss-96-ord1_trace.fits")
np.mean(data['TRACE'] - data2['TRACE'])

ord2 = Table.read("../../jwst/niriss/wavepix/jwst_niriss_soss-256-ord2_trace.fits")
ord2['TRACE'] -= 70.0
ord2.write("../../jwst/niriss/wavepix/jwst_niriss_soss-96-ord2_trace.fits")

ord3 = Table.read("../../jwst/niriss/wavepix/jwst_niriss_soss-256-ord3_trace.fits")
ord3['TRACE'] -= 70.0
ord3.write("../../jwst/niriss/wavepix/jwst_niriss_soss-96-ord3_trace.fits")

f = plt.figure()
plt.plot(data['WAVELENGTH'], data['THROUGHPUT'])
plt.plot(old_data['WAVELENGTH'], old_data['THROUGHPUT'])
plt.show()

data

file = "/Users/pickering/STScI/smite/pandeia_test/tests/engine/jwst/instrument/nircam/defaults/sw_imaging/int.npz"
effdat = np.load(file)

plt.figure()
plt.plot(effdat['arr_0'], effdat['arr_1'])
plt.plot(effdat['arr_0'], effdat['arr_2'])
#plt.plot(effdat['arr_0'], effdat['arr_3'])
#plt.plot(effdat['arr_0'], effdat['arr_4'])
plt.show()

fits.open("../../jwst/nircam/optical/jwst_nircam_lw_dbs.fits")[1].data

3 * 0.02186666666666667 * 256

int(np.round(16.7936 / 0.02186666666666667 / 3))

filename = "jwst_miri_configuration_20160914154718.json"

import re, glob

n = re.compile(r"_\d{14}")

timestamp = n.search(filename).group(0)

f = filename.replace(timestamp, '')

f

print(n.search(f))

for k, v in filter_config.items():
    print(k, v)

p, s = f.split('.')

p, s

fileglob = "/grp/jwst/wit/refdata/pandeia/jwst_etc_prerelease_05/nircam/filters/jwst*.fits"
sorted(glob.glob(fileglob))

get_ipython().magic('pinfo glob.glob')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

bcolors.HEADER

a

np.isnan([np.nan]).any()




from os import path
from collections import OrderedDict

# Third-party
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from astropy.constants import G, c
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
plt.style.use('apw-notebook')
get_ipython().magic('matplotlib inline')
from scipy.stats import scoreatpercentile
import sqlalchemy
from astropy.stats import median_absolute_deviation

import corner
import emcee
from scipy.integrate import quad
from scipy.misc import logsumexp
import schwimmbad

from gwb.data import TGASData
from gwb.fml import ln_H1_FML, ln_H2_FML

from comoving_rv.log import logger
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo, PriorRV,
                                  SpectralLineInfo, SpectralLineMeasurement, RVMeasurement,
                                  GroupToObservations)

# base_path = '/Volumes/ProjectData/gaia-comoving-followup/'
base_path = '../../data/'
db_path = path.join(base_path, 'db.sqlite')
engine = db_connect(db_path)
session = Session()

tbl = Table.read('group_prob_dv.ecsv', format='ascii.ecsv')

smoh_tbl = Table.read('../../../gaia-comoving-stars/paper/t1-1-star.txt', format='ascii.csv')

def obs_to_row(obs, group_id):
    row = OrderedDict()
    
    # smoh group id
    row['Oh17_group_id'] = group_id
        
    # TGAS source id
    row['tgas_source_id'] = obs.tgas_source.source_id
    
    # preferred name
    row['name'] = obs.simbad_info.preferred_name
    
    # TGAS info
    star = obs.tgas_star()
    row['ra'] = obs.tgas_source.ra.to(u.degree).value
    row['dec'] = obs.tgas_source.dec.to(u.degree).value
    row['parallax'] = obs.tgas_source.parallax
    row['distance'] = star.get_distance(True).to(u.pc).value
    row['G'] = obs.tgas_source.phot_g_mean_mag
    
    # 2MASS magnitude
    row['J'] = obs.photometry.j_m
    
    # RV
    row['rv'] = (obs.rv_measurement.rv + obs.v_bary).to(u.km/u.s).value
    row['rv_err'] = (obs.rv_measurement.err).to(u.km/u.s).value
    
    return row

base_q = session.query(Observation).join(RVMeasurement).filter(RVMeasurement.rv != None)

rows = []
for gid in tbl['group_id']:
    group = session.query(GroupToObservations).filter(GroupToObservations.group_id == gid).one()

    try:
        gto = session.query(GroupToObservations).filter(GroupToObservations.group_id == gid).one()        
    except:
        print("No obs for {0}".format(gid))
        
    obs1 = base_q.filter(Observation.id == gto.observation1_id).one()
    obs2 = base_q.filter(Observation.id == gto.observation2_id).one()
        
    row1 = obs_to_row(obs1, group.new_group_id) # fill with new group id
    row2 = obs_to_row(obs2, group.new_group_id)
    
    rows.append(row1)
    rows.append(row2)

data_tbl = Table(rows)

# reorder because passing in to Table doesnt preserve order
data_tbl = data_tbl[list(row1.keys())]

# sort on group id
data_tbl.sort('Oh17_group_id')

data_tbl

group_tbl = tbl['group_id', 'prob'].copy()

base_q = session.query(Observation).join(RVMeasurement).filter(RVMeasurement.rv != None)

n_samples = 16384

more_cols = OrderedDict()
more_cols['group_id'] = [] # need to update to new group ids from db table
more_cols['sep_tan'] = []
more_cols['sep_tan_err'] = []
more_cols['relative_rv'] = []
more_cols['relative_rv_err'] = []

for gid in tbl['group_id']:
    group = session.query(GroupToObservations).filter(GroupToObservations.group_id == gid).one()

    try:
        gto = session.query(GroupToObservations).filter(GroupToObservations.group_id == gid).one()        
    except:
        print("No obs for {0}".format(gid))
        continue
        
    more_cols['group_id'].append(group.new_group_id)
        
    obs1 = base_q.filter(Observation.id == gto.observation1_id).one()
    obs2 = base_q.filter(Observation.id == gto.observation2_id).one()
    
    icrs1 = obs1.icrs_samples(size=n_samples)
    icrs2 = obs2.icrs_samples(size=n_samples)
    
    R = np.min([icrs1.distance.value, icrs2.distance.value], axis=0) * u.pc
    sep_tan = 2*R*np.sin(icrs1.separation(icrs2)/2)
    
    more_cols['sep_tan'].append(np.median(sep_tan).to(u.pc).value)
    more_cols['sep_tan_err'].append(1.5 * median_absolute_deviation(sep_tan).to(u.pc).value)
    
    # relative RV
    raw_rv_diff = (obs1.measurements[0].x0 - obs2.measurements[0].x0) / 6563. * c.to(u.km/u.s)        
    raw_rv_err = np.sqrt(obs1.measurements[0].x0_error**2 + obs2.measurements[0].x0_error**2) / 6563. * c.to(u.km/u.s)
    more_cols['relative_rv'].append(raw_rv_diff.to(u.km/u.s).value)
    more_cols['relative_rv_err'].append(raw_rv_err.to(u.km/u.s).value)

for name in more_cols.keys():
    group_tbl[name] = more_cols[name]
    
# rename
group_tbl.rename_column('group_id', 'Oh17_group_id')

(np.abs(group_tbl['relative_rv']) < 2*group_tbl['relative_rv_err']).sum()

group_tbl.sort('Oh17_group_id')
group_tbl

data_tbl.write('../1-star.txt', format='ascii.csv', overwrite=True)
group_tbl.write('../2-group.txt', format='ascii.csv', overwrite=True)




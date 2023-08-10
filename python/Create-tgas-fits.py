from os import path

# Third-party
from astropy.io import ascii
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from astropy.constants import G, c
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
plt.style.use('apw-notebook')
get_ipython().magic('matplotlib inline')
import sqlalchemy

from gwb.data import TGASData

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

base_q = session.query(Observation).join(RVMeasurement).filter(RVMeasurement.rv != None)

group_ids = np.array([x[0] 
                      for x in session.query(Observation.group_id).distinct().all() 
                      if x[0] is not None and x[0] > 0 and x[0] != 10])
len(group_ids)

star1_dicts = []
star2_dicts = []
for gid in np.unique(group_ids):
    try:
        gto = session.query(GroupToObservations).filter(GroupToObservations.group_id == gid).one()
        obs1 = base_q.filter(Observation.id == gto.observation1_id).one()
        obs2 = base_q.filter(Observation.id == gto.observation2_id).one()
    except sqlalchemy.orm.exc.NoResultFound:
        print('Skipping group {0}'.format(gid))
        continue
    
    raw_rv_diff = (obs1.measurements[0].x0 - obs2.measurements[0].x0) / 6563. * c.to(u.km/u.s)    
    mean_rv = np.mean([obs1.rv_measurement.rv.value, 
                       obs2.rv_measurement.rv.value]) * obs2.rv_measurement.rv.unit
    
    rv1 = mean_rv + raw_rv_diff/2.
    rv_err1 = obs1.measurements[0].x0_error / 6563. * c.to(u.km/u.s)
    rv2 = mean_rv - raw_rv_diff/2.
    rv_err2 = obs2.measurements[0].x0_error / 6563. * c.to(u.km/u.s)
    
    # -------
    # Star 1:
    row_dict = dict()
    star1 = obs1.tgas_star()
    for k in star1._data.dtype.names:
        if k in ['J', 'J_err', 'H', 'H_err', 'Ks', 'Ks_err']: continue
        row_dict[k] = star1._data[k]

    row_dict['RV'] = rv1.to(u.km/u.s).value
    row_dict['RV_err'] = rv_err1.to(u.km/u.s).value
    row_dict['group_id'] = gid
    star1_dicts.append(row_dict)

    # -------
    # Star 2:
    
    row_dict = dict()
    star2 = obs2.tgas_star()
    for k in star2._data.dtype.names:
        if k in ['J', 'J_err', 'H', 'H_err', 'Ks', 'Ks_err']: continue
        row_dict[k] = star2._data[k]

    row_dict['RV'] = rv2.to(u.km/u.s).value
    row_dict['RV_err'] = rv_err2.to(u.km/u.s).value
    row_dict['group_id'] = gid
    star2_dicts.append(row_dict)

tbl1 = Table(star1_dicts)
tbl2 = Table(star2_dicts)

tbl1.write('../../data/tgas_apw1.fits', overwrite=True)
tbl2.write('../../data/tgas_apw2.fits', overwrite=True)

tgas = TGASData('../../../gaia-comoving-stars/data/stacked_tgas.fits')

star = ascii.read('../../../gaia-comoving-stars/paper/t1-1-star.txt')
rave_stars = star[(star['group_size'] == 2) & (~star['rv'].mask)]
rave_stars = rave_stars.group_by('group_id')

group_idx = np.array([i for i,g in enumerate(rave_stars.groups) if len(g) > 1])
rave_stars = rave_stars.groups[group_idx]

star1_dicts = []
star2_dicts = []
for gid in np.unique(rave_stars['group_id']):
    rows = rave_stars[rave_stars['group_id'] == gid]
    
    if len(rows) != 2:
        print("skipping group {0} ({1})".format(gid, len(rows)))
        continue
        
    i1 = np.where(tgas._data['source_id'] == rows[0]['tgas_source_id'])[0][0]
    i2 = np.where(tgas._data['source_id'] == rows[1]['tgas_source_id'])[0][0]
    
    star1 = tgas[i1]
    star2 = tgas[i2]
    
    # -------
    # Star 1:
    row_dict = dict()
    for k in star1._data.dtype.names:
        if k in ['J', 'J_err', 'H', 'H_err', 'Ks', 'Ks_err']: continue
        row_dict[k] = star1._data[k]

    row_dict['RV'] = rows[0]['rv']
    row_dict['RV_err'] = rows[0]['erv']
    row_dict['group_id'] = gid
    star1_dicts.append(row_dict)

    # -------
    # Star 2:
    
    row_dict = dict()
    for k in star2._data.dtype.names:
        if k in ['J', 'J_err', 'H', 'H_err', 'Ks', 'Ks_err']: continue
        row_dict[k] = star2._data[k]

    row_dict['RV'] = rows[1]['rv']
    row_dict['RV_err'] = rows[1]['erv']
    row_dict['group_id'] = gid
    star2_dicts.append(row_dict)

tbl1 = Table(star1_dicts)
tbl2 = Table(star2_dicts)
print(len(tbl1))

tbl1.write('../../data/tgas_rave1.fits', overwrite=True)
tbl2.write('../../data/tgas_rave2.fits', overwrite=True)






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
from scipy.stats import scoreatpercentile
import sqlalchemy

import corner
import emcee
from scipy.integrate import quad
from scipy.misc import logsumexp

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

# Produced by/for scripts/mixture_model.py - see Create-tgas-fits.ipynb
prob_gids = Table.read('../../data/tgas_apw1.fits')['group_id']
pair_probs = np.load('../../data/pair_probs_apw.npy')

group_ids = np.array([x[0] 
                      for x in session.query(Observation.group_id).distinct().all() 
                      if x[0] is not None and x[0] > 0 and x[0] != 10])
len(group_ids)

base_q = session.query(Observation).join(RVMeasurement).filter(RVMeasurement.rv != None)

names = ['group_id', 'dv', 'prob']
rows = []
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
    
    # Compute point-estimate difference in 3D velocity
    icrs1 = obs1.icrs(with_rv=rv1, lutz_kelker=False)
    icrs2 = obs2.icrs(with_rv=rv2, lutz_kelker=False)
    
    icrs1.set_representation_cls(coord.CartesianRepresentation, coord.CartesianDifferential)
    icrs2.set_representation_cls(coord.CartesianRepresentation, coord.CartesianDifferential)
    
    dv = np.sqrt((icrs1.v_x-icrs2.v_x)**2 + 
                 (icrs1.v_y-icrs2.v_y)**2 + 
                 (icrs1.v_z-icrs2.v_z)**2)
    
    prob = pair_probs[prob_gids == gid]
    assert len(prob) == 1
    
    rows.append((gid, dv.value, prob[0]))
    
dtype = dict(names=names, formats=['i4']+['f8']*(len(names)-1))
tbl = np.array(rows, dtype)
tbl = Table(tbl)
tbl['dv'].unit = u.km/u.s

base_q = session.query(Observation).join(RVMeasurement).filter(RVMeasurement.rv != None)

dv_15s = []
dv_meds = []
dv_85s = []

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
    
    # Compute point-estimate difference in 3D velocity
    icrs1 = obs1.icrs_samples(size=2**16, custom_rv=(rv1,rv_err1))
    icrs2 = obs2.icrs_samples(size=2**16, custom_rv=(rv2,rv_err2))
    
    icrs1.set_representation_cls(coord.CartesianRepresentation, coord.CartesianDifferential)
    icrs2.set_representation_cls(coord.CartesianRepresentation, coord.CartesianDifferential)
    
    dv = np.sqrt((icrs1.v_x-icrs2.v_x)**2 + 
                 (icrs1.v_y-icrs2.v_y)**2 + 
                 (icrs1.v_z-icrs2.v_z)**2)
    
    dv_15, dv_med, dv_85 = scoreatpercentile(dv.value, [15, 50, 85])
    dv_15s.append(dv_15)
    dv_meds.append(dv_med)
    dv_85s.append(dv_85)
    
tbl['dv_15'] = dv_15s*u.km/u.s
tbl['dv_50'] = dv_meds*u.km/u.s
tbl['dv_85'] = dv_85s*u.km/u.s

base_q = session.query(Observation).join(RVMeasurement).filter(RVMeasurement.rv != None)

sep_2d = []
sep_3d = []
sep_tan = []
d_mins = []
for gid in tbl['group_id']:
    try:
        gto = session.query(GroupToObservations).filter(GroupToObservations.group_id == gid).one()
        obs1 = base_q.filter(Observation.id == gto.observation1_id).one()
        obs2 = base_q.filter(Observation.id == gto.observation2_id).one()
    except sqlalchemy.orm.exc.NoResultFound:
        print('Skipping group {0}'.format(gid))
        continue
    
    icrs1 = obs1.icrs(lutz_kelker=False)
    icrs2 = obs2.icrs(lutz_kelker=False)
    
    sep_2d.append(icrs1.separation(icrs2))
    sep_3d.append(icrs1.separation_3d(icrs2))
    
    R = min(icrs1.distance.value, icrs2.distance.value) * u.pc
    sep_tan.append(2*R*np.sin(icrs1.separation(icrs2)/2))
    d_mins.append(min(icrs1.distance.value, icrs2.distance.value))
    
tbl['sep_2d'] = u.Quantity(sep_2d)
tbl['sep_3d'] = u.Quantity(sep_3d)
tbl['chord_length'] = u.Quantity(sep_tan)
tbl['d_min'] = d_mins * icrs1.distance.unit

tbl.write('group_prob_dv.ecsv', format='ascii.ecsv', overwrite=True)

# Produced by/for scripts/mixture_model.py - see Create-tgas-fits.ipynb
rave_prob_gids = Table.read('../../data/tgas_rave1.fits')['group_id']
rave_pair_probs = np.load('../../data/pair_probs_rave.npy')

tgas = TGASData('../../../gaia-comoving-stars/data/stacked_tgas.fits')

star = ascii.read('../../../gaia-comoving-stars/paper/t1-1-star.txt')
rave_stars = star[(star['group_size'] == 2) & (~star['rv'].mask)]
rave_stars = rave_stars.group_by('group_id')

group_idx = np.array([i for i,g in enumerate(rave_stars.groups) if len(g) > 1])
rave_stars = rave_stars.groups[group_idx]

names = ['group_id', 'prob']
all_rows = []
for gid in np.unique(rave_stars['group_id']):
    rows = rave_stars[rave_stars['group_id'] == gid]
    
    if len(rows) != 2:
        print("skipping group {0} ({1})".format(gid, len(rows)))
        continue
        
    i1 = np.where(tgas._data['source_id'] == rows[0]['tgas_source_id'])[0][0]
    i2 = np.where(tgas._data['source_id'] == rows[1]['tgas_source_id'])[0][0]
    
    star1 = tgas[i1]
    star2 = tgas[i2]

    star1._rv = rows[0]['rv']
    star2._rv = rows[1]['rv']
    star1._rv_err = rows[0]['erv']
    star2._rv_err = rows[1]['erv']
    
    prob = rave_pair_probs[rave_prob_gids == gid]
    assert len(prob) == 1
    
    all_rows.append((gid, prob))
    
dtype = dict(names=names, formats=['i4']+['f8']*(len(names)-1))
tbl = np.array(all_rows, dtype)
tbl = Table(tbl)

dv_pts = []
dv_15s = []
dv_meds = []
dv_85s = []
sep_2d = []
sep_3d = []
sep_tan = []
d_mins = []

for gid in np.unique(rave_stars['group_id']):
    rows = rave_stars[rave_stars['group_id'] == gid]
    
    if len(rows) != 2:
        print("skipping group {0} ({1})".format(gid, len(rows)))
        continue
        
    i1 = np.where(tgas._data['source_id'] == rows[0]['tgas_source_id'])[0][0]
    i2 = np.where(tgas._data['source_id'] == rows[1]['tgas_source_id'])[0][0]
    
    star1 = tgas[i1]
    star2 = tgas[i2]
    
    icrs1 = star1.get_icrs(rows[0]['rv']*u.km/u.s)
    icrs2 = star2.get_icrs(rows[1]['rv']*u.km/u.s)
    d_mins.append(min(icrs1.distance.value, icrs2.distance.value))
    
    R = min(icrs1.distance.value, icrs2.distance.value) * u.pc
    sep_tan.append(2*R*np.sin(icrs1.separation(icrs2)/2))
    
    # separations    
    sep_2d.append(icrs1.separation(icrs2))
    sep_3d.append(icrs1.separation_3d(icrs2))
    
    icrs1.set_representation_cls(coord.CartesianRepresentation, coord.CartesianDifferential)
    icrs2.set_representation_cls(coord.CartesianRepresentation, coord.CartesianDifferential)
    
    dv_pt = np.sqrt((icrs1.v_x-icrs2.v_x)**2 + 
                    (icrs1.v_y-icrs2.v_y)**2 + 
                    (icrs1.v_z-icrs2.v_z)**2)
    dv_pts.append(dv_pt)
    
    # Compute difference in 3D velocity for samples
    icrs1 = star1.get_icrs_samples(size=2**16, rv=rows[0]['rv']*u.km/u.s, 
                                   rv_err=rows[0]['erv']*u.km/u.s)
    icrs2 = star2.get_icrs_samples(size=2**16, rv=rows[1]['rv']*u.km/u.s, 
                                   rv_err=rows[1]['erv']*u.km/u.s)
    
    icrs1.set_representation_cls(coord.CartesianRepresentation, coord.CartesianDifferential)
    icrs2.set_representation_cls(coord.CartesianRepresentation, coord.CartesianDifferential)
    
    dv = np.sqrt((icrs1.v_x-icrs2.v_x)**2 + 
                 (icrs1.v_y-icrs2.v_y)**2 + 
                 (icrs1.v_z-icrs2.v_z)**2)
    
    dv_15, dv_med, dv_85 = scoreatpercentile(dv.value, [15, 50, 85])
    dv_15s.append(dv_15)
    dv_meds.append(dv_med)
    dv_85s.append(dv_85)

tbl['dv'] = u.Quantity(dv_pts)
tbl['dv_15'] = dv_15s*u.km/u.s
tbl['dv_50'] = dv_meds*u.km/u.s
tbl['dv_85'] = dv_85s*u.km/u.s
tbl['sep_2d'] = u.Quantity(sep_2d)
tbl['sep_3d'] = u.Quantity(sep_3d)
tbl['chord_length'] = u.Quantity(sep_tan)
tbl['d_min'] = d_mins * u.pc

tbl.write('group_prob_dv_rave.ecsv', format='ascii.ecsv', overwrite=True)




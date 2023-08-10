from os import path

from astropy.constants import c
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')
get_ipython().magic('matplotlib inline')
from scipy.optimize import minimize

from comoving_rv.log import logger
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo, PriorRV,
                                  SpectralLineInfo, SpectralLineMeasurement)

base_path = '/Volumes/ProjectData/gaia-comoving-followup/'
db_path = path.join(base_path, 'db.sqlite')
engine = db_connect(db_path)
session = Session()

Halpha, = session.query(SpectralLineInfo.wavelength).filter(SpectralLineInfo.name == 'Halpha').one()

q = session.query(Observation).join(Run, SpectralLineMeasurement, PriorRV)
q = q.filter(Run.name == 'mdm-spring-2017')
q = q.filter(SpectralLineMeasurement.x0 != None)
q = q.filter(PriorRV.rv != None)
q.distinct().count()

observations = q.all()

raw_offsets = np.zeros(len(observations)) * u.angstrom
all_sky_offsets = np.full((len(observations), 3), np.nan) * u.angstrom
true_rv = np.zeros(len(observations)) * u.km/u.s
meas_rv = np.zeros(len(observations)) * u.km/u.s
night_id = np.zeros(len(observations), dtype=int)
obs_time = np.zeros(len(observations))
airmass = np.zeros(len(observations))
hour_angle = np.zeros(len(observations))
color_by = np.zeros(len(observations))

for i,obs in enumerate(observations):
    print(obs.object, obs.filename_1d)
    
    # color_by[i] = obs.airmass
    # color_by[i] = obs.exptime
    
    obs_time[i] = np.sum(np.array(list(map(float, obs.time_obs.split(':')))) / np.array([1., 60., 3600.]))
    color_by[i] = obs_time[i]
    night_id[i] = obs.night
    airmass[i] = obs.airmass
    hour_angle[i] = obs.ha.degree

    x0 = obs.measurements[0].x0 * u.angstrom
    offset = (x0 - Halpha)
    raw_offsets[i] = offset
    
    sky_offsets = []
    for j,meas in enumerate(obs.measurements[1:]):
        sky_offset = meas.x0*u.angstrom - meas.info.wavelength
        if meas.amp > 16 and meas.std_G < 2 and meas.std_G > 0.3 and np.abs(sky_offset) < 4*u.angstrom:
            sky_offsets.append(sky_offset)
            all_sky_offsets[i,j] = sky_offset
            print(meas.info.name, meas.x0, meas.amp, sky_offset)
            
    sky_offsets = u.Quantity(sky_offsets)
    
    if len(sky_offsets) > 0:
        sky_offset = np.mean(sky_offsets)
    else:
        sky_offset = 0. * u.angstrom
        print("WARNING: not correcting with sky line")
    
    # sky_offset = -1.2*u.angstrom
    meas_rv[i] = (offset - sky_offset) / Halpha * c.to(u.km/u.s) + obs.v_bary
    true_rv[i] = obs.prior_rv.rv - obs.v_bary
    print(meas_rv[i] - true_rv[i])
    print()

raw_rv = raw_offsets / Halpha * c.to(u.km/u.s)

grid = np.linspace(0, 14, 256)

diff = all_sky_offsets[:,0] - ((raw_rv - true_rv)/c*5577*u.angstrom).decompose()
# diff = all_sky_offsets[:,0] - ((meas_rv - true_rv)/c*5577*u.angstrom).decompose()
diff[np.abs(diff) > 2*u.angstrom] = np.nan * u.angstrom

night_polys = dict()
for n in range(1,5+1):
    mask = (night_id == n) & np.isfinite(diff)
    coef = np.polyfit(obs_time[mask], diff[mask], deg=1, w=np.full(mask.sum(), 1/0.1))
    poly = np.poly1d(coef)
    night_polys[n] = poly
    
    plt.figure()
    sc = plt.scatter(obs_time[mask], diff[mask])
    plt.plot(grid, poly(grid), color=sc.get_facecolor()[0], marker='')
    plt.title('n{}'.format(n))
    plt.xlim(-0, 15)
    plt.ylim(-2, 2)

corrected_rv = np.zeros(len(observations)) * u.km/u.s
derps = dict()
for n in np.unique(night_id):
    mask = night_id == n
    
    sky_offset = np.nanmean(all_sky_offsets[mask,:2], axis=1)
    sky_offset[np.isnan(sky_offset)] = 0.*u.angstrom
    sky_offset -= night_polys[n](obs_time[mask]) * u.angstrom
    
    corrected_rv[mask] = (raw_offsets[mask] - sky_offset) / Halpha * c.to(u.km/u.s)
    
for n in np.unique(night_id):
    mask = night_id == n
    derps[n] = np.nanmedian(corrected_rv[mask] - true_rv[mask])
    corrected_rv[mask] = corrected_rv[mask] - derps[n]

fig,axes = plt.subplots(2, 5, figsize=(15,6.5), sharex='row', sharey='row')

_lim = (-200, 200)
_grid = np.linspace(_lim[0], _lim[1], 16)
_bins = np.linspace(-100, 100, 31)
for n in range(1,5+1):
    ax = axes[0, n-1]
    
    mask = night_id == n
    
    cb = ax.scatter(corrected_rv[mask], true_rv[mask], c=color_by[mask], 
                    marker='.', alpha=0.75, vmin=0, vmax=12, linewidth=1., 
                    edgecolor='#aaaaaa', s=50)
    ax.plot(_grid, _grid, marker='', zorder=-10, color='#888888')
    
    # histogram
    ax = axes[1, n-1]
    drv = corrected_rv[mask] - true_rv[mask]
    ax.hist(drv, bins=_bins)
    
    print(n, np.median(drv), 1.5 * np.median(np.abs(drv - np.median(drv))), np.mean(drv), np.std(drv))
    
axes[0,0].set_xlim(_lim)
axes[0,0].set_ylim(_lim)
axes[1,0].set_xlim(_bins.min(), _bins.max())

fig.tight_layout()

# all:
drv = corrected_rv - true_rv
print('total:', 1.5 * np.median(np.abs(drv - np.median(drv))), np.std(drv[np.abs(drv) < 50*u.km/u.s]))

plt.figure()
plt.hist(drv[np.abs(drv) < 100*u.km/u.s], bins='auto');

fig,axes = plt.subplots(1, 2, figsize=(6.5, 3.1))

_lim = (-200, 200)
_grid = np.linspace(_lim[0], _lim[1], 16)
_bins = np.linspace(-100, 100, 31)

cb = axes[0].scatter(corrected_rv, true_rv, c=color_by, 
                     marker='.', alpha=0.75, vmin=0, vmax=12, linewidth=1., 
                     edgecolor='#aaaaaa', s=50)
axes[0].plot(_grid, _grid, marker='', zorder=-10, color='#888888')

# histogram
drv = corrected_rv - true_rv
axes[1].hist(drv, bins=_bins)

print(n, np.median(drv), 1.5 * np.median(np.abs(drv - np.median(drv))), np.mean(drv), np.std(drv))
    
axes[0].set_xlim(_lim)
axes[0].set_ylim(_lim)
axes[1].set_xlim(_bins.min(), _bins.max())

fig.tight_layout()

class RVCorrector(object):

    def __init__(self, session, run_name):
        self.session = session
        self.run_name = str(run_name)

        # get wavelength for Halpha
        self.Halpha, = session.query(SpectralLineInfo.wavelength)                              .filter(SpectralLineInfo.name == 'Halpha').one()

        self._compute_offset_corrections()

    def _compute_offset_corrections(self):
        session = self.session
        run_name = self.run_name

        q = session.query(Observation).join(Run, SpectralLineMeasurement, PriorRV)
        q = q.filter(Run.name == run_name)
        q = q.filter(SpectralLineMeasurement.x0 != None)
        q = q.filter(PriorRV.rv != None)
        logger.debug('{0} observations with prior RV measurements'
                     .format(q.distinct().count()))

        # retrieve all observations with measured centroids and previous RV's
        observations = q.all()

        # What we do below is look at the residual offsets between applying a naïve
        # sky-line correction and the true RV (with the barycentric velocity
        # applied)

        raw_offsets = np.zeros(len(observations)) * u.angstrom
        all_sky_offsets = np.full((len(observations), 3), np.nan) * u.angstrom
        true_rv = np.zeros(len(observations)) * u.km/u.s
        obs_time = np.zeros(len(observations))
        night_id = np.zeros(len(observations), dtype=int)
        corrected_rv = np.zeros(len(observations)) * u.km/u.s

        for i,obs in enumerate(observations):
            # convert obstime into decimal hour
            obs_time[i] = np.sum(np.array(list(map(float, obs.time_obs.split(':')))) / np.array([1., 60., 3600.]))

            # Compute the raw offset: difference between Halpha centroid and true
            # wavelength value
            x0 = obs.measurements[0].x0 * u.angstrom
            offset = (x0 - self.Halpha)
            raw_offsets[i] = offset

            night_id[i] = obs.night

            # For each sky line (that passes certain quality checks), compute the
            # offset between the predicted wavelength and measured centroid
            # TODO: generalize these quality cuts - see also below in
            # get_corrected_rv
            sky_offsets = []
            for j,meas in enumerate(obs.measurements[1:]):
                sky_offset = meas.x0*u.angstrom - meas.info.wavelength
                if (meas.amp > 16 and meas.std_G < 2 and meas.std_G > 0.3 and
                        np.abs(sky_offset) < 4*u.angstrom): # MAGIC NUMBER: quality cuts
                    sky_offsets.append(sky_offset)
                    all_sky_offsets[i,j] = sky_offset

            sky_offsets = u.Quantity(sky_offsets)

            if len(sky_offsets) > 0:
                sky_offset = np.mean(sky_offsets)
            else:
                sky_offset = np.nan * u.angstrom
                logger.debug("not correcting with sky line for {0}".format(obs))

            true_rv[i] = obs.prior_rv.rv - obs.v_bary

        raw_rv = raw_offsets / self.Halpha * c.to(u.km/u.s)

        # unique night ID's
        unq_night_id = np.unique(night_id)
        unq_night_id.sort()

        # Now we do a totally insane thing. From visualizing the residual
        # differences, there seems to be a trend with the observation time. We
        # fit a line to these residuals and use this to further correct the
        # wavelength solutions using just the (strongest) [OI] 5577 Å line.
        diff = all_sky_offsets[:,0] - ((raw_rv - true_rv)/c*5577*u.angstrom).decompose()
        diff[np.abs(diff) > 2*u.angstrom] = np.nan * u.angstrom # reject BIG offsets

        self._night_polys = dict()
        self._night_final_offsets = dict()
        for n in unq_night_id:
            mask = (night_id == n) & np.isfinite(diff)
            coef = np.polyfit(obs_time[mask], diff[mask], deg=1, w=np.full(mask.sum(), 1/0.1))
            poly = np.poly1d(coef)
            self._night_polys[n] = poly

            sky_offset = np.nanmean(all_sky_offsets[mask,:2], axis=1)
            sky_offset[np.isnan(sky_offset)] = 0.*u.angstrom
            sky_offset -= self._night_polys[n](obs_time[mask]) * u.angstrom

            corrected_rv[mask] = (raw_offsets[mask] - sky_offset) / self.Halpha * c.to(u.km/u.s)

            # Finally, we align the median of each night's ∆RV distribution with 0
            drv = corrected_rv[mask] - true_rv[mask]
            self._night_final_offsets[n] = np.nanmedian(drv)

        # now estimate the std. dev. uncertainty using the MAD
        all_drv = corrected_rv - true_rv
        self._abs_err = 1.5 * np.nanmedian(np.abs(all_drv - np.nanmedian(all_drv)))

    def get_corrected_rv(self, obs):
        """Compute a corrected radial velocity for the given observation"""

        # Compute the raw offset: difference between Halpha centroid and true
        # wavelength value
        x0 = obs.measurements[0].x0 * u.angstrom
        raw_offset = (x0 - self.Halpha)

        # precision estimate from line centroid error
        precision = (obs.measurements[0].x0_error* u.angstrom) / self.Halpha * c.to(u.km/u.s)

        # For each sky line (that passes certain quality checks), compute the
        # offset between the predicted wavelength and measured centroid
        # TODO: generalize these quality cuts - see also above in
        # _compute_offset_corrections
        sky_offsets = np.full(3, np.nan) * u.angstrom
        for j,meas in enumerate(obs.measurements[1:]):
            sky_offset = meas.x0*u.angstrom - meas.info.wavelength
            if (meas.amp > 16 and meas.std_G < 2 and meas.std_G > 0.3 and
                    np.abs(sky_offset) < 3.3*u.angstrom): # MAGIC NUMBER: quality cuts
                sky_offsets[j] = sky_offset

        # final sky offset to apply
        flag = 0
        sky_offset = np.nanmean(sky_offsets)
        if np.isnan(sky_offset.value):
            logger.debug("not correcting with sky line for {0}".format(obs))
            sky_offset = 0*u.angstrom
            flag = 1

        # apply global sky offset correction - see _compute_offset_corrections()
        sky_offset -= self._night_polys[obs.night](obs.utc_hour) * u.angstrom

        # compute radial velocity and correct for sky line
        rv = (raw_offset - sky_offset) / self.Halpha * c.to(u.km/u.s)

        # correct for offset of median of ∆RV distribution from targets with
        # prior/known RV's
        rv -= self._night_final_offsets[obs.night]

        # rv error
        err = np.sqrt(self._abs_err**2 + precision**2)

        return rv, err, flag

rv_corr = RVCorrector(session, 'mdm-spring-2017')

corrected_rv2 = np.zeros_like(corrected_rv)
err = np.zeros_like(corrected_rv)
flags = np.zeros(len(corrected_rv2))
for i,obs in enumerate(observations):
    corrected_rv2[i], err[i], flags[i] = rv_corr.get_corrected_rv(obs)

fig,axes = plt.subplots(2, 5, figsize=(15,6.5), sharex='row', sharey='row')

_lim = (-200, 200)
_grid = np.linspace(_lim[0], _lim[1], 16)
_bins = np.linspace(-100, 100, 31)
for n in range(1,5+1):
    ax = axes[0, n-1]
    
    mask = (night_id == n) & (flags == 0)
    cb = ax.scatter(corrected_rv2[mask], true_rv[mask], c=color_by[mask], 
                   marker='.', alpha=0.75, vmin=0, vmax=12, linewidth=1., 
                    edgecolor='#aaaaaa', s=50)
    ax.plot(_grid, _grid, marker='', zorder=-10, color='#888888')
    
    # histogram
    ax = axes[1, n-1]
    drv = corrected_rv2[mask] - true_rv[mask]
    ax.hist(drv, bins=_bins)
    
    print(n, np.median(drv), 1.5 * np.median(np.abs(drv - np.median(drv))), np.mean(drv), np.std(drv))
    
axes[0,0].set_xlim(_lim)
axes[0,0].set_ylim(_lim)
axes[1,0].set_xlim(_bins.min(), _bins.max())

fig.tight_layout()

# all:
drv = corrected_rv2 - true_rv
print('total:', 1.5 * np.median(np.abs(drv - np.median(drv))), np.std(drv[np.abs(drv) < 50*u.km/u.s]))

plt.figure()
plt.hist(drv[np.abs(drv) < 100*u.km/u.s], bins='auto');

fig,axes = plt.subplots(1, 2, figsize=(6.5, 3.1))

_lim = (-200, 200)
_grid = np.linspace(_lim[0], _lim[1], 16)
_bins = np.linspace(-100, 100, 31)

# cb = axes[0].scatter(corrected_rv2, true_rv, c=color_by, 
#                      marker='.', alpha=0.75, vmin=0, vmax=12, linewidth=1., 
#                      edgecolor='#aaaaaa', s=50)
cb = axes[0].scatter(corrected_rv2, true_rv, 
                     marker='.', alpha=0.75, linewidth=1., 
                     edgecolor='#aaaaaa', s=50)
axes[0].plot(_grid, _grid, marker='', zorder=-10, color='#888888')

# histogram
drv = corrected_rv2 - true_rv
axes[1].hist(drv, bins=_bins)

print(n, np.median(drv), 1.5 * np.median(np.abs(drv - np.median(drv))), np.mean(drv), np.std(drv))
    
axes[0].set_xlim(_lim)
axes[0].set_ylim(_lim)
axes[1].set_xlim(_bins.min(), _bins.max())

fig.tight_layout()









fig,axes = plt.subplots(3, 5, figsize=(15,8), sharex=True, sharey=True)

for n in range(1,5+1):
    ax = axes[:,n-1]
    mask = night_id == n
    
    for j in range(3):
        ax[j].scatter(obs_time[mask], all_sky_offsets[mask,j], c=np.arange(mask.sum()), cmap='jet')
    
    ax[0].set_title('night {0}'.format(n))

axes[2,2].set_xlabel('time [hour]')
axes[0,0].set_ylabel('5577 offset')
axes[1,0].set_ylabel('6300 offset')
axes[2,0].set_ylabel('6364 offset')

fig.tight_layout()

# ---

fig,axes = plt.subplots(3, 5, figsize=(15,8), sharex=True, sharey=True)

for n in range(1,5+1):
    ax = axes[:,n-1]
    mask = night_id == n
    
    for j in range(3):
#         ax[j].scatter(hour_angle[mask], all_sky_offsets[mask,j], c=np.arange(mask.sum()), cmap='jet')
        ax[j].scatter(hour_angle[mask], all_sky_offsets[mask,j], c=airmass[mask], 
                      cmap='Spectral', edgecolor='#555555', linewidth=1)
    
    ax[0].set_title('night {0}'.format(n))

axes[2,2].set_xlabel('time [hour]')
axes[0,0].set_ylabel('5577 offset')
axes[1,0].set_ylabel('6300 offset')
axes[2,0].set_ylabel('6364 offset')

axes[0,0].set_xlim(-75, 75)

fig.tight_layout()

((raw_rv - true_rv)[mask]/c*5577).decompose()

plt.figure(figsize=(6,6))

for n in range(1,5+1):
    mask = night_id == n
    plt.scatter(all_sky_offsets[mask,0], 
                ((raw_rv - true_rv)[mask]/c*5577).decompose())
plt.xlim(-2.5,2.5)
# plt.ylim(-150,150)
plt.ylim(-2.5,2.5)



q = session.query(Observation.group_id).join(Run, SpectralLineMeasurement)
q = q.filter(Run.name == 'mdm-spring-2017')
print(q.distinct().count())
gids = np.array([gid for gid, in q.distinct().all()])

rv_diffs = []
all_x0s = []
for gid in gids:
    meas_q = session.query(SpectralLineMeasurement).join(Observation, SpectralLineInfo)
    meas_q = meas_q.filter(SpectralLineInfo.name == 'Halpha')
    meas_q = meas_q.filter(Observation.group_id == gid)
    x0s = [meas.x0 for meas in meas_q.all()]
    
    if len(x0s) == 2:
        rv_diff = (x0s[1] - x0s[0])*u.angstrom / Halpha * c.to(u.km/u.s)
        rv_diffs.append(rv_diff)
        
        all_x0s = all_x0s + x0s
    
    else:
        rv_diffs.append(np.nan*u.km/u.s)
        
rv_diffs = u.Quantity(rv_diffs)
all_x0s = u.Quantity(all_x0s)

# random shuffle
# TODO: redo this after correcting for barycenter, sky shifts, etc.
_derp = np.random.choice(len(all_x0s), size=len(all_x0s), replace=False)
random_rv_diff = (all_x0s - all_x0s[_derp])*u.angstrom / Halpha * c.to(u.km/u.s)

_,bins,_ = plt.hist(rv_diffs[np.isfinite(rv_diffs)], bins=np.linspace(-150, 150, 31), 
                    normed=True, alpha=0.5)

plt.hist(np.random.normal(0, np.sqrt(25**2 + 25**2 + 5**2 + 5**2), size=64000), 
         bins=bins, normed=True, alpha=0.5);

_,bins,_ = plt.hist(random_rv_diff[np.isfinite(random_rv_diff)], bins=bins, normed=True, alpha=0.4)

sky_diffs = []
for gid in gids:
    meas_q = session.query(SpectralLineMeasurement).join(Observation, SpectralLineInfo)
    meas_q = meas_q.filter(SpectralLineInfo.name == '[OI] 6300')
    meas_q = meas_q.filter(Observation.group_id == gid)
    x0s = [meas.x0 for meas in meas_q.all()]

    if len(x0s) == 2:
        sky_diff = (x0s[1] - x0s[0])*u.angstrom / (6300*u.angstrom) * c.to(u.km/u.s)
        sky_diffs.append(sky_diff)
    
    else:
        sky_diffs.append(np.nan*u.km/u.s)
    
sky_diffs = u.Quantity(sky_diffs)

plt.hist(sky_diffs[np.isfinite(sky_diffs)], bins=np.linspace(-50, 50, 30));





seps = []
for gid in gids[np.isfinite(rv_diffs) & (rv_diffs < 15*u.km/u.s)]:
    _q = session.query(TGASSource).join(Observation)
    _q = _q.filter(Observation.group_id == gid)
    tgas_data = _q.all()
    
    c1 = tgas_data[0].skycoord
    c2 = tgas_data[1].skycoord
    
    sep = c1.separation_3d(c2)
    seps.append(sep)

seps = u.Quantity(seps)

plt.hist(seps[seps > 0*u.pc], bins='auto')




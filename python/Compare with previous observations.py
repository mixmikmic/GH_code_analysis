from os import path

from astropy.constants import c
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')
get_ipython().magic('matplotlib inline')

from comoving_rv.log import logger
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo, PriorRV,
                                  SpectralLineInfo, SpectralLineMeasurement, RVMeasurement)

# base_path = '/Volumes/ProjectData/gaia-comoving-followup/'
base_path = '../../data/'
db_path = path.join(base_path, 'db.sqlite')
engine = db_connect(db_path)
session = Session()

q = session.query(Observation).join(Run, SpectralLineMeasurement, PriorRV, RVMeasurement)
q = q.filter(Run.name == 'mdm-spring-2017')
q = q.filter(SpectralLineMeasurement.x0 != None)
q = q.filter(PriorRV.rv != None)
q = q.filter(RVMeasurement.rv != None)
q.distinct().count()

observations = q.all()

apw_rv = u.Quantity([obs.rv_measurement.rv+obs.v_bary for obs in observations])
apw_rv_err = u.Quantity([obs.rv_measurement.err for obs in observations])

true_rv = u.Quantity([obs.prior_rv.rv for obs in observations])
true_rv_err = u.Quantity([obs.prior_rv.err for obs in observations])

fig,axes = plt.subplots(1, 2, figsize=(8,4))

_lim = (-275, 275)
_grid = np.linspace(_lim[0], _lim[1], 16) # for 1-to-1 line

axes[0].scatter(apw_rv, true_rv, marker='.', alpha=0.75, s=10)
axes[0].errorbar(apw_rv.value, true_rv.value, xerr=apw_rv_err.value, yerr=true_rv_err.value, 
                 marker='None', ecolor='#aaaaaa', elinewidth=1., zorder=-1, linestyle='none')
axes[0].plot(_grid, _grid, marker='', zorder=-10, color='#888888')
    
# histogram
drv = apw_rv - true_rv
axes[1].hist(drv[np.abs(drv)<100*u.km/u.s], bins='auto')

axes[0].xaxis.set_ticks(np.arange(-200, 200+1, 100))
axes[0].yaxis.set_ticks(np.arange(-200, 200+1, 100))
axes[1].xaxis.set_ticks(np.arange(-100, 100+1, 50))

axes[0].set_xlim(_lim)
axes[0].set_ylim(_lim)
axes[1].set_xlim(-110, 110)

axes[0].set_xlabel(r"${{\rm RV}}$ (this work) [{0}]"
                   .format((u.km/u.s).to_string('latex_inline')), fontsize=20)
axes[0].set_ylabel(r"${{\rm RV}}_{{\rm lit}}$ (previous) [{0}]"
                   .format((u.km/u.s).to_string('latex_inline')), fontsize=20)

axes[1].set_xlabel(r"$\Delta$RV [{0}]".format((u.km/u.s).to_string('latex_inline')))

fig.tight_layout()

# fig.savefig('rv-comparison.pdf')




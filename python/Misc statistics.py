from astropy.io import fits, ascii
from astropy.table import vstack
from astropy.time import Time
from astropy.constants import c
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')
get_ipython().magic('matplotlib inline')
from scipy.stats import scoreatpercentile
from sqlalchemy import func

from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo,
                                  SpectralLineInfo, SpectralLineMeasurement)

db_path = '/Volumes/ProjectData/gaia-comoving-followup/db.sqlite'
engine = db_connect(db_path)
session = Session()

session.query(Observation.object).distinct().count()

t = Time([x[0] for x in session.query(Observation.jd).all()])

t.datetime.min(), t.datetime.max()

derp = np.array([x.group_id for x in session.query(Observation).all()
                 if x.group_id is not None and x.group_id > 0 and x.group_id != 10])
np.unique(derp).size

exptime = np.array([x[0] for x in session.query(Observation.exptime).all()])

plt.hist(exptime);

scoreatpercentile(exptime, [15, 85])




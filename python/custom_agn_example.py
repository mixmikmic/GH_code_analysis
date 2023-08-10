import numpy as np
import os

opsimdb = os.path.join("/Users","danielsf","physics")
opsimdb = os.path.join(opsimdb, "lsst_150412", "Development", "garage")
opsimdb = os.path.join(opsimdb, "OpSimData", "kraken_1042_sqlite.db")

from lsst.sims.catUtils.utils import AgnLightCurveGenerator
from lsst.sims.catUtils.baseCatalogModels import GalaxyObj
import time

agn_db = GalaxyObj()

# we must tell the light curve generator about both the database of sources (fatboy)
# and the database of opsim pointings (opsimdb)
lc_gen = AgnLightCurveGenerator(agn_db, opsimdb)

ptngs = lc_gen.get_pointings((-2.5, 2.5), (-2.25, 2.25))

lc_dict, truth_dict = lc_gen.light_curves_from_pointings(ptngs, lc_per_field=10)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

obj_name = lc_dict.keys()[5]
lc = lc_dict[obj_name]

fig, ax = plt.subplots()

ax.errorbar(lc['r']['mjd'], lc['r']['mag'], lc['r']['error'],
            fmt='', linestyle='None')
ax.scatter(lc['r']['mjd'], lc['r']['mag'], s=5, color='r')
ax.set(xlabel='MJD', ylabel='r-band magnitude')

from lsst.sims.catalogs.decorators import compound
from lsst.sims.catUtils.mixins import PhotometryGalaxies
from lsst.sims.catUtils.utils.LightCurveGenerator import _baseLightCurveCatalog

# _lightCurveCatalogClasses must inherit from _baseLightCurveCatalog.
# _baseLightCurveCatalog defines some functionality that the LightCurveGenerator expects
class _agnMeanMagCatalog(_baseLightCurveCatalog, PhotometryGalaxies):
    
    @compound("lightCurveMag", "sigma_lightCurveMag")
    def get_lightCurvePhotometry(self):
        """
        This method calculates the lightCurveMag and sigma_lightCurveMag values expected
        by the LightCurveGenerator.  [u,g,r,i,z,y]Agn and sigma_[u,g,r,i,z,y]Agn are
        calculated by methods defined in the PhotometryGalaxies mixin imported above.
        """
        return np.array([self.column_by_name("%sAgn" % self.obs_metadata.bandpass),
                         self.column_by_name("sigma_%sAgn" % self.obs_metadata.bandpass)])
        

lc_gen._lightCurveCatalogClass= _agnMeanMagCatalog

# this is just a constraint on our SQL query to make sure we do not get
# any objects that lack an AGN component
lc_gen._constraint = 'sedname_agn IS NOT NULL'

ra_bound = (-2.5, 2.5)
dec_bound = (-2.25, 2.25)
pointings = lc_gen.get_pointings(ra_bound, dec_bound, bandpass='r')

lc_dict, truth_dict = lc_gen.light_curves_from_pointings(pointings,
                                                         lc_per_field=10)

obj_name = lc_dict.keys()[5]
lc = lc_dict[obj_name]

fig, ax = plt.subplots()

ax.errorbar(lc['r']['mjd'], lc['r']['mag'], lc['r']['error'],
            fmt='', linestyle='None')
ax.scatter(lc['r']['mjd'], lc['r']['mag'], s=5, color='r')
ax.set(xlabel='MJD', ylabel='r-band magnitude')

class _agnRandomMagCatalog(_baseLightCurveCatalog, PhotometryGalaxies):
    
    rng = np.random.RandomState(119)
    
    @compound('uAgn_rando', 'gAgn_rando', 'rAgn_rando',
              'iAgn_rando', 'zAgn_rando', 'yAgn_rando')
    def get_randomMagnitudes(self):
        """
        Calculate a varying magnitude that is the mean magnitude plus random noise.
        """
        n_mags = len(self.column_by_name('uAgn'))
        return np.array([self.column_by_name('uAgn') + self.rng.random_sample(n_mags)*10.0,
                         self.column_by_name('gAgn') + self.rng.random_sample(n_mags)*10.0,
                         self.column_by_name('rAgn') + self.rng.random_sample(n_mags)*10.0,
                         self.column_by_name('iAgn') + self.rng.random_sample(n_mags)*10.0,
                         self.column_by_name('zAgn') + self.rng.random_sample(n_mags)*10.0,
                         self.column_by_name('yAgn') + self.rng.random_sample(n_mags)*10.0])

    @compound('sigma_uAgn_rando', 'sigma_gAgn_rando', 'sigma_rAgn_rando',
              'sigma_iAgn_rando', 'sigma_zAgn_ranod', 'sigma_yAgn_rando')
    def get_rando_uncertainties(self):
        """
        Calculate the uncertainty in the random magnitudes.
        
        The method _magnitudeUncertaintyGetter is defined in the PhotometryGalaxies mixin.
        The arguments for that method are:
        
        list of the magnitudes for which uncertainties are to be calculated
        list of the bandpass names associated with these magnitudes
            (so that m5 in that bandpass can be looked up from self.obs_metadata)
        name of the attribute containing the bandpasses
            (self.lsstBandpassDict is set by the method that calculates [u,g,r,i,z,y]Agn)
        """
        return self._magnitudeUncertaintyGetter(['uAgn_rando', 'gAgn_rando', 'rAgn_rando',
                                                 'iAgn_rando', 'zAgn_rando', 'yAgn_rando'],
                                                ['u', 'g', 'r', 'i', 'z', 'y'],
                                                'lsstBandpassDict')
    
    @compound("lightCurveMag", "sigma_lightCurveMag")
    def get_lightCurvePhotometry(self):
        return np.array([self.column_by_name("%sAgn_rando" % self.obs_metadata.bandpass),
                         self.column_by_name("sigma_%sAgn_rando" % self.obs_metadata.bandpass)])
        
    

from lsst.sims.catUtils.baseCatalogModels import GalaxyObj

lc_gen._lightCurveCatalogClass = _agnRandomMagCatalog

ra_bound = (-2.5, 2.5)
dec_bound = (-2.25, 2.25)
pointings = lc_gen.get_pointings(ra_bound, dec_bound, bandpass='r')

lc_dict_rando, truth_dict = lc_gen.light_curves_from_pointings(pointings,
                                                               lc_per_field=10)

obj_name = lc_dict_rando.keys()[5]
lc = lc_dict_rando[obj_name]

fig, ax = plt.subplots()

ax.errorbar(lc['r']['mjd'], lc['r']['mag'], lc['r']['error'],
            fmt='', linestyle='None')
ax.scatter(lc['r']['mjd'], lc['r']['mag'], s=5, color='r')
ax.set(xlabel='MJD', ylabel='r-band magnitude')

from lsst.sims.photUtils import BandpassDict

class _alternateAgnCatalog(_baseLightCurveCatalog, PhotometryGalaxies):

    @compound('uAgn_x', 'gAgn_x', 'rAgn_x', 'iAgn_x', 'zAgn_x', 'yAgn_x')
    def get_magnitudes(self):
        """
        Add a component to the mean magnitude that grows linearly with time
        (which probably means it is not a mean magnitude any more...)
        """
        
        if not hasattr(self, 'lsstBandpassDict'):
            self.lsstBandpassDict = BandpassDict.loadTotalBandpassesFromFiles()
        
        delta = 5.0 * (self.obs_metadata.mjd.TAI-59580.0)/3650.0

        # self._magnitudeGetter is defined in the PhotometryGalaxies mixin.
        # It's arguments are: a str indicating which galaxy component is being
        # simulated ('agn', 'disk', or 'bulge'), the BandpassDict containing
        # the bandpasses of the survey, a list of the columns defined by this
        # current getter method
        return self._magnitudeGetter('agn', self.lsstBandpassDict,
                                     self.get_magnitudes._colnames) + delta
    
    @compound('sigma_uAgn_x', 'sigma_gAgn_x', 'sigma_rAgn_x',
              'sigma_iAgn_x', 'sigma_zAgn_x', 'sigma_yAgn_x')
    def get_uncertainties(self):
        return self._magnitudeUncertaintyGetter(['uAgn_x', 'gAgn_x', 'rAgn_x',
                                                 'iAgn_x', 'zAgn_x', 'yAgn_x'],
                                                ['u', 'g', 'r', 'i', 'z', 'y'],
                                                'lsstBandpassDict')
    
    @compound("lightCurveMag", "sigma_lightCurveMag")
    def get_lightCurvePhotometry(self):
        return np.array([self.column_by_name("%sAgn_x" % self.obs_metadata.bandpass),
                         self.column_by_name("sigma_%sAgn_x" % self.obs_metadata.bandpass)])

lc_gen._lightCurveCatalogClass = _alternateAgnCatalog

ra_bound = (-2.5, 2.5)
dec_bound = (-2.25, 2.25)
pointings = lc_gen.get_pointings(ra_bound, dec_bound, bandpass='r')

lc_dict_alt, truth_dict = lc_gen.light_curves_from_pointings(pointings,
                                                               lc_per_field=10)

obj_name = lc_dict_alt.keys()[5]
lc = lc_dict_alt[obj_name]

fig, ax = plt.subplots()

ax.errorbar(lc['r']['mjd'], lc['r']['mag'], lc['r']['error'],
            fmt='', linestyle='None')
ax.scatter(lc['r']['mjd'], lc['r']['mag'], s=5, color='r')
ax.set(xlabel='MJD', ylabel='r-band magnitude')

class _variableAgnCatalog(_baseLightCurveCatalog, PhotometryGalaxies):

    rng = np.random.RandomState(88)
    
    def get_truthInfo(self):
        if not hasattr(self, 'truth_cache'):
            self.truth_cache = {}
            
        # get the uniqueIds of all of the AGn
        # (mostly so you know how many of them there are)
        id_val = self.column_by_name('uniqueId')
        
        output = []
        for ii in id_val:
            if ii in self.truth_cache:
                output.append(self.truth_cache[ii])
            else:
                period = self.rng.random_sample()*365.25
                phase = self.rng.random_sample()*2.0*np.pi
                amplitude = self.rng.random_sample()*10.0
                output.append((period, phase, amplitude))

        return np.array(output)
    
    @compound('uAgn_x', 'gAgn_x', 'rAgn_x', 'iAgn_x', 'zAgn_x', 'yAgn_x')
    def get_magnitudes(self):
        
        if not hasattr(self, 'lsstBandpassDict'):
            self.lsstBandpassDict = BandpassDict.loadTotalBandpassesFromFiles()
        
        delta = 5.0 * (self.obs_metadata.mjd.TAI-59580.0)/3650.0

        var_params = self.column_by_name('truthInfo')
        wave = [vv[2]*np.sin(2.0*np.pi*self.obs_metadata.mjd.TAI/vv[0] + vv[1]) for vv in var_params]
        
        return self._magnitudeGetter('agn', self.lsstBandpassDict,
                                     self.get_magnitudes._colnames) + delta + wave
    
    @compound('sigma_uAgn_x', 'sigma_gAgn_x', 'sigma_rAgn_x',
              'sigma_iAgn_x', 'sigma_zAgn_x', 'sigma_yAgn_x')
    def get_uncertainties(self):
        return self._magnitudeUncertaintyGetter(['uAgn_x', 'gAgn_x', 'rAgn_x',
                                                 'iAgn_x', 'zAgn_x', 'yAgn_x'],
                                                ['u', 'g', 'r', 'i', 'z', 'y'],
                                                'lsstBandpassDict')
    
    @compound("lightCurveMag", "sigma_lightCurveMag")
    def get_lightCurvePhotometry(self):
        return np.array([self.column_by_name("%sAgn_x" % self.obs_metadata.bandpass),
                         self.column_by_name("sigma_%sAgn_x" % self.obs_metadata.bandpass)])

lc_gen._lightCurveCatalogClass = _variableAgnCatalog

ra_bound = (-2.5, 2.5)
dec_bound = (-2.25, 2.25)
pointings = lc_gen.get_pointings(ra_bound, dec_bound, bandpass='r')

lc_dict_var, truth_dict = lc_gen.light_curves_from_pointings(pointings,
                                                               lc_per_field=10)

obj_name = lc_dict_var.keys()[5]
lc = lc_dict_var[obj_name]

fig, ax = plt.subplots()

ax.errorbar(lc['r']['mjd'], lc['r']['mag'], lc['r']['error'],
            fmt='', linestyle='None')
ax.scatter(lc['r']['mjd'], lc['r']['mag'], s=5, color='r')
ax.set(xlabel='MJD', ylabel='r-band magnitude')

truth_dict[obj_name]




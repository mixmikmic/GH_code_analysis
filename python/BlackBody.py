import os
home_dir = os.environ.get('HOME')

# Please enter the filename of the ztf_sim output file you would like to use. The example first determines
# your home directory and then uses a relative path (useful if working on several machines with different usernames)
survey_file = os.path.join(home_dir, 'data/ZTF/one_year_sim_incomplete.db')

# Please enter the path to where you have placed the Schlegel, Finkbeiner & Davis (1998) dust map files
# You can also set the environment variable SFD_DIR to this path (in that case the variable below should be None)
sfd98_dir = os.path.join(home_dir, 'data/sfd98')

import warnings
## No annoying warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from astropy.cosmology import Planck15
import simsurvey
import sncosmo

# Load the CCD corners from file
ccd_corners = np.genfromtxt('data/ZTF_corners.txt')
ccds = [ccd_corners[4*k+16:4*k+20] for k in range(16)]

bands = { 
  'ztfr' : 'data/ztfr_eff.txt',
  'ztfg' : 'data/ztfg_eff.txt',
  }

for bandname in bands.keys() :
    fname = bands[bandname]
    b = np.loadtxt(fname)
    band = sncosmo.Bandpass(b[:,0], b[:,1], name=bandname)
    sncosmo.registry.register(band)

# Load simulated survey from file (download from ftp://ftp.astro.caltech.edu/users/ebellm/one_year_sim_incomplete.db)
plan = simsurvey.SurveyPlan(load_opsim=survey_file, band_dict={'g': 'ztfg', 'r': 'ztfr'}, ccds=ccds)

mjd_range = (plan.cadence['time'].min(), plan.cadence['time'].max())

# Need to define a basic lightcurve shape in arbitrary flux units
# Using eq. (1) of Bazin et al. (2011) for a basic shape
t_rise = 30.
t_fall = 60.
phase = np.linspace(-6*t_rise, 4*t_fall, 100)
flux = np.exp(-phase/t_fall)/(1 + np.exp(-phase/t_rise))

source = simsurvey.BlackBodySource(phase, flux)

dust = sncosmo.CCM89Dust()
model = sncosmo.Model(source=source,
                      effects=[dust],
                      effect_names=['host'],
                      effect_frames=['rest'])

def random_parameters(redshifts, model,
                      mag=(-17.5, 0.5),
                      T=(2e4, 5e3),
                      r_v=2., ebv_rate=0.11,
                      **kwargs):
    temperatures = np.random.uniform(T[0]-T[1], T[0]+T[1], len(redshifts))
    
    # Amplitude
    amp = []
    for z, temp in zip(redshifts, temperatures):
        model.set(z=z, T=temp)
        mabs = np.random.normal(mag[0], mag[1])
        model.set_source_peakabsmag(mabs, 'bessellb', 'vega', cosmo=Planck15)
        amp.append(model.get('amplitude'))

    return {
        'amplitude': np.array(amp),
        'T': temperatures,
        'hostr_v': r_v * np.ones(len(redshifts)),
        'hostebv': np.random.exponential(ebv_rate, len(redshifts))
    }

transientprop = dict(lcmodel=model,
                    lcsimul_func=random_parameters,
                    lcsimul_prop=dict(mag=(-17.5, 0.5)))

z_max = 0.1
tr = simsurvey.get_transient_generator([0.0, z_max], ratekind='custom',
                                       ratefunc=lambda z: 1e-6,
                                       dec_range=[-30,90],
                                       mjd_range=[mjd_range[0] - model.maxtime()*(1.+z_max),
                                                  mjd_range[1] + model.mintime()*(1.+z_max)],
                                       transientprop=transientprop,
                                       sfd98_dir=sfd98_dir)

instprop = {"ztfg":{"gain":1.,"zp":30,"zpsys":'ab'},
            "ztfr":{"gain":1.,"zp":30,"zpsys":'ab'}}

survey = simsurvey.SimulSurvey(generator=tr, plan=plan, instprop=instprop)
    
lcs = survey.get_lightcurves(
    progress_bar=True, notebook=True # If you get an error because of the progress_bar, delete this line.
)

lcs[0]




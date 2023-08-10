import os
home_dir = os.environ.get('HOME')

# Please enter the filename of the ztf_sim output file you would like to use. The example first determines
# your home directory and then uses a relative path (useful if working on several machines with different usernames)
survey_file = os.path.join(home_dir, 'data/ZTF/one_year_sim_incomplete.db')

# Please enter the path to where you have placed the Schlegel, Finkbeiner & Davis (1998) dust map files
# You can also set the environment variable SFD_DIR to this path (in that case the variable below should be None)
sfd98_dir = os.path.join(home_dir, 'data/sfd98')

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import simsurvey
import simsurvey_tools as sst
import sncosmo
from astropy.cosmology import Planck15

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

fields = sst.load_ztf_fields()

# Load simulated survey from file
plan_in = np.loadtxt('data/surveyplan_combo.dat')
t_ = plan_in[:,0]
f_ = np.array(plan_in[:,1], dtype=int)
bands_ = {0: 'ztfg', 1: 'ztfr', 2: 'desi'}
b_ = [bands_[k] for k in np.array(plan_in[:,2], dtype=int)]
s_ = plan_in[:, 3]

plan = simsurvey.SurveyPlan(time=t_, band=b_, obs_field=f_, skynoise=s_,
                            fields={k: v for k, v in fields.items()
                                    if k in ['ra', 'dec', 'field_id',
                                             'width', 'height']})

mjd_range = (plan.cadence['time'].min(), plan.cadence['time'].max())

# To review the pointing schedule, you can use this table
plan.cadence

# Load phase, wavelengths and flux from file
phase, wave, flux = sncosmo.read_griddata_ascii('data/macronova_sed_wind20.dat')

# Create a time series source
source = sncosmo.TimeSeriesSource(phase, wave, flux)

# Create the model that combines SED and propagation effects
dust = sncosmo.CCM89Dust()
model = sncosmo.Model(source=source,
                      effects=[dust],
                      effect_names=['host'],
                      effect_frames=['rest'])

def random_parameters(redshifts, model,
                      r_v=2., ebv_rate=0.11,
                      **kwargs):
    cosmo = Planck15
    
    # Amplitude
    amp = []
    for z in redshifts:
        d_l = cosmo.luminosity_distance(z).value * 1e5
        amp.append(d_l**-2)

    return {
        'amplitude': np.array(amp),
        'hostr_v': r_v * np.ones(len(redshifts)),
        'hostebv': np.random.exponential(ebv_rate, len(redshifts))
    }

transientprop = dict(lcmodel=model,
                    lcsimul_func=random_parameters)

mjd_range = (plan.cadence['time'].min(), plan.cadence['time'].max())

tr = simsurvey.get_transient_generator([0.0, 0.05], ratekind='custom',
                                       ratefunc=lambda z: 3e-5,
                                       dec_range=[-30,90],
                                       mjd_range=[mjd_range[0] - 20,
                                                  mjd_range[1]],
                                       transientprop=transientprop,
                                       sfd98_dir=sfd98_dir)

instprop = {"ztfg":{"gain":1.,"zp":30,"zpsys":'ab'},
            "ztfr":{"gain":1.,"zp":30,"zpsys":'ab'},
            "desi":{"gain":1.,"zp":30,"zpsys":'ab'}}

survey = simsurvey.SimulSurvey(generator=tr, plan=plan, instprop=instprop)
    
lcs = survey.get_lightcurves(
    progress_bar=True, notebook=True # If you get an error because of the progress_bar, delete this line.
)

lcs[0]

lc = lcs[0]

lc.as_array()

idx = lcs.meta['idx_orig']
n_obs = np.zeros(survey.generator.ntransient)
n_obs[idx] = np.array([len(a) for a in lcs])

survey.generator.show_skycoverage(cscale=n_obs, cblabel="Number of pointings", mask=idx)
print 'MNe pointed to: %i out of %i'%(np.sum(n_obs > 0), survey.generator.ntransient)

lcs_det = [lc[lc['flux']/lc['fluxerr'] > 5] for lc in lcs]

idx_det = lcs.meta['idx_orig'][np.array([len(lc) > 0 for lc in lcs_det])]
n_det = np.zeros(survey.generator.ntransient)
lcs_det = [lc for lc in lcs_det if len(lc) > 0]
n_det[idx_det] = np.array([len(lc) for lc in lcs_det])

survey.generator.show_skycoverage(cscale=n_det, cblabel="Number of detections", mask=idx_det)
print 'MNe detected: %i out of %i'%(np.sum(n_det > 0), survey.generator.ntransient)

k = np.where(lcs.meta['idx_orig'] == np.where(n_det == n_det.max())[0][0])[0][0]
lcs[k]

_ = sncosmo.plot_lc(lcs[k])

z = np.array([lc.meta['z'] for lc in lcs_det])
p_i = np.array([lc['time'].min() - lc.meta['t0'] for lc in lcs_det])
p_e = np.array([lc['time'].max() - lc.meta['t0'] for lc in lcs_det])

plt.hist(p_i, lw=2, histtype='step', range=(0,10), bins=20)
print 'MNe found within a day:', np.sum(p_i < 1)
plt.xlabel('Detection phase (observer-frame)', fontsize='x-large')
plt.ylabel(r'$N_{SNe}$', fontsize='x-large')

n = np.array([len(lc) for lc in lcs_det])
snr_max = np.array([max(lc['flux']/lc['fluxerr']) for lc in lcs_det])
plt.hist(z, lw=2, histtype='step', range=(0,0.05),bins=20)
plt.xlabel('Redshift', fontsize='x-large')
plt.ylabel(r'$N_{SNe}$', fontsize='x-large')
_ = plt.xlim((0, 0.05))




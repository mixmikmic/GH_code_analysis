import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

class sizeme():
    """ Class to change html fontsize of object's representation"""
    def __init__(self,ob, size=50, height=120):
        self.ob = ob
        self.size = size
        self.height = height
    def _repr_html_(self):
        repl_tuple = (self.size, self.height, self.ob._repr_html_())
        return u'<span style="font-size:{0}%; line-height:{1}%">{2}</span>'.format(*repl_tuple)

pd.options.display.max_columns = 999

template_catalog = {197790: [197802, 198372, 198376, 198380, 198384],
                    197662: [198668, 199009, 199021, 199033],
                    197408: [197400, 197404, 197412],
                    197384: [197388, 197392],
                    197371: [197367, 197375, 197379]}

import lsst.daf.persistence as dp
#butler = dp.Butler('decam_lzogy_forcephot')
butler = dp.Butler('decam_rescaled_diffims')

df1 = None
#for template, sciImgs in template_catalog.items():
template = 197371
sciImgs = template_catalog[template]
for science in sciImgs:
    for ccdnum in range(1, 60):
        try:
            #sources = butler.get('forced_src', visit=197367, ccdnum=ccdnum)
            sources = butler.get('forced_src', visit=science, ccdnum=ccdnum)
            tmp = pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})
            tmp['visit'] = science
            tmp['ccdnum'] = ccdnum
            #print template, science, ccdnum, tmp.shape
            if df1 is None:
                df1 = tmp
            else:
                df1 = df1.append(tmp)
        except:
            continue

print df1.shape
print df1.base_PsfFlux_flag.values.sum()
print df1.base_PsfFlux_flag_edge.values.sum()
print df1.base_PsfFlux_flag_noGoodPixels.values.sum()
print df1.base_TransformedCentroid_flag.values.sum()
#print df1.ip_diffim_DipoleFit_flag_classification.values.sum()
sizeme(df1.head())

df1[['base_PsfFlux_flux', 'template_base_PsfFlux_flux']].head()
df1['s_to_n'] = df1.base_PsfFlux_flux / df1.base_PsfFlux_fluxSigma
df1['template_s_to_n'] = df1.template_base_PsfFlux_flux / df1.template_base_PsfFlux_fluxSigma
df1['diff_s_to_n'] = (df1.base_PsfFlux_flux - df1.template_base_PsfFlux_flux) /     np.sqrt(df1.base_PsfFlux_fluxSigma**2. + df1.template_base_PsfFlux_fluxSigma**2.)
#df1[['s_to_n', 'template_s_to_n']].head()

plt.rcParams['figure.figsize'] = (10.0, 6.5)
#colors = np.where(df1.ip_diffim_DipoleFit_flag_classification == 1, 'r', 'k')
colors = np.where(np.abs(df1.diff_s_to_n) > 5., 'b', 'r')
df1.plot.scatter('s_to_n', 'template_s_to_n', c=colors)
plt.xlim(-10, 25)
plt.ylim(-10, 25)
x = np.linspace(-25,25)
plt.plot(x, x-5*np.sqrt(2), 'k-')
plt.plot(x, x+5*np.sqrt(2), 'k-')

import lsst.daf.persistence as dp
butler = dp.Butler('decam_lzogy_forcephot')
#butler = dp.Butler('decam_rescaled_diffims')

df2 = None
#for template, sciImgs in template_catalog.items():
template = 197371
sciImgs = template_catalog[template]
for science in sciImgs:
    for ccdnum in range(1, 60):
        try:
            #sources = butler.get('forced_src', visit=197367, ccdnum=ccdnum)
            sources = butler.get('forced_src', visit=science, ccdnum=ccdnum)
            tmp = pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})
            tmp['visit'] = science
            tmp['ccdnum'] = ccdnum
            #print template, science, ccdnum, tmp.shape
            if df2 is None:
                df2 = tmp
            else:
                df2 = df2.append(tmp)
        except:
            continue

print df2.shape
print df2.base_PsfFlux_flag.values.sum()
print df2.base_PsfFlux_flag_edge.values.sum()
print df2.base_PsfFlux_flag_noGoodPixels.values.sum()
print df2.base_TransformedCentroid_flag.values.sum()
print df2.ip_diffim_DipoleFit_flag_classification.values.sum()
sizeme(df2.head())

df2[['base_PsfFlux_flux', 'template_base_PsfFlux_flux']].head()
df2['s_to_n'] = df2.base_PsfFlux_flux / df2.base_PsfFlux_fluxSigma
df2['template_s_to_n'] = df2.template_base_PsfFlux_flux / df2.template_base_PsfFlux_fluxSigma
df2['diff_s_to_n'] = (df2.base_PsfFlux_flux - df2.template_base_PsfFlux_flux) /     np.sqrt(df2.base_PsfFlux_fluxSigma**2. + df2.template_base_PsfFlux_fluxSigma**2.)
#df[['s_to_n', 'template_s_to_n']].head()

plt.rcParams['figure.figsize'] = (10.0, 6.5)
#colors = np.where(df2.ip_diffim_DipoleFit_flag_classification == 1, 'r', 'k')
colors = np.where(np.abs(df2.diff_s_to_n) > 5., 'b', 'r')
df2.plot.scatter('s_to_n', 'template_s_to_n', c=colors)
plt.xlim(-10, 25)
plt.ylim(-10, 25)
x = np.linspace(-25,25)
plt.plot(x, x-5*np.sqrt(2), 'k-')
plt.plot(x, x+5*np.sqrt(2), 'k-')

import cPickle
import gzip
if False:
    cPickle.dump((df1, df2), gzip.GzipFile('20. compare photometry-corrected-many-DECam-images.p.gz', 'wb'))
else:
    df1, df2 = cPickle.load(gzip.GzipFile('20. compare photometry-corrected-many-DECam-images.p.gz', 'rb'))

import diffimTests as dit
reload(dit);
plt.rcParams['figure.figsize'] = (15.0, 6.5)
dit.mosaicDIASources('decam_lzogy_forcephot', visitid=197367, ccdnum=1, xnear=1720, ynear=82)

tmp_df = df2[(df2.s_to_n.abs() <= 0.1) & (df2.template_s_to_n.abs() <= 0.1)]
print tmp_df.shape
dit.mosaicDIASources('decam_lzogy_forcephot', visitid=197367, ccdnum=1,                      xnear=tmp_df.base_TransformedCentroid_x, ynear=tmp_df.base_TransformedCentroid_y)

sizeme(tmp_df.head())

reload(dit)
tmp_df = df2[((df2.s_to_n - -2.5).abs() <= 0.25) & ((df2.template_s_to_n - 2.5).abs() <= 0.25)]
print tmp_df.shape
dit.mosaicDIASources('decam_lzogy_forcephot', visitid=197367, ccdnum=31,                      xnear=tmp_df.base_TransformedCentroid_x, ynear=tmp_df.base_TransformedCentroid_y)

df1a = df1.ix[np.abs(df1.diff_s_to_n) > 5.]
df2a = df2.ix[np.abs(df2.diff_s_to_n) > 5.]
print df1.shape, df2.shape, float(df1.shape[0])/float(df2.shape[0]), df1a.shape, df2a.shape

dist = np.sqrt(np.add.outer(df1a.coord_dec, -df2a.coord_dec)**2. +                np.add.outer(df1a.coord_ra, -df2a.coord_ra)**2.) * 206264.806247  # convert to arcsec ?
print dist.shape

print np.sum(dist < 0.5)
matches = np.where(dist < 0.5)
#print matches
print dist[0,0], df1a.iloc[0].coord_dec, df1a.iloc[0].coord_ra, '\t', df2a.iloc[0].coord_dec, df2a.iloc[0].coord_ra

matches1 = df1a.iloc[matches[0]]
matches2 = df2a.iloc[matches[1]]

fluxes1 = matches1.diffim_base_PsfFlux_flux.values
fluxes2 = matches2.diffim_base_PsfFlux_flux.values

fluxSigs1 = matches1.diffim_base_PsfFlux_fluxSigma.values
fluxSigs2 = matches2.diffim_base_PsfFlux_fluxSigma.values

isgood = ~np.isnan(fluxes1) & ~np.isnan(fluxes2)
isgood = isgood & (np.abs(fluxes1/fluxSigs1) > 5.0)
isgood = isgood & ~np.isnan(fluxSigs1) & ~np.isnan(fluxSigs2) 
isgood = isgood & (np.abs(fluxes2/fluxSigs2) > 5.0)

fluxes1 = fluxes1[isgood]
fluxes2 = fluxes2[isgood]
fluxSigs1 = np.sqrt(fluxSigs1[isgood])
fluxSigs2 = np.sqrt(fluxSigs2[isgood])

pars, cov = np.polyfit(fluxes1, fluxes2, deg=1, cov=True)
print pars, np.sqrt(np.diag(cov))
#print pearsonr(fluxes2, fluxes1)
print np.median(fluxes2/fluxes1), np.std(fluxes2/fluxes1)

pars, cov = np.polyfit(fluxSigs1, fluxSigs2, deg=1, cov=True)
print pars, np.sqrt(np.diag(cov))
#print pearsonr(fluxSigs2, fluxSigs1)
print np.median(fluxSigs2/fluxSigs1)

pars, cov = np.polyfit(fluxes1/fluxSigs1, fluxes2/fluxSigs2, deg=1, cov=True)
print pars, np.sqrt(np.diag(cov))
#print pearsonr(fluxSigs2, fluxSigs1)
print np.median((fluxes2/fluxSigs2)/(fluxes1/fluxSigs1)), np.std((fluxes2/fluxSigs2)/(fluxes1/fluxSigs1))

plt.figure(1, (16,8))
plt.subplot(131)
plt.plot(fluxes1, fluxes2, 'o')
plt.xlim(-10000, 10000); plt.ylim(-10000, 10000); plt.xlabel('Flux, uncorrected'); plt.ylabel('Flux, corrected')
plt.subplot(132)
plt.plot(fluxSigs1, fluxSigs2, 'o')
plt.ylim(8, 11); plt.xlabel('FluxSigma, uncorrected'); plt.ylabel('FluxSigma, corrected')
plt.subplot(133)
plt.plot((fluxes1/fluxSigs1), fluxes2/fluxSigs2, 'o')
plt.xlim(-7000, 7000); plt.ylim(-7000, 7000); plt.xlabel('SNR, uncorrected'); plt.ylabel('SNR, corrected')

import statsmodels.api as smapi
import statsmodels.graphics as smgraphics
regression = smapi.OLS(fluxes1, fluxes2).fit()
print regression.params
# Find outliers #
#test = regression.outlier_test()
#test2 = np.array([t[2] for t in test])
#print (test2 < 0.5).sum()
figure = smgraphics.regressionplots.plot_fit(regression, 0)
plt.xlim(-1000, 1000); plt.ylim(-1000, 1000)

import os
x = np.array(os.popen("grep -R 'Variance (template)' decam_lzogy_logs/* | awk '{print $4}'").read().split('\n'), dtype='|S')
templ_vars = x[x != ''].astype(float)
x = np.array(os.popen("grep -R 'Variance (science)' decam_lzogy_logs/* | awk '{print $4}'").read().split('\n'), dtype='|S')
sci_vars = x[x != ''].astype(float)
x = np.array(os.popen("grep -R 'Variance (uncorrected diffim)' decam_lzogy_logs/* | awk '{print $5}'").read().split('\n'), dtype='|S')
uncor_vars = x[x != ''].astype(float)
x = np.array(os.popen("grep -R 'Variance (corrected diffim)' decam_lzogy_logs/* | awk '{print $5}'").read().split('\n'), dtype='|S')
cor_vars = x[x != ''].astype(float)

df = pd.DataFrame({'templ': templ_vars, 'science': sci_vars, 'uncorrected': uncor_vars, 'corrected': cor_vars,
                  'templ+sci': templ_vars+sci_vars})
df.plot.hist(alpha=0.5, bins=200)
plt.xlim(50, 400)

df['templ+sci'] = templ_vars + sci_vars
#plt.plot(templ_vars + sci_vars, cor_vars, 'o')
df.plot.scatter(x='templ+sci', y='corrected')
plt.xlim(100, 1200); plt.ylim(100, 800)

import os
x = np.array(os.popen("grep -R 'Merging detections into' decam_lzogy_logs/* | awk '{print $5}'").read().split('\n'), dtype='|S')
detections = x[x != ''].astype(float)
print len(detections[detections >= 1000.])
x = np.array(os.popen("grep -R 'Merging detections into' decam_rescaled_diffims_logs/* | awk '{print $5}'").read().split('\n'), dtype='|S')
detections2 = x[x != ''].astype(float)
print len(detections2[detections2 >= 1000.])
print len(detections), len(detections2)
print detections.sum(), detections2.sum(), detections2.sum()/detections.sum()
df['ndet_corrected'] = detections
df['ndet_uncorrected'] = detections2
df[['ndet_corrected','ndet_uncorrected']].plot.hist(alpha=0.5, bins=500);
plt.xlim(0, 1000)

plt.plot(np.log10(detections2), np.log10(detections), 'o')
plt.title('log10(Number of detections)'); plt.xlabel('Uncorrected'); plt.ylabel('Corrected')

x = np.array(os.popen("grep -R 'imageDifference: Processing OrderedDict' decam_lzogy_logs/* | awk '{print $4}'").read().split('\n'), dtype='|S')
science_visits = np.array([xx.rstrip('),') for xx in x[x != '']]).astype(int)
x = np.array(os.popen("grep -R 'imageDifference: Processing OrderedDict' decam_lzogy_logs/* | awk '{print $6}'").read().split('\n'), dtype='|S')
science_ccds = np.array([xx.rstrip(')]))') for xx in x[x != '']]).astype(int)
x = np.array(os.popen("grep -R 'imageDifference.getTemplate: Fetching calexp' decam_lzogy_logs/* | awk '{print $5}'").read().split('\n'), dtype='|S')
templ_visits = np.array([xx.rstrip('),') for xx in x[x != '']]).astype(int)
x = np.array(os.popen("grep -R 'imageDifference.getTemplate: Fetching calexp' decam_lzogy_logs/* | awk '{print $7}'").read().split('\n'), dtype='|S')
templ_ccds = np.array([xx.rstrip(')]))') for xx in x[x != '']]).astype(int)

import cPickle
import gzip
if False:
    cPickle.dump((science_visits, science_ccds, templ_visits, templ_ccds, templ_vars, sci_vars, uncor_vars, cor_vars), gzip.GzipFile('20. compare photometry-corrected-many-DECam-images-logfileData.p.gz', 'wb'))
else:
    science_visits, science_ccds, templ_visits, templ_ccds, templ_vars, sci_vars, uncor_vars, cor_vars = cPickle.load(gzip.GzipFile('20. compare photometry-corrected-many-DECam-images-logfileData.p.gz', 'rb'))

def noise_detections(nu, sigma_g=1.8, npix=2000.*4000):
    out = nu * np.exp(-(nu**2.)/2) / (2.**(5./2.) * np.pi**(3./2.))
    out *= 1. / sigma_g**2. * npix
    return out

print noise_detections(5.)

visit = 197367
df1a = df1[df1.visit == visit]
df2a = df2[df2.visit == visit]
print df1a.shape, df2a.shape

x = np.linspace(2, 8)
nd = noise_detections(x)
## Need to scale by number of visits and ccd's used... and times 2 since we include pos. and neg. detections
print np.sum(science_visits == visit)
nd *= np.sum(science_visits == visit) * 2.

import warnings
warnings.filterwarnings("ignore")

df1a['abs_diff_s_to_n'] = df1a[['diff_s_to_n']].abs()
df1a['abs_diffim_s_to_n'] = df1a[['diffim_base_PsfFlux_flux']].abs() / df1a[['diffim_base_PsfFlux_fluxSigma']].values
df1a.loc[(df1a.abs_diff_s_to_n < 10.) & (df1a.abs_diffim_s_to_n < 10.)][['abs_diff_s_to_n', 'abs_diffim_s_to_n']].plot.hist(alpha=0.5, bins=np.arange(2, 8.2, 0.2))
plt.xlim(2., 8.); plt.ylim(0, 5000)
plt.plot(x, nd, 'k-')

df2a['abs_diff_s_to_n'] = df2a[['diff_s_to_n']].abs()
df2a['abs_diffim_s_to_n'] = df2a[['diffim_base_PsfFlux_flux']].abs() / df2a[['diffim_base_PsfFlux_fluxSigma']].values
df2a.loc[(df2a.abs_diff_s_to_n < 10.) & (df2a.abs_diffim_s_to_n < 10.)][['abs_diff_s_to_n', 'abs_diffim_s_to_n']].plot.hist(alpha=0.5, bins=np.arange(2, 8.2, 0.2))
plt.xlim(2., 8.); plt.ylim(0, 150)
plt.plot(x, nd, 'k-')

plt.rcParams['figure.figsize'] = (10.0, 6.5)
tmp_df = df1a.loc[(df1a.abs_diff_s_to_n < 10.) & (df1a.abs_diffim_s_to_n < 10.)][['abs_diff_s_to_n', 'abs_diffim_s_to_n', 'classification_dipole']]
tmp_df.plot.scatter(x='abs_diff_s_to_n', y='abs_diffim_s_to_n', s=50) #, c='classification_dipole')
plt.xlim(3., 8.); plt.ylim(3., 8.)
x = np.linspace(-25,25)
plt.plot(x, x, 'k-')
plt.xlabel('Expected SNR'); plt.ylabel("Measured SNR in diffim")

tmp_df = tmp_df.loc[tmp_df.classification_dipole == False][['abs_diff_s_to_n', 'abs_diffim_s_to_n']]
tmp_df.plot.scatter(x='abs_diff_s_to_n', y='abs_diffim_s_to_n', s=50)
plt.xlim(3., 8.); plt.ylim(3., 8.)
x = np.linspace(-25,25)
plt.plot(x, x, 'k-')
plt.xlabel('Expected SNR'); plt.ylabel("Measured SNR in diffim")

dist = np.sqrt(np.add.outer(df1a.coord_dec, -df2a.coord_dec)**2. +                np.add.outer(df1a.coord_ra, -df2a.coord_ra)**2.) * 206264.806247  # convert to arcsec ?
print dist.shape

print np.sum(dist < 2.5)
matches = np.where(dist < 2.5)
#print matches
print dist[0,0], df1a.iloc[0].coord_dec, df1a.iloc[0].coord_ra, '\t', df2a.iloc[0].coord_dec, df2a.iloc[0].coord_ra

matches1 = df1a.iloc[matches[0]]
matches2 = df2a.iloc[matches[1]]

print df1a.shape, df2a.shape
print matches1.shape, matches2.shape

plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelcolor'] = "#000000"
plt.rcParams['figure.figsize'] = (10.0, 6.5)
tmp_df = df2a.loc[(df2a.abs_diff_s_to_n < 10.) & (df2a.abs_diffim_s_to_n < 10.)][['abs_diff_s_to_n', 'abs_diffim_s_to_n']] #, 'classification_dipole']]
tmp_df.plot.scatter(x='abs_diff_s_to_n', y='abs_diffim_s_to_n', s=50) #, c='classification_dipole')
plt.xlim(3., 8.); plt.ylim(3., 8.)
plt.plot(x, x, 'k-')
plt.xlabel('Expected SNR'); plt.ylabel("Measured SNR in diffim")

plt.rcParams['figure.figsize'] = (10.0, 6.5)
tmp_df = matches2.copy()
tmp_df['classification_dipole'] = matches1.classification_dipole.values
tmp_df = tmp_df.loc[(tmp_df.abs_diff_s_to_n < 10.) & (tmp_df.abs_diffim_s_to_n < 10.)][['abs_diff_s_to_n', 'abs_diffim_s_to_n', 'classification_dipole']]
tmp_df = tmp_df.loc[tmp_df.classification_dipole == False][['abs_diff_s_to_n', 'abs_diffim_s_to_n']]
tmp_df.plot.scatter(x='abs_diff_s_to_n', y='abs_diffim_s_to_n', s=50) #, c='classification_dipole')
plt.xlim(3., 8.); plt.ylim(3., 8.)
plt.plot(x, x, 'k-')
plt.xlabel('Expected SNR'); plt.ylabel("Measured SNR in diffim")

print df1.shape, df1.classification_dipole.sum()
print df2.shape, df2.ip_diffim_DipoleFit_flag_classification.sum()
sizeme(df2a.head())

print np.in1d(84768485325931010, df1.objectId.values)[0]
print np.isnan(df2a.diffim_base_PsfFlux_flux.values)
objIds = df2a.loc[(df2a.ip_diffim_DipoleFit_flag_classification == 1) & (~np.isnan(df2a.diffim_base_PsfFlux_flux.values))].objectId.values

reload(dit)
dit.mosaicDIASources('decam_lzogy_forcephot', visitid=197367, ccdnum=1,                      sourceIds = np.array(objIds)[0:20])

objIds = df2a.loc[(df2a.ip_diffim_DipoleFit_flag_classification == 0) & (~np.isnan(df2a.diffim_base_PsfFlux_flux.values))].objectId.values

reload(dit)
dit.mosaicDIASources('decam_lzogy_forcephot', visitid=197367, ccdnum=8,                      sourceIds = np.array(objIds)[0:20])




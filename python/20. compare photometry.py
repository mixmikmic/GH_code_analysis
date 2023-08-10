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

import lsst.daf.persistence as dp
if False:
    butler = dp.Butler('decamDirTest')
    sources = butler.get('deepDiff_diaSrc',visit=289820,ccdnum=11)
    #print sources[0].extract('ip_diffim_Naive*')

if False:
# df1 will be the corrected one.
    df1 = pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})
    #df1.head()

if False:
# df2 will be the uncorrected one, run with 5.5-sigma threshold
    df2 = pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})
    #df2.head()

if True:
    print df1.shape, df2.shape

# Now let's save them out...
if True:
    import cPickle
    import gzip
    cPickle.dump((df1, df2), gzip.GzipFile('twoCatalogs.p.gz', 'wb'))

import cPickle
import gzip

df1, df2 = cPickle.load(gzip.GzipFile('twoCatalogs.p.gz', 'rb'))
print df1.shape, df2.shape, float(df2.shape[0])/float(df1.shape[0]), 395./float(df1.shape[0])
#print df1.columns.values

sizeme(df1.head())

#print df1.columns.values
fluxes1 = df1.base_CircularApertureFlux_50_0_flux
fluxes2 = df2.base_CircularApertureFlux_50_0_flux
fluxes1 = np.append(fluxes1, np.repeat(np.nan, len(fluxes2)-len(fluxes1)))
print len(fluxes2),len(fluxes1)
df = pd.DataFrame({'corr': fluxes1, 'orig': fluxes2})
df.plot.hist(alpha=0.5, bins=20)

dist = np.sqrt(np.add.outer(df1.coord_dec, -df2.coord_dec)**2. +                np.add.outer(df1.coord_ra, -df2.coord_ra)**2.) * 206264.806247  # convert to arcsec ?
print dist.min(), np.unravel_index(np.argmin(dist), dist.shape), dist[18,48]
print df1.iloc[18].coord_dec, df1.iloc[18].coord_ra, '\t', df2.iloc[48].coord_dec, df2.iloc[48].coord_ra
print np.sum(dist < 4.)
matches = np.where(dist < 4.)
#print matches
print dist[0,0], df1.iloc[0].coord_dec, df1.iloc[0].coord_ra, '\t', df2.iloc[0].coord_dec, df2.iloc[0].coord_ra

matches1 = df1.iloc[matches[0]]
matches2 = df2.iloc[matches[1]]

fluxes1 = matches1.base_CircularApertureFlux_50_0_flux.values
fluxes2 = matches2.base_CircularApertureFlux_50_0_flux.values

fluxSigs1 = matches1.base_CircularApertureFlux_50_0_fluxSigma.values
fluxSigs2 = matches2.base_CircularApertureFlux_50_0_fluxSigma.values

isgood = ~np.isnan(fluxes1) & ~np.isnan(fluxes2)
isgood2 = ~np.isnan(fluxes1) & ~np.isnan(fluxes2) & ~np.isnan(fluxSigs1) & ~np.isnan(fluxSigs2)

fluxes1 = fluxes1[isgood]
fluxes2 = fluxes2[isgood]
fluxSigs1 = np.sqrt(fluxSigs1[isgood2])
fluxSigs2 = np.sqrt(fluxSigs2[isgood2])

pars, cov = np.polyfit(fluxes2, fluxes1, deg=1, cov=True)
print pars, np.sqrt(np.diag(cov))
#print pearsonr(fluxes2, fluxes1)
print np.median(fluxes1/fluxes2), np.std(fluxes1/fluxes2)

pars, cov = np.polyfit(fluxSigs2, fluxSigs1, deg=1, cov=True)
print pars, np.sqrt(np.diag(cov))
#print pearsonr(fluxSigs2, fluxSigs1)
print np.median(fluxSigs1/fluxSigs2)

pars, cov = np.polyfit(fluxes2/fluxSigs2, fluxes1/fluxSigs1, deg=1, cov=True)
print pars, np.sqrt(np.diag(cov))
#print pearsonr(fluxSigs2, fluxSigs1)
print np.median((fluxes1/fluxSigs1)/(fluxes2/fluxSigs2)), np.std((fluxes1/fluxSigs1)/(fluxes2/fluxSigs2))

plt.figure(1, (12,4))
plt.subplot(131)
plt.plot(fluxes2, fluxes1, 'o')
plt.xlabel('Flux, uncorrected'); plt.ylabel('Flux, corrected')
#plt.ylim(-10000, 10000)
plt.subplot(132)
plt.plot(fluxSigs2, fluxSigs1, 'o')
plt.xlabel('FluxSigma, uncorrected'); plt.ylabel('FluxSigma, corrected')
plt.subplot(133)
plt.plot((fluxes2/fluxSigs2), fluxes1/fluxSigs1, 'o')
plt.xlabel('SNR, uncorrected'); plt.ylabel('SNR, corrected')
#plt.ylim(-100, 100); plt.xlim(0, 5)

df1.columns.values

print np.mean(df1.ip_diffim_DipoleFit_flag_classification.values)
print np.mean(df2.ip_diffim_DipoleFit_flag_classification.values)
print np.mean(matches1.ip_diffim_DipoleFit_flag_classification.values)
print np.mean(matches2.ip_diffim_DipoleFit_flag_classification.values)
print;
print np.mean(df1.ip_diffim_ClassificationDipole_value)
print np.mean(df2.ip_diffim_ClassificationDipole_value)
print np.mean(matches1.ip_diffim_ClassificationDipole_value)
print np.mean(matches2.ip_diffim_ClassificationDipole_value)

import lsst.daf.persistence as dp
butler=dp.Butler('decamDirTest')
sources=butler.get('forced_src',visit=289820,ccdnum=11)
df = pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})
print df.shape
sizeme(df.head())

df[['base_PsfFlux_flux', 'template_base_PsfFlux_flux']].head()
df['s_to_n'] = df.base_PsfFlux_flux / df.base_PsfFlux_fluxSigma
df['template_s_to_n'] = df.template_base_PsfFlux_flux / df.template_base_PsfFlux_fluxSigma
print df.columns.values
df[['s_to_n', 'template_s_to_n']].head()

df.plot.scatter('s_to_n', 'template_s_to_n')
plt.xlim(-10, 25)
plt.ylim(-10, 25)
x = np.linspace(-25,25)
plt.plot(x, x-5*np.sqrt(2), 'k-')
plt.plot(x, x+5*np.sqrt(2), 'k-')




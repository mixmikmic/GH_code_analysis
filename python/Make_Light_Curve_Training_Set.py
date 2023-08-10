from __future__ import division
import numpy as np
import astropy.io.fits as fits
import astropy.coordinates as coords
import astropy.units as u
from astropy.time import Time
import astropy.utils
from glob import glob
import matplotlib.pyplot as plt
import shelve, pickle
import uuid
from IPython.display import HTML, Javascript, display
import time
import FATS
get_ipython().magic('matplotlib inline')

# START with field 4163

reference_catalog = '../data/other_fields/4163/PTF_d004163_f02_c09_u000152621_p12_sexcat.ctlg'
# select R-band data (f02)
epoch_catalogs = glob('../data/other_fields/4163/PTF_2*f02*.ctlg')

def load_ref_catalog(reference_catalog):
    hdus = fits.open(reference_catalog)
    data = hdus[1].data
    # filter flagged detections
    w = ((data['flags'] & 506 == 0) & (data['MAG_AUTO'] < 99))
    data = data[w]

    ref_coords = coords.SkyCoord(data['X_WORLD'], data['Y_WORLD'],frame='icrs',unit='deg')

    star_class = np.array(data["CLASS_STAR"]).T
    
    return np.vstack([data['MAG_AUTO'],data['MAGERR_AUTO']]).T, ref_coords, star_class

ref_mags, ref_coords, star_class = load_ref_catalog(reference_catalog)

print "There are %s sources in the reference image" % len(ref_mags)
print "..."
print "There are %s epochs for this field" % len(epoch_catalogs)

def crossmatch_epochs(reference_coords, epoch_catalogs):
    
    n_stars = len(reference_coords)
    n_epochs = len(epoch_catalogs)
    
    mags = np.ma.zeros([n_stars, n_epochs])
    magerrs = np.ma.zeros([n_stars, n_epochs])
    mjds = np.ma.zeros(n_epochs)
    
    with astropy.utils.console.ProgressBar(len(epoch_catalogs),ipython_widget=True) as bar:
        for i, catalog in enumerate(epoch_catalogs):
            hdus = fits.open(catalog)
            data = hdus[1].data
            hdr = hdus[2].header
            # filter flagged detections
            w = ((data['flags'] & 506 == 0) & (data['imaflags_iso'] & 1821 == 0))
            data = data[w]

            epoch_coords = coords.SkyCoord(data['X_WORLD'], data['Y_WORLD'],frame='icrs',unit='deg')
            idx, sep, dist = coords.match_coordinates_sky(epoch_coords, reference_coords)
        
            wmatch = (sep <= 1.5*u.arcsec)
        
        # store data
            if np.sum(wmatch):
                mags[idx[wmatch],i] = data[wmatch]['MAG_APER'][:,2] + data[wmatch]['ZEROPOINT']
                magerrs[idx[wmatch],i] = data[wmatch]['MAGERR_APER'][:,2]
                mjds[i] = hdr['OBSMJD']

            bar.update()
    return mjds, mags, magerrs

mjds,mags,magerrs = crossmatch_epochs(ref_coords, epoch_catalogs)

wbad = (mags < 10) | (mags > 25)
mags[wbad] = np.ma.masked
magerrs[wbad] = np.ma.masked

def relative_photometry(ref_mags, star_class, mags, magerrs):
    #make copies, as we're going to modify the masks
    all_mags = mags.copy()
    all_errs = magerrs.copy()
    
    # average over observations
#     medmags = np.ma.median(all_mags,axis=1)  # use the mag in the reference image
    refmags = np.ma.array(ref_mags[:,0])
#     stdmags = np.ma.std(all_mags,axis=1)     # use outlier resistant median absolute deviation
    madmags = 1.48*np.ma.median(np.abs(all_mags - np.ma.median(all_mags, axis = 1).reshape(len(ref_mags),1)), axis = 1)
    MSE = np.ma.mean(all_errs**2.,axis=1)

    # exclude bad stars: highly variable, saturated, or faint
    # use excess variance to find bad objects
    excess_variance = madmags**2. - MSE
    wbad = np.where((np.abs(excess_variance) > 0.1) | (refmags < 14.5) | (refmags > 17) | (star_class < 0.9))
    # mask them out
    refmags[wbad] = np.ma.masked
    
    # exclude stars that are not detected in a majority of epochs
    Nepochs = len(all_mags[0,:])
    nbad = np.where(np.ma.sum(all_mags > 1, axis = 1) <= Nepochs/2.)
    refmags[nbad] = np.ma.masked

    # for each observation, take the median of the difference between the median mag and the observed mag
    # annoying dimension swapping to get the 1D vector to blow up right
    relative_zp = np.ma.median(all_mags - refmags.reshape((len(all_mags),1)),axis=0)

    return relative_zp

# compute the relative photometry and subtract it. Don't fret about error propagation
rel_zp = relative_photometry(ref_mags, star_class, mags, magerrs)
mags -= np.ma.resize(rel_zp, mags.shape)

outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')
shelf = shelve.open('../data/'+outfile,flag='c',protocol=pickle.HIGHEST_PROTOCOL)
shelf['mjds'] = mjds
shelf['mags'] = mags
shelf['magerrs'] = magerrs
shelf['ref_coords'] = ref_coords
shelf.close()

def source_lightcurve(rel_phot_shlv, ra, dec, matchr = 1.0):
    """Crossmatch ra and dec to a PTF shelve file, to return light curve of a given star"""
    shelf = shelve.open(rel_phot_shlv)
    ref_coords = coords.SkyCoord(shelf["ref_coords"].ra, shelf["ref_coords"].dec,frame='icrs',unit='deg')    
    
    source_coords = coords.SkyCoord(ra, dec,frame='icrs',unit='deg')
    idx, sep, dist = coords.match_coordinates_sky(source_coords, ref_coords)        
    
    wmatch = (sep <= matchr*u.arcsec)
    
    if sum(wmatch) == 1:
        mjds = shelf["mjds"]
        mags = shelf["mags"][idx]
        magerrs = shelf["magerrs"][idx]
        
        return mjds, mags, magerrs

    else:
        return "There are no matches to the provided coordinates within %.1f arcsec" % (matchr)

reference_catalog = '../data/other_fields/4163/PTF_d004163_f02_c09_u000152621_p12_sexcat.ctlg'
outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')

ra = np.array([253.177886, 252.830368, 253.063609, 253.286147])
dec = np.array([32.266276, 32.02584, 31.901347, 32.535967])
RRLfeats = []
for r, d in zip(ra, dec):
    source_mjds, source_mags, source_magerrs = source_lightcurve('../data/'+outfile, r, d)
    [mag, time, error] = FATS.Preprocess_LC(source_mags, source_mjds, source_magerrs).Preprocess()

    lc = np.array([mag, time, error])
    feats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error']).calculateFeature(lc)
    feat_row = np.reshape(feats.result(method='array'), (1,59))
        
    if len(RRLfeats) == 0:
        RRLfeats = feat_row
    else:
        RRLfeats = np.append(RRLfeats, feat_row, axis = 0)

reference_catalog = '../data/other_fields/4163/PTF_d004163_f02_c09_u000152621_p12_sexcat.ctlg'
outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')

ra = np.array([252.97529, 252.92493,252.97529, 252.88804, 253.33945, 253.10062, 253.3395, 253.4526])
dec = np.array([31.551398, 32.244859, 31.551391,31.707791, 31.538562,32.017281,31.538542,31.518162])
QSOfeats = []
for r, d in zip(ra, dec):
    source_mjds, source_mags, source_magerrs = source_lightcurve('../data/'+outfile, r, d)
    [mag, time, error] = FATS.Preprocess_LC(source_mags, source_mjds, source_magerrs).Preprocess()

    lc = np.array([mag, time, error])
    feats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error']).calculateFeature(lc)
    feat_row = np.reshape(feats.result(method='array'), (1,59))
        
    if len(QSOfeats) == 0:
        QSOfeats = feat_row
    else:
        QSOfeats = np.append(QSOfeats, feat_row, axis = 0)

ra, dec = 253.4526,31.518162
shelf = shelve.open('../data/'+outfile)
ref_coords = coords.SkyCoord(shelf["ref_coords"].ra, shelf["ref_coords"].dec,frame='icrs',unit='deg')    
    
source_coords = coords.SkyCoord(ra, dec,frame='icrs',unit='deg')
idx, sep, dist = coords.match_coordinates_sky(source_coords, ref_coords)        
    
wmatch = (sep <= 1*u.arcsec)

wmatch

## 2nd with field 3696

reference_catalog = '../data/other_fields/3696/PTF_d003696_f02_c06_u000154869_p12_sexcat.ctlg'
# select R-band data (f02)
epoch_catalogs = glob('../data/other_fields/3696/PTF_2*f02*.ctlg')

ref_mags, ref_coords, star_class = load_ref_catalog(reference_catalog)

print "There are %s sources in the reference image" % len(ref_mags)
print "..."
print "There are %s epochs for this field" % len(epoch_catalogs)

mjds,mags,magerrs = crossmatch_epochs(ref_coords, epoch_catalogs)

wbad = (mags < 10) | (mags > 25)
mags[wbad] = np.ma.masked
magerrs[wbad] = np.ma.masked

# compute the relative photometry and subtract it. Don't fret about error propagation
rel_zp = relative_photometry(ref_mags, star_class, mags, magerrs)
mags -= np.ma.resize(rel_zp, mags.shape)

outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')
shelf = shelve.open('../data/'+outfile,flag='c',protocol=pickle.HIGHEST_PROTOCOL)
shelf['mjds'] = mjds
shelf['mags'] = mags
shelf['magerrs'] = magerrs
shelf['ref_coords'] = ref_coords
shelf.close()

reference_catalog = '../data/other_fields/3696/PTF_d003696_f02_c06_u000154869_p12_sexcat.ctlg'
outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')

ra = np.array([253.65622, 253.589594])
dec = np.array([20.743137, 20.696087])
for r, d in zip(ra, dec):
    source_mjds, source_mags, source_magerrs = source_lightcurve('../data/'+outfile, r, d)
    [mag, time, error] = FATS.Preprocess_LC(source_mags, source_mjds, source_magerrs).Preprocess()

    lc = np.array([mag, time, error])
    feats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error']).calculateFeature(lc)
    feat_row = np.reshape(feats.result(method='array'), (1,59))
        
    if len(RRLfeats) == 0:
        RRLfeats = feat_row
    else:
        RRLfeats = np.append(RRLfeats, feat_row, axis = 0)

reference_catalog = '../data/other_fields/3696/PTF_d003696_f02_c06_u000154869_p12_sexcat.ctlg'
outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')

ra = np.array([253.47151, 253.50356, 253.50356, 253.59031, 253.6131,  253.78818, 253.86388, 253.86388, 253.91854, 253.92526])
dec = np.array([20.346592, 21.149149, 21.149149, 20.494092, 20.564037,  21.322201, 20.663265, 20.66327, 20.622025, 20.651342])
for r, d in zip(ra, dec):
    source_mjds, source_mags, source_magerrs = source_lightcurve('../data/'+outfile, r, d)
    [mag, time, error] = FATS.Preprocess_LC(source_mags, source_mjds, source_magerrs).Preprocess()

    lc = np.array([mag, time, error])
    feats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error']).calculateFeature(lc)
    feat_row = np.reshape(feats.result(method='array'), (1,59))
        
    if len(QSOfeats) == 0:
        QSOfeats = feat_row
    else:
        QSOfeats = np.append(QSOfeats, feat_row, axis = 0)

## 3rd with field 3696

reference_catalog = '../data/other_fields/22682/PTF_d022682_f02_c11_u000096411_p12_sexcat.ctlg'
# select R-band data (f02)
epoch_catalogs = glob('../data/other_fields/22682/PTF_2*f02*.ctlg')

ref_mags, ref_coords, star_class = load_ref_catalog(reference_catalog)

print "There are %s sources in the reference image" % len(ref_mags)
print "..."
print "There are %s epochs for this field" % len(epoch_catalogs)

mjds,mags,magerrs = crossmatch_epochs(ref_coords, epoch_catalogs)

wbad = (mags < 10) | (mags > 25)
mags[wbad] = np.ma.masked
magerrs[wbad] = np.ma.masked

# compute the relative photometry and subtract it. Don't fret about error propagation
rel_zp = relative_photometry(ref_mags, star_class, mags, magerrs)
mags -= np.ma.resize(rel_zp, mags.shape)

outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')
shelf = shelve.open('../data/'+outfile,flag='c',protocol=pickle.HIGHEST_PROTOCOL)
shelf['mjds'] = mjds
shelf['mags'] = mags
shelf['magerrs'] = magerrs
shelf['ref_coords'] = ref_coords
shelf.close()

reference_catalog = '../data/other_fields/22682/PTF_d022682_f02_c11_u000096411_p12_sexcat.ctlg'
outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')

ra = np.array([311.527209, 311.546485, 311.711246])
dec = np.array([-0.133972, -0.042581, -0.898823])
for r, d in zip(ra, dec):
    source_mjds, source_mags, source_magerrs = source_lightcurve('../data/'+outfile, r, d)
    [mag, time, error] = FATS.Preprocess_LC(source_mags, source_mjds, source_magerrs).Preprocess()

    lc = np.array([mag, time, error])
    feats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error']).calculateFeature(lc)
    feat_row = np.reshape(feats.result(method='array'), (1,59))
        
    if len(RRLfeats) == 0:
        RRLfeats = feat_row
    else:
        RRLfeats = np.append(RRLfeats, feat_row, axis = 0)

reference_catalog = '../data/other_fields/22682/PTF_d022682_f02_c11_u000096411_p12_sexcat.ctlg'
outfile = reference_catalog.split('/')[-1].replace('ctlg','shlv')

ra = np.array([311.54785, 311.59198, 311.81836])
dec = np.array([-0.60713278, -0.48408186, -0.28332211])
for r, d in zip(ra, dec):
    source_mjds, source_mags, source_magerrs = source_lightcurve('../data/'+outfile, r, d)
    [mag, time, error] = FATS.Preprocess_LC(source_mags, source_mjds, source_magerrs).Preprocess()

    lc = np.array([mag, time, error])
    feats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error']).calculateFeature(lc)
    feat_row = np.reshape(feats.result(method='array'), (1,59))
        
    if len(QSOfeats) == 0:
        QSOfeats = feat_row
    else:
        QSOfeats = np.append(QSOfeats, feat_row, axis = 0)

STARfeats = []
for mags, magerrs in zip(shelf['mags'], shelf['magerrs']):
    if (sum(mags.mask) - len(mags)) > -20 or np.ma.median(mags) > 19.5:
        continue
    else:
        lc_mag = mags
        lc_mjd = shelf['mjds']
        lc_magerr = magerrs

        [mag, time, error] = FATS.Preprocess_LC(lc_mag, lc_mjd, lc_magerr).Preprocess()

        lc = np.array([mag, time, error])
        feats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error']).calculateFeature(lc)
        feat_row = np.reshape(feats.result(method='array'), (1,59))
        
        if len(STARfeats) == 0:
            STARfeats = feat_row
        elif len(STARfeats) < 100:
            STARfeats = np.append(STARfeats, feat_row, axis = 0)
        else:
            break

TSfeats = np.append(STARfeats, QSOfeats, axis = 0)
TSfeats = np.append(TSfeats, RRLfeats, axis = 0)

TSlabels = np.empty(len(TSfeats), dtype = '|S4')
TSlabels[0:len(STARfeats)] = 'star'
TSlabels[len(STARfeats):len(STARfeats)+len(QSOfeats)] = 'qso'
TSlabels[-len(RRLfeats):] = 'rrl'

from astropy.table import Table
feat_table = Table(TSfeats, names = tuple(feats.result(method='features')))
feat_table['class'] = TSlabels
feat_table.write('../data/TS_PTF_feats.csv', format='csv')

feat_table

len(QSOfeats)




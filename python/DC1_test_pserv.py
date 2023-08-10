import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import sklearn.neighbors
import desc.pserv
import desc.pserv.utils as pserv_utils

import treecorr as tc

import glob

import healpy as hp

import time

import astropy.table

from multiprocessing import Pool

from scipy.stats import binned_statistic

import lsst.daf.persistence
DATA_DIR_DITHERED = "/global/cscratch1/sd/descdm/DC1/DC1-imsim-dithered/"
DATA_DIR_UNDITHERED = "/global/cscratch1/sd/descdm/DC1/DC1-imsim-undithered/"
DATA_DIR_PHOSIM ="/global/cscratch1/sd/descdm/DC1/DC1-phoSim-3a/"

table_name = 'StarTruth'
#projectId = 0
#projectName = 'DC1 imsim undithered'
projectId = 1
projectName = 'DC1 imsim dithered'

conn = desc.pserv.DbConnection(host='nerscdb04.nersc.gov',
                               port=3306,
                               database='DESC_DC1_Level_2')

star_truth = conn.get_pandas_data_frame('select * from StarTruth')
tree = sklearn.neighbors.KDTree(
    np.array(((star_truth['raICRS'].values*np.pi/180.,
               star_truth['decICRS'].values*np.pi/180.))).transpose())

query = '''select id, coord_ra, coord_dec,
           base_PsfFlux_flux, base_PsfFlux_fluxSigma
           from Coadd_Object where deblend_nChild=0
           and base_ClassificationExtendedness_value=0
           and projectId=%i limit 100000''' % projectId
query_u = '''select id, coord_ra, coord_dec,
           base_PsfFlux_flux, base_PsfFlux_fluxSigma
           from Coadd_Object where deblend_nChild=0
           and base_ClassificationExtendedness_value=0
           and projectId=%i limit 100000''' % 0
tstart = time.time()
imsim_stars = conn.get_pandas_data_frame(query)
imsim_stars_u = conn.get_pandas_data_frame(query_u)
print('query time:', time.time() - tstart)
candidates = np.array(((imsim_stars['coord_ra'].values,
                        imsim_stars['coord_dec'].values))).transpose()

candidates_u = np.array(((imsim_stars_u['coord_ra'].values,
                        imsim_stars_u['coord_dec'].values))).transpose()

offset, index = tree.query(candidates, k=1)

offset_u, index_u = tree.query(candidates_u, k=1)

x0 = star_truth['raICRS'].values[index][::50].flatten()
y0 = star_truth['decICRS'].values[index][::50].flatten()
dx = star_truth['raICRS'].values[index][::50].flatten()-imsim_stars['coord_ra'].values[::50].flatten()*180/np.pi
dy = star_truth['decICRS'].values[index][::50].flatten()-imsim_stars['coord_dec'].values[::50].flatten()*180/np.pi

x0_u = star_truth['raICRS'].values[index_u][::50].flatten()
y0_u = star_truth['decICRS'].values[index_u][::50].flatten()
dx_u = star_truth['raICRS'].values[index_u][::50].flatten()-imsim_stars_u['coord_ra'].values[::50].flatten()*180/np.pi
dy_u = star_truth['decICRS'].values[index_u][::50].flatten()-imsim_stars_u['coord_dec'].values[::50].flatten()*180/np.pi

# We are going to multiply the arrows times 20 to be able to see them
plt.quiver(x0,y0,dx,dy, scale_units='xy', angles='xy', scale=0.05)
plt.xlim(97.0,98.5)
plt.ylim(-30.5,-29.0)
plt.xlabel('RA [deg]')
plt.ylabel('DEC [deg]')

plt.quiver(x0_u,y0_u,dx_u,dy_u, scale_units='xy', angles='xy', scale=0.05)
plt.xlim(97.0,98.5)
plt.ylim(-30.5,-29.0)
plt.xlabel('RA [deg]')
plt.ylabel('DEC [deg]')

cat = tc.Catalog(ra=x0_u, dec=y0_u, g1=dx_u, g2=dy_u, ra_units='deg', dec_units='deg')
cat2 = tc.Catalog(ra=x0, dec=y0, g1=dx, g2=dy, ra_units='deg', dec_units='deg')

gg = tc.GGCorrelation(bin_size=0.1, min_sep=1, max_sep=900, 
                            sep_units='arcmin', bin_slop=0.5)

gg.process(cat,cat)

plt.plot(gg.meanr/60,gg.xip,label=r'$\xi_{+}$')
plt.plot(gg.meanr/60,gg.xim,label=r'$\xi_{-}$')
plt.xscale('log')
plt.xlabel(r'$\theta$ [deg]')

gg_d = tc.GGCorrelation(bin_size=0.1, min_sep=1, max_sep=900, 
                            sep_units='arcmin', bin_slop=0.5)
gg_d.process(cat2,cat2)
plt.plot(gg_d.meanr/60,gg_d.xip,label=r'$\xi_{+}$')
plt.plot(gg_d.meanr/60,gg_d.xim,label=r'$\xi_{-}$')
plt.xscale('log')
plt.xlabel(r'$\theta$ [deg]')

plt.plot(gg.meanr,gg.xip,label=r'$\xi_{u,+}$')
#plt.plot(gg.meanr,gg.xim,label=r'$\xi_{u,-}$')
plt.plot(gg_d.meanr,gg_d.xip,'--',label=r'$\xi_{d,+}$')
plt.plot(3.5*60*np.ones(3),np.linspace(-3e-7,3e-7,3),'r--')
plt.plot(0.2/60*np.ones(3),np.linspace(-3e-7,3e-7,3),'r--')
#plt.plot(gg_d.meanr,gg_d.xim,'--',label=r'$\xi_{d,-}$')
plt.xscale('log')
plt.ylabel(r'$\xi_{+}$')
plt.xlabel(r'$\theta$ [arcmin]')

plt.plot(gg.meanr,gg.xim,label=r'$\xi_{u,-}$')
plt.plot(gg_d.meanr,gg_d.xim,'--',label=r'$\xi_{d,-}$')
plt.xscale('log')
plt.ylabel(r'$\xi_{-}$')
plt.xlabel(r'$\theta$ [arcmin]')

ccd_visit = conn.get_pandas_data_frame('select * from CcdVisit')

ccd_visit.keys()

plt.hist(ccd_visit['skyBg'],range=(0,1000),bins=500);
print np.nanmean(ccd_visit['skyBg'])

plt.hist(ccd_visit['seeing'],range=(0,3),bins=100);

butler = lsst.daf.persistence.Butler(DATA_DIR_DITHERED)
butler_u = lsst.daf.persistence.Butler(DATA_DIR_UNDITHERED)
butler_p = lsst.daf.persistence.Butler(DATA_DIR_PHOSIM)

print butler.getKeys("calexp")

dithered_visits = ccd_visit['projectId'].values==1

undithered_visits = ccd_visit['projectId'].values==0

phosim_visits = ccd_visit['projectId'].values==2

print np.count_nonzero(dithered_visits), np.count_nonzero(undithered_visits), np.count_nonzero(phosim_visits)

def get_ra_dec(entry_num,visit_table=ccd_visit[phosim_visits],b=butler_p):
    i=entry_num
    if i%1000==0:
        print i
    visitId = {'filter':visit_table['filterName'].values[i], 'raft':visit_table['raftName'].values[i],
               'sensor':visit_table['ccdName'].values[i], 'visit':visit_table['visitId'].values[i]}
    try:
        calexp = b.get("calexp",visitId,immediate=True)
        ra=calexp.getWcs().getSkyOrigin().toIcrs().getRa().asDegrees()
        dec=calexp.getWcs().getSkyOrigin().toIcrs().getDec().asDegrees()
    except:
        ra=np.nan
        dec=np.nan
    return ra,dec

get_ipython().magic('time get_ra_dec(10)')

p = Pool(4)
ra_dec_array_p = p.map(get_ra_dec, [i for i in range(40000)])

ra_dec_array_p = np.array(ra_dec_array_p)

len(ra_dec_array_p[:,0][~np.isnan(ra_dec_array_p[:,0])])

tab = astropy.table.Table([ra_dec_array_p[:,0],ra_dec_array_p[:,1]],names=('ra [deg]', 'dec [deg]'))
tab.write('phosim_pointings_ra_dec_40k.fits.gz', overwrite=True)

def make_hp_metric(visit_table,var,coord_array,metric='mean',nside=4096):
    good = np.logical_not(np.isnan(coord_array[:,0]))
    pix_nums = hp.ang2pix(nside,np.pi/2.-coord_array[:,1][good]*np.pi/180,coord_array[:,0][good]*np.pi/180)
    pix_counts = np.bincount(pix_nums,minlength=12*nside**2)
    pix_weight = np.bincount(pix_nums,weights=visit_table[var].values[good],minlength=12*nside**2)
    pix_weight2 = np.bincount(pix_nums,weights=visit_table[var].values[good]**2,minlength=12*nside**2)
    if metric=='mean':
        map_out = np.zeros_like(pix_counts)
        map_out[pix_counts!=0] = pix_weight[pix_counts!=0]/pix_counts[pix_counts!=0]
        return map_out
    if metric=='rms':
        map_out = np.zeros_like(pix_counts)
        map_out[pix_counts!=0] = np.sqrt(pix_weight2[pix_counts!=0]/pix_counts[pix_counts!=0]                                          -(pix_weight[pix_counts!=0]/pix_counts[pix_counts!=0])**2)
        return map_out
    if metric=='median':
        map_out, _ , _ = binned_statistic(pix_nums,visit_table[var].values[good],bins=12*nside**2,statistic=metric,range=(0,12*nside**2))
        return map_out
    else:
        print 'Only mean and rms defined!'

map_seeing = make_hp_metric(ccd_visit[undithered_visits],'seeing',ra_dec_array_u,nside=2048)
map_seeing_rms = make_hp_metric(ccd_visit[undithered_visits],'seeing',ra_dec_array_u,metric='rms',nside=2048)
map_seeing_median = make_hp_metric(ccd_visit[undithered_visits],'seeing',ra_dec_array_u,metric='median',nside=2048)

hp.gnomview(map_seeing, rot=(94, -28), reso=4, title='Mean seeing', max=1.5, min=0.4)

hp.gnomview(map_seeing_rms, rot=(94, -28), reso=4, title='RMS seeing')

hp.gnomview(map_seeing_median, rot=(94, -28), reso=4, title='Median seeing', max=1.5, unit='arcsec')

map_bg = make_hp_metric(ccd_visit[undithered_visits],'skyBg',ra_dec_array_u,nside=2048)
map_bg_rms = make_hp_metric(ccd_visit[undithered_visits],'skyBg',ra_dec_array_u,metric='rms',nside=2048)
map_gb_median = make_hp_metric(ccd_visit[undithered_visits],'skyBg',ra_dec_array_u,metric='median',nside=2048)

hp.gnomview(map_bg, rot=(94, -28), reso=4, title='Mean background',max=800, min=0)

hp.gnomview(map_gb_median, rot=(94, -28), reso=4, title='Median background',max=800, min=0,unit='ADU')




get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

import globalETAS
import global_etas_auto
import os
import pylab as plt
import multiprocessing as mpp
import datetime as dtm
import matplotlib.dates as mpd
import numpy
#
import pytz
from yodiipy import ANSStools as atp
#
# 2016-10-21 05:07:23 UTC 35.358°N   133.801°E 10.0 km depth

### 
#etas = global_etas_auto.auto_etas(dt_0=6, lon_center=179.055, lat_center=-37.401, d_lat_0=3., d_lon_0=3., fnameroot='etas_auto_global_2016_09_01',
#                                   kmldir='/home/myoder/Dropbox/Research/etas/etas_auto_2016_09_01',
#                                   to_dt=dtm.datetime(2016,8,31, tzinfo=pytz.timezone('UTC')))
#
# japan oct 2016
d_lat=3.
d_lon=3.
ev_lon, ev_lat = (133.801, 35.358)
t_now = dtm.datetime.now(pytz.timezone('UTC'))
m_c = 1.5
cat_len=3650.
#
lats = [ev_lat-d_lat, ev_lat+d_lat]
lons = [ev_lon-d_lon, ev_lon+d_lon]
#

#
cat_prelim = atp.catfromANSS(lon=lons, lat=lats, minMag=m_c, dates0=[t_now-dtm.timedelta(days=cat_len), t_now], 
                             Nmax=None, fout=None, rec_array=True)

#
f_dates = [x.astype(float) for x in cat_prelim['event_date']]
delta_ts = [x-f_dates[k] for k,x in enumerate(f_dates[1:])]

#print(cat_prelim.dtype.names)
d = cat_prelim['event_date'][0]
d.astype(dtm.datetime)
#
from mpl_toolkits.basemap import Basemap as Basemap
#
def basic_basemap(fignum=0, ax=None, fig_size=(6.,6.), map_resolution='i', map_projection='cyl', d_lon_range=None,
                  d_lat_range=None, lats=None, lons=None ):
	'''
	# plot contours over a map.
	# TODO: move a version of this to ANSS_tools, or another support module meant to be used with ANSS_tools
    # to be used as a standard diagnostic/pre-evaluation script.
	'''
	#
	# first, get contours:
	#etas_contours = self.calc_etas_contours(n_contours=n_contours, fignum=fignum, contour_fig_file=contour_fig_file, contour_kml_file=contour_kml_file, kml_contours_bottom=kml_contours_bottom, kml_contours_top=kml_contours_top, alpha_kml=alpha_kml, refresh_etas=refresh_etas)
	#
	# now, clear away the figure and set up the basemap...
	#
	d_lon_range = (d_lon_range or 1.)
	d_lat_range = (d_lat_range or 1.)
	#
    # TODO: sort out the default behavior
	if ax==None:
		fignum = (fignum or 0)
		ax = plt.gca()
		fg = plt.figure(fignum, fig_size)
		plt.clf()
	#
	cntr = [numpy.mean(lons), numpy.mean(lats)]
	#
	cm = Basemap(llcrnrlon=lons[0], llcrnrlat=lats[0], urcrnrlon=lons[1], urcrnrlat=lats[1],
                 resolution=map_resolution, projection=map_projection, lon_0=cntr[0], lat_0=cntr[1], ax=ax)
	#
	#cm.drawlsmask(land_color='0.8', ocean_color='b', resolution=map_resolution)
	cm.drawcoastlines(color='gray', zorder=1)
	cm.drawcountries(color='black', zorder=1)
	cm.drawstates(color='black', zorder=1)
	cm.drawrivers(color='blue', zorder=1)
	cm.fillcontinents(color='beige', lake_color='blue', zorder=0)
	# drawlsmask(land_color='0.8', ocean_color='w', lsmask=None, lsmask_lons=None, lsmask_lats=None, lakes=True, resolution='l', grid=5, **kwargs)
	#cm.drawlsmask(land_color='0.8', ocean_color='c', lsmask=None, lsmask_lons=None, lsmask_lats=None, lakes=True, resolution=self.mapres, grid=5)
	#
	#
	cm.drawmeridians(numpy.arange(int(lons[0]/d_lon_range)*d_lon_range, lons[1], d_lon_range),
                     color='k', labels=[0,0,1,1])
	cm.drawparallels(numpy.arange(int(lats[0]/d_lat_range)*d_lat_range, lats[1], d_lat_range),
                     color='k', labels=[1, 1, 0, 0])
	#
	return cm

fg  = plt.figure(figsize=(14,6))
ax0 = fg.add_axes([.1,.1,.35,.38])
ax1 = fg.add_axes([.1,.55,.35,.38], sharex=ax0)
ax2 = fg.add_axes([.5,.1, .35,.85])
#
cm = basic_basemap(fignum=None, ax=ax2, fig_size=None, map_resolution='i', map_projection='cyl', d_lon_range=None,
                  d_lat_range=None, lats=lats, lons=lons )
#
# mags:
ax0.vlines([x.astype(dtm.datetime) for x in cat_prelim['event_date']], ymin=[m_c-.5 for _ in cat_prelim], 
           ymax=cat_prelim['mag'], colors='b')
ax1.plot([x.astype(dtm.datetime) for x in cat_prelim['event_date'][1:]], delta_ts, ls='-', lw=2, marker='.')
cm0 = basic_basemap(lats=lats, lons=lons, map_resolution='i', map_projection='cyl', d_lon_range=None,
                  d_lat_range=None)
for ev in cat_prelim:
    cm0.scatter(ev['lon'], ev['lat'], marker='o', c='none', edgecolor='b', s=2.*ev['mag'])
#
#fg1 = plt.figure()
#ax3=plt.gca()
ax3 = fg.add_axes([.4,.7,.2, .3])
ax3.plot(sorted(cat_prelim['mag']), numpy.log(numpy.linspace(1., 0., len(cat_prelim))), ls='-', marker='.')
#ax0.set_xlabel('Event time $t$')
ax0.set_ylabel('Event Magnitude $m$')
ax1.set_ylabel('Interval $\Delta t$')
#

#
etas_range_factor=5.
etas_range_padding=.5
etas = globalETAS.ETAS_mpp(n_cpu=mpp.cpu_count(), lats=[ev_lat-d_lat-.5, ev_lat+d_lat], lons=[ev_lon-d_lon-.5,
                    ev_lon+d_lon], mc=m_c, transform_ratio_max=5., etas_range_factor=etas_range_factor,
                           etas_range_padding=etas_range_padding, t_now=t_now)
#

# get mainshock:
for ev in reversed(etas.catalog):
    if ev['mag']>6.:
        nz_mainshock = ev
        break

plt.figure(0, figsize=(10,8))
plt.clf()
ax=plt.gca()
etas.make_etas_contour_map(ax=ax, n_contours=25)
#
x,y = etas.cm(nz_mainshock['lon'], nz_mainshock['lat'])
etas.cm.scatter([x], [y], marker='o', edgecolor='b', facecolor='none', s=100)
#
out_path = '/home/myoder/Dropbox/Research/etas/japan_oct_2016/japan_oct_2016_{}'.format(t_now)
out_fname = 'japan_oct_2016_{}'.format(t_now)
if not os.path.isdir(out_path): os.makedirs(out_path)
#
etas.export_kml(os.path.join(out_path, '{}.kml'.format(out_fname)))
etas.export_xyz(os.path.join(out_path, '{}.xyz'.format(out_fname)))
plt.savefig((os.path.join(out_path, '{}.png'.format(out_fname))))

# can we run a brute-force (loop-loop) ETAS?
etas_brute = globalETAS.ETAS_mpp(n_cpu=mpp.cpu_count(), lats=[ev_lat-d_lat, ev_lat+d_lat], lons=[ev_lon-d_lon,
                    ev_lon+d_lon], mc=m_c, transform_ratio_max=5., t_now=t_now, worker_class=globalETAS.ETAS_brute)

# get mainshock:
for ev in reversed(etas_brute.catalog):
    if ev['mag']>6.:
        nz_mainshock = ev
        break

plt.figure(0, figsize=(10,8))
plt.clf()
ax=plt.gca()
etas_brute.make_etas_contour_map(ax=ax, n_contours=25)
#
x,y = etas_brute.cm(nz_mainshock['lon'], nz_mainshock['lat'])
etas_brute.cm.scatter([x], [y], marker='o', edgecolor='b', facecolor='none', s=100)
#
# out_path = '/home/myoder/Dropbox/Research/etas/japan_oct_2016/japan_oct_2016_{}'.format(t_now)
# out_fname = 'japan_oct_2016_{}'.format(t_now)
# if not os.path.isdir(out_path): os.makedirs(out_path)
# #
# etas_brute.export_kml(os.path.join(out_path, '{}.kml'.format(out_fname)))
# etas_brute.export_xyz(os.path.join(out_path, '{}.xyz'.format(out_fname)))
# plt.savefig((os.path.join(out_path, '{}.png'.format(out_fname))))






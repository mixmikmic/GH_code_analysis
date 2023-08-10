get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

import nepal_figs
import etas_analyzer

#aa=nepal_figs.toy_gs_roc()

diagnostic=True
etas_fc=None
etas_test=None
do_log=True
#do_log=False

#etas_test.make_etas()

#def analyze_etas_roc_geospatial(etas_fc=None, etas_test=None, do_log=True, diagnostic=False):
from etas_analyzer import *
if True:
	
	# do_log should pretty much always be True.
	# this script draws a bunch of geospatial ROC figures. we'll use this script to draw a quad-figure with
	# z_fc, z_test, hits, falsies.
	#
	if etas_fc   == None: etas_fc   = get_nepal_etas_fc(n_procs=2*mpp.cpu_count())
	if etas_test == None: etas_test = get_nepal_etas_test(n_procs=2*mpp.cpu_count())
	#

# splitting this script, just for development and maintenance purposes.
if True:
	f_quad = plt.figure(42, figsize=(12,12))
	plt.clf()
	ax0 = f_quad.add_axes([.05, .05, .4, .4])
	ax1 = f_quad.add_axes([.05, .55, .4, .4], sharex=ax0, sharey=ax0)
	ax2 = f_quad.add_axes([.55, .05, .4, .4], sharex=ax0, sharey=ax0)
	ax3 = f_quad.add_axes([.55, .55, .4, .4], sharex=ax0, sharey=ax0)		
	#
	# what we really want to do here is to calc_etas() (or whatever we call it). we do a full on _contour_map() so we can look at it.
	# in the end, to do the gs_roc, we just need the ETAS xyz array.
	etas_fc.make_etas_contour_map(fignum=0)
	etas_test.make_etas()
	etas_test.make_etas_contour_map(fignum=1)
	#
	lon_vals = sorted(list(set(etas_fc.ETAS_array['x'])))
	lat_vals = sorted(list(set(etas_fc.ETAS_array['y'])))
	#
	# we need normalization here...
	# ... and we need to think a bit more about what we mean by normalize. here, we just shift the values to be equal. do
	# we also want to normailze their range?
	z_fc_norm = etas_fc.ETAS_array['z'].copy()
	z_test_norm = etas_test.ETAS_array['z'].copy()
	#
	if do_log:
		z_fc_norm   = numpy.log10(z_fc_norm)
		z_test_norm = numpy.log10(z_test_norm)
	#
	#Normalize: first, subtract bias (shift to zero), then normalize total contribution --> 1.0
	z_fc_norm -= min(z_fc_norm)
	z_test_norm -= min(z_test_norm)
	#
	#norm_fc   = sum(z_fc_norm)
	#norm_test = sum(z_test_norm)
	#
	#z_fc_norm /= norm_fc
	#z_test_norm /= norm_test
	z_fc_norm /= sum(z_fc_norm)
	z_test_norm /= sum(z_test_norm)
	#
	z1 = z_fc_norm
	z2 = z_test_norm
	#
	#
	# [z1, z2, diff, h, m, f(predicted, didn't happen)
	#diffs = [[z1, z2, z1-z2, max(z1, z2), -min(z1-z2,0.), max(z1-z2,0.)] for z1,z2 in zip(z_fc_norm, z_test_norm)] 
	# hits: accurately predicted; min(z1,z2)
	# misses: prediction deficite, or excess events: min(z2-z1,0.)
	# falsie: excess prediction: min(z1-z2,0.)
	# then rates: H = hits/sum(z2), F =falsies/sum(z1)
	#diffs = [[z1, z2, z1-z2, min(z1, z2), max(z2-z1,0.), max(z1-z2, 0.)] for z1,z2 in zip(z_fc_norm, z_test_norm)]
	#
	# so we can test this properly, we'll want to move diffs offline to a function call (eventually)...
	
	#diffs = [[z1, z2, z1-z2, min(z1, z2), max(z2-z1,0.), max(z1-z2, 0.)] for z1,z2 in zip(z1, z2)]
	diffs = get_gs_diffs(z1,z2)
	diffs_lbls = ['z_fc', 'z_test', 'z1-z2', 'hits: min(z1,z2)','misses:min(z2-z1,0)', 'falsie: min(z1-z2,0)']
	diffs_lbl_basic = ['z_fc', 'z_test', 'z1-z2', 'hits','misses', 'falsie']
	#

	# to plot contours, we'll want to use the shape from: etas.lattice_sites.shape
	#
	sh1 = etas_fc.lattice_sites.shape
	sh2 = etas_test.lattice_sites.shape
	#
	print('shapes: ', sh1, sh2)
	#
	zs_diff, h, m, f = list(zip(*diffs))[2:]
	#
	# and ROC bits:
	# (i think really Molchan bits, aka, n_predicted/n_sites_covered); roc is a bit more subtle, but
	# they are approximately equal for large catalogs.
	# ... and this seems to be left over from something in the past; we don't use them.    
	#H = sum(h)/sum(z2)
	#F = sum(f)/sum(z1)
	#
	#for z in [zs_diff, h, m, f]:
	# plot the varous roc_gs contous (z1, z2, z2-z2, hits, etc.)
	for j,z in enumerate(list(zip(*diffs))):
		plt.figure(j+2)
		plt.clf()
		#
		zz=numpy.array(z)
		zz.shape=sh1
		#plt.contourf(list(set(etas_fc.ETAS_array['x'])), list(set(etas_fc.ETAS_array['y'])), zz, 25)
		#plt.contourf(numpy.log10(zz), 25)
		plt.contourf(lon_vals, lat_vals, zz, 25)
		plt.title(diffs_lbls[j])
		plt.colorbar()
		#
		# ... and make our quad-plot too:
		if j==0:
			ax1.contourf(lon_vals, lat_vals, zz, 25)
			ax1.set_title('Forecast ETAS')
			#ax1.colorbar()
		if j==1:
			ax3.contourf(lon_vals, lat_vals, zz, 25)
			ax3.set_title('Test ETAS')
			#ax3.colorbar()
		if j==3:
			ax0.contourf(lon_vals, lat_vals, zz, 25)
			ax0.set_title('Hit Rate')
			#ax0.colorbar()
		if j==5:
			ax2.contourf(lon_vals, lat_vals, zz, 25)
			ax2.set_title('False Alarm Rate')
			#ax2.colorbar()
	#
	#if diagnostic:
	#	print('***', diffs_lbls, type(diffs))
	#	#return [diffs_lbls] + diffs
	#	return diffs
	#else:
	#	return F,H
	##return F,H
	#

print(z_fc_norm[0:10], z1[0:10], z2[0:10])

#bb=etas_analyzer.nepal_linear_roc()
diffs = etas_analyzer.analyze_etas_roc_geospatial(etas_fc=None, etas_test=None, do_log=True, diagnostic=True)

AA=etas_analyzer.roc_gs_linear_figs(diffs)

j0=8000
#j1=len(diffs)
j1=9000
#
mydiffs = diffs[j0:j1]

#H = [min(z_fc, z_t) for z_fc, z_t in zip(mydiffs['z_fc'], mydiffs['z_test'])]

fg=plt.figure(figsize=(10,8))
plt.plot(mydiffs['z_fc'], label='forecast', color='b', lw=2.5)
plt.plot(mydiffs['z_test'], label='test', color='g', lw=2.5)
# hit rate:
plt.fill_between(range(len(mydiffs)), y1=numpy.zeros(len(mydiffs)), y2=mydiffs['hits'], color='c', alpha=.5,
                 label='hits')
#
# false alarm rate
plt.fill_between(range(len(mydiffs)), y1=mydiffs['z_test'], 
                 y2=[max(zfc, zt) for zfc,zt in zip(mydiffs['z_fc'], mydiffs['z_test'])], color='m',
                 alpha=.5, label='false-alarms')
#
# misses:
plt.fill_between(range(len(mydiffs)), y1=[min(zfc, zt) for zfc,zt in zip(mydiffs['z_fc'], mydiffs['z_test'])],
                 y2=mydiffs['z_test'], color='r', alpha=.5, label='misses')

#get the {somthing}% for a maximum y value:
y_max = sorted(mydiffs['z_fc'].tolist() + mydiffs['z_test'].tolist())[int(2*len(mydiffs)*.996)]
plt.gca().set_ylim(0.,y_max)


plt.legend(loc=0)

fg.savefig('/home/myoder/Dropbox/Research/globalETAS/nepal_etas/data_n_figs/linear_geospatial_roc_sample_figure.png')
f_quad.savefig('/home/myoder/Dropbox/Research/globalETAS/nepal_etas/data_n_figs/geospatial_roc_nepal_quad.png')

### Hits vs Falsies (subset) figure (probably not going into production)
print(mydiffs.dtype.names)
plt.figure(figsize=(10,8))
plt.plot(mydiffs['hits'])
plt.plot(mydiffs['falsie'])
#
#################################
# ROC plot of geospatial roc:
fg_roc = plt.figure(figsize=(10,8))
ax=fg_roc.gca()
H0 = max(diffs['hits'])
F0 = max(diffs['falsie'])
H0,F0 = 1., 1.
#
H = diffs['hits']/H0
F = diffs['falsie']/F0
ax.plot(F,H, marker='.', ls='')
#
# H=F diagonal (but allow for non-normalized and use a bunch of points in case we log-scale.)
ax.plot(numpy.linspace(0,max(list(F) + list(H)), 250), 
         numpy.linspace(0,max(list(F) + list(H)), 250), lw=2., color='r', ls='--')
h_f = [h-f for h,f in zip(H,F)]

h_f_m = numpy.mean(h_f)
std_h = numpy.std(h_f)
# separate greater- less-than stdevs to get asymmetrical uncertainty. sounds like a good idea, but the LT
# have greater uncertainty because of the small sample size. what we really want is a proper bayes analysis
# with fixed probability thresholds (aka, 30%, 70% boundaries). a bayes analysis should properly account for the
# asymmetry in the distribution.
#
#std_h_gt = numpy.std([x for x in h_f if x>0])
#std_h_lt = numpy.std([x for x in h_f if x<=0])
#
ax.set_xlabel('False Alarm Rate $F$', size=18)
ax.set_ylabel('Hit Rate $H$', size=18)
#
# plot "random" (H=F), mean(h-f), and random stdev range (<h-f> +/- stdev).
ax.plot(numpy.linspace(0,max(list(F) + list(H)), 250), 
         numpy.linspace(0,max(list(F) + list(H)), 250)+h_f_m, lw=2., color='r', ls='-.')

ax.fill_between(numpy.linspace(0,max(list(F) + list(H)), 250), 
         numpy.linspace(0,max(list(F) + list(H)), 250)+h_f_m-std_h, 
         numpy.linspace(0,max(list(F) + list(H)), 250)+h_f_m+std_h, color='r', alpha=.2)

#print('mean h-f: ', h_f_m, ' +/ ', std_h, std_h_gt, std_h_lt)
print('prob <0: ', len([x for x in h_f if x<=0])/len([x for x in h_f if x>0]))
#
##########################################
# roc of individual elements (eventually, i think the correct way to normalize this figuure is to normalize)
# each site so that its maximum value is 1. (nearly) equivalently (???), i think if we take what we've got and normalize
# so that for each element h+f=1, that might make sense. note that does not make sense for normal ROC, but maybe for
# this variant...
#
fg_hf_CDF_lin = plt.figure()
plt.clf()
ax=plt.gca()
X = sorted(h_f)
Y = numpy.arange(1,len(h_f)+1)/len(h_f)
ax.plot(X, Y, '-', lw=2.)
x_lt = [x for x in X if x<=0]
ax.fill_between(x_lt, numpy.zeros(len(x_lt)), Y[0:len(x_lt)], color='b', alpha=.2)
ax.set_xlabel('Skill score, $h-f$', size=18)
ax.set_ylabel('Probability $P(<[h-f])$', size=16)
ax.set_title('ROC geospatial, $P(h-f)$', size=14)
fg_hf_CDF_lin.savefig('/home/myoder/Dropbox/Research/globalETAS/nepal_etas/data_n_figs/roc_geospatial_elements_nepal_hf_dist_lin.png')
#
fg_hf_CDF_log = plt.figure()
plt.clf()
ax=plt.gca()
ax.set_yscale('log')
ax.plot(sorted(h_f), numpy.arange(1,len(h_f)+1)/len(h_f), '-', lw=2.)
ax.fill_between(x_lt, numpy.zeros(len(x_lt)), Y[0:len(x_lt)], color='b', alpha=.2)
ax.set_xlabel('Skill score, $h-f$', size=18)
ax.set_ylabel('Probability $P(<[h-f])$', size=16)
ax.set_title('ROC geospatial, $P(h-f)$', size=14)
fg_hf_CDF_log.savefig('/home/myoder/Dropbox/Research/globalETAS/nepal_etas/data_n_figs/roc_geospatial_elements_nepal_hf_dist_log.png')

# plot the ensemble ROC for the geospatial figures.
#
gs_roc_file = '/home/myoder/Dropbox/Research/globalETAS/nepal_etas/data_n_figs/roc_geospatial_raw.csv'
with open(gs_roc_file,'r') as f:
    # cols will be q_fc, q_test, F,H
    #
    HF_qq_gs = [[float(x) for x in rw.split()] for rw in f if not rw.startswith('#')]
    #
#
FH = [[x2,x3] for x0,x1,x2,x3 in HF_qq_gs]
plt.figure(figsize=(8,6))
plt.clf()
plt.plot(*zip(*FH), marker='o', ls='')
plt.plot(numpy.linspace(0,1,250), numpy.linspace(0,1,250), ls='-', lw=2.5, color='r', marker='')
plt.xlabel('False Alarm Rate $F$', size=16)
plt.ylabel('Hit Rate, $H$', size=16)
plt.title('Geospatial ROC (Nepal)', size=15)
plt.savefig('/home/myoder/Dropbox/Research/globalETAS/nepal_etas/data_n_figs/roc_geospatial_nepal.png')
#
xyz = [[q1, q1, h-f] for q1, q2, f,h in HF_qq_gs]




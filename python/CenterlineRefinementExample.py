get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
from SWOTRiver import SWOTL2
from RiverObs import ReachExtractor
from RiverObs import RiverObs
from RiverObs import FitRiver
from RiverObs import WidthDataBase
from RiverObs import IteratedRiverObs

get_ipython().run_line_magic('pylab', 'inline')
rcParams['font.size'] = 14

# This is optional, but it allows zooming and panning of figures
# Uncomment the block below if you want this feature

#try:
#    import mpld3
#    mpld3.enable_notebook()
#    print 'Interactive plotting enabled'
#except:
#    pass

import os
from os.path import exists, join
def find_riverobs_test_data_dir():
    """Fin the location of the test data root directory"""
    
    if 'RIVEROBS_TESTDATA_DIR' in os.environ:
        test_data_dir = os.environ('RIVEROBS_TESTDATA_DIR')
    else: # try the default location
        test_data_dir = '../../../RiverObsTestData'
        
    if not exists(test_data_dir):
        print('You must either set the environment variable RIVEROBS_TESTDATA_DIR')
        print('or locate the test data directory at ../../../RiverObsTestData')
        raise Exception('Test data directory not found.')
        
    return test_data_dir

data_dir = find_riverobs_test_data_dir()
data_dir

# This is the file for the width data base

db_dir = join(data_dir,'GRWDL')
width_db_file = join(db_dir,'nAmerica_GRWDL.h5')

# This is the SWOT data

l2_file = join(data_dir,'L2','L2v0','simulated_sacramento_swot_test_data_v0.nc')
assert exists(l2_file)

# This is the file for the reach data base

shape_file_root = join(db_dir,'nAmerica_GRWDL_river_topo','nAmerica_GRWDL_river_topo')

class_list=[1]
lat_kwd='no_layover_latitude'
lon_kwd='no_layover_longitude'

l2 = SWOTL2(l2_file,class_list=class_list,lat_kwd=lat_kwd,lon_kwd=lon_kwd)

h = l2.get('height')
htrue = l2.get('water_height')

clip_buffer = 0.02
reaches = ReachExtractor(shape_file_root, l2,clip_buffer=clip_buffer)

print('Reach indexes:',reaches.reach_idx)
print('Metadata:')
reaches[0].metadata

width_db = WidthDataBase(width_db_file)

max_width = width_db.get_river(reaches.reach_idx[0],
                                      columns=['width'],
                             asarray=True,transpose=False,
                             bounding_box=l2.bounding_box,
                             clip_buffer=clip_buffer)

print('max_width length:',len(max_width))
print('x length:',len(reaches[0].x))

figsize(8,8)
plot(l2.x/1.e3,l2.y/1.e3,'k.',alpha=0.05)
scatter(reaches[0].x/1.e3,reaches[0].y/1.e3,
        c=max_width,
        edgecolor='none',alpha=0.4,
        vmin=0,vmax=300,cmap='jet')
xlim(-12,-2)
ylim(-20,0)
colorbar(label='Width');

# First step, initialize observations

scalar_max_width = 600.

ds = 50.
minobs = 10
river_obs = IteratedRiverObs(reaches[0],l2.x,l2.y,
                         ds=ds,max_width=scalar_max_width,minobs=minobs) 

figsize(10,5)
subplot(1,2,1)
hist(river_obs.d,bins=100,log=False)
xlabel('Distance to node (m)')
ylabel('N observations')
grid();
subplot(1,2,2)
hist(river_obs.n,bins=100,log=False)
xlabel('Normal coordinate (m)')
ylabel('N observations')
grid()
tight_layout();

weights = True
smooth = 1.e-2
river_obs.iterate(weights=weights,smooth=smooth)

# retrieve the centerline coordinates

xc, yc = river_obs.get_centerline_xy()

figsize(8,8)
plot(l2.x/1.e3,l2.y/1.e3,'k.',alpha=0.05)
scatter(xc/1.e3,yc/1.e3,c='b',
        #c=max_width,
        edgecolor='none',alpha=0.4)#,
        #vmin=0,vmax=300)
xlim(-12,-2)
ylim(-20,0);

figsize(10,5)
subplot(1,2,1)
hist(river_obs.d,bins=100,log=False)
xlabel('Distance to node (m)')
ylabel('N observations')
grid();
subplot(1,2,2)
hist(river_obs.n,bins=100,log=False)
xlabel('Normal coordinate (m)')
ylabel('N observations')
grid()
tight_layout();

# These are the old centerline coordinates

xw = reaches[0].x
yw = reaches[0].y

# This step makes the association

river_obs.add_centerline_obs(xw,yw,max_width,'max_width')

xi, yi, wi = river_obs.get_centerline_xyv('max_width')

figsize(8,8)
plot(l2.x/1.e3,l2.y/1.e3,'k.',alpha=0.05)
scatter(xi/1.e3,yi/1.e3,
        c=wi,
        edgecolor='none',alpha=0.4,
        vmin=0,vmax=300,cmap='jet')
xlim(-12,-2)
ylim(-20,0);
colorbar(label='Width');

river_obs.reinitialize()

figsize(5,5)
scatter(river_obs.centerline.x/1.e3,river_obs.centerline.y/1.e3,
        c=river_obs.max_width,
        edgecolor='none',alpha=0.4,
        vmin=0,vmax=300,cmap='jet')
colorbar(label='Width');

river_obs.add_obs('htrue',htrue)
river_obs.add_obs('h',h)
river_obs.load_nodes(['h','htrue'])

hn_mean = np.array(river_obs.get_node_stat('mean','h'))
hn_median = np.array(river_obs.get_node_stat('median','h'))
hstdn = np.array(river_obs.get_node_stat('stderr','h'))
htn = np.array(river_obs.get_node_stat('mean','htrue'))
sn = np.array(river_obs.get_node_stat('mean','s'))

ave = np.mean(hn_mean - htn)*100
err = np.std(hn_mean - htn)*100
print('Mean statitics:   average: %.1f cm std: %.1f cm'%(ave,err))

ave = np.mean(hn_median - htn)*100
err = np.std(hn_median - htn)*100
print('Median statitics: average: %.1f cm std: %.1f cm'%(ave,err))

figsize(10,5)
subplot(1,2,1)
hist(river_obs.d,bins=100,log=False)
xlabel('Distance to node (m)')
ylabel('N observations')
grid();
subplot(1,2,2)
hist(river_obs.n,bins=100,log=False)
xlabel('Normal coordinate (m)')
ylabel('N observations')
grid()
tight_layout();

figsize(10,10)
subplot(2,2,1)
idx = river_obs.populated_nodes
plot(river_obs.centerline.x[idx]/1.e3,river_obs.centerline.y[idx]/1.e3,
     '.',alpha=0.5)
xlabel('x (km)')
ylabel('y (km)')
subplot(2,2,2)
idx = river_obs.populated_nodes
plot(river_obs.centerline.s[idx]/1.e3,river_obs.nobs[idx],'o',alpha=0.5)
grid()
xlabel('Reach (km)')
ylabel('Number of observations')
subplot(2,2,3)
plot(htrue,h,'.',alpha=0.05)
plot([-10,60],[-10,60],'--k',alpha=0.5)
xlim(-10,60)
ylim(-10,60)
grid()
xlabel('h true (m)')
ylabel('h measured (m)')
subplot(2,2,4)
plot(river_obs.s/1.e3,river_obs.htrue,'.',alpha=0.1)
plot(river_obs.s/1.e3,river_obs.h,'.',alpha=0.05)
grid()
xlabel('Reach (km)')
ylabel('Height (m)')
tight_layout();

figsize(10,5)
subplot(1,2,1)
plot(sn/1.e3,htn,'.',alpha=0.1)
plot(sn/1.e3,hn_mean,'.',alpha=0.2)
plot(sn/1.e3,htn+2*hstdn,'-k',alpha=0.1)
plot(sn/1.e3,htn-2*hstdn,'-k',alpha=0.1)
xlabel('Reach (km)')
ylabel('Height (m)')
title('Mean')

subplot(1,2,2)
plot(sn/1.e3,htn,'.',alpha=0.1)
plot(sn/1.e3,hn_median,'.',alpha=0.2)
plot(sn/1.e3,htn+2*hstdn,'-k',alpha=0.1)
plot(sn/1.e3,htn-2*hstdn,'-k',alpha=0.1)
xlabel('Reach (km)')
title('Median')
ylabel('Height (m)')
tight_layout();


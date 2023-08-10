get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('pylab', 'inline')
rcParams['font.size'] = 14

# This is optional, but it allows zooming and panning of figures
# Uncomment the block below if you want this feature

interactive = False # Set this to false if no iteractive data are desired.
if interactive:
    try:
        import mpld3
        from mpld3 import plugins
        mpld3.enable_notebook()
        print('Interactive plotting enabled')
    except:
        interactive = False

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

from SWOTRiver import SWOTL2
from RiverObs import ReachPreProcessor, RiverReachWriter

class_list=[1]
lat_kwd='no_layover_latitude'
lon_kwd='no_layover_longitude'

l2 = SWOTL2(l2_file,class_list=class_list,lat_kwd=lat_kwd,lon_kwd=lon_kwd)

clip_buffer = 0.02
reaches = ReachPreProcessor(shape_file_root, l2,clip_buffer=clip_buffer,width_db_file=width_db_file)

print('Number of reaches found:',reaches.nreaches)
print('Reach indexes:',reaches.reach_idx)
for i,reach in enumerate(reaches):
    print('Reach %d Metadata'%i)
    print(reach.metadata)

figsize(6,6)
plot(l2.lon[::10],l2.lat[::10],'o',alpha=0.1,color='aqua')
for reach in reaches:
    plot(reach.lon,reach.lat,'.',alpha=0.4,label='Reach %d'%i)
legend(loc='best')
title('Reaches vs Data')
if interactive:
    plugins.connect(gcf(),plugins.MousePosition(fmt='.3f'))

start_lons = [-122.010,-121.965,-121.962,-121.978]
start_lats = [39.760,39.735,39.685,39.648]
end_lons = [-121.965,-121.962,-121.978,-121.998]
end_lats = [39.735,39.685,39.648,39.601]

reach_start_list = list(zip(start_lons,start_lats))
reach_end_list = list(zip(end_lons,end_lats))

figsize(6,6)
plot(l2.lon[::10],l2.lat[::10],'o',alpha=0.1,color='aqua')
for reach in reaches:
    plot(reach.lon,reach.lat,'.',alpha=0.4,label='Reach %d'%i)
scatter(start_lons,start_lats,s=300,c='r',marker='+')
scatter(end_lons,end_lats,s=300,c='g',marker='x')
legend(loc='best')
title('Reaches vs Data')
if interactive:
    plugins.connect(gcf(),plugins.MousePosition(fmt='.3f'))

edited_reaches = reaches.split_by_coordinates(reach_start_list,reach_end_list)

for reach in edited_reaches:
    print(reach.metadata)

figsize(6,6)
for i,reach in enumerate(edited_reaches):
    plot(reach.lon,reach.lat,'.',alpha=0.4,label='Reach %d'%i)
scatter(start_lons,start_lats,s=300,c='r',marker='+')
scatter(end_lons,end_lats,s=300,c='g',marker='x')
legend(loc='best')
title('Edited Reaches')
if interactive:
    plugins.connect(gcf(),plugins.MousePosition(fmt='.3f'));

start_s = 25.e3
ds = 10.e3
end_s = start_s + 4*ds
edited_reaches = reaches.split_by_reach_length(ds,start_s=start_s,end_s=end_s)

for reach in edited_reaches:
    print(reach.metadata)

figsize(6,6)
for i,reach in enumerate(edited_reaches):
    plot(reach.lon,reach.lat,'.',alpha=1,label='Reach %d'%i)
legend(loc='best')
title('Edited Reaches')
if interactive:
    plugins.connect(gcf(),plugins.MousePosition(fmt='.3f'));

node_output_name = 'edited_nodes'
reach_output_name = 'edited_reaches'


get_ipython().system('rm -rf $node_output_name')
get_ipython().system('rm -rf $reach_output_name')

node_output_variables = ['width']
reach_output_variables = edited_reaches[0].metadata.keys()
reach_writer = RiverReachWriter(edited_reaches,node_output_variables,reach_output_variables)

reach_writer.write_nodes_ogr(node_output_name)

reach_writer.write_reaches_ogr(reach_output_name)

reach_writer.write_nodes_ogr(node_output_name+'.kml',driver='KML')
reach_writer.write_reaches_ogr(reach_output_name+'.kml',driver='KML')
get_ipython().system('ls *.kml')

width_db_file = 'edited_width_db'
river_df, reach_df = reach_writer.write_width_db(width_db_file,output_format='h5')
get_ipython().system('ls *.h5')

print(reach_df.head())

print(river_df.head())

print(river_df.tail())




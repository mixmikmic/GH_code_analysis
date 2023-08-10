import os
import sys
import subprocess

# create GRASS GIS runtime environment
gisbase = subprocess.check_output(["grass", "--config", "path"]).strip()
os.environ['GISBASE'] = gisbase
sys.path.append(os.path.join(gisbase, "etc", "python"))

import grass.script as gs
import grass.script.setup as gsetup

# set GRASS GIS session data
rcfile = gsetup.init(gisbase, "/home/jovyan/grassdata", "nc_spm_08_grass7", "user1")

gs.message('Current GRASS GIS 7 environment:')
print gs.gisenv()

print 'Available raster maps:'
for rast in gs.list_strings(type='raster'):
    print rast

print 'Available vector maps:'
for vect in gs.list_strings(type='vector'):
    print vect

# end GRASS GIS session
os.remove(rcfile)


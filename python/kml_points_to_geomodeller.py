# basic imports
import matplotlib.pyplot as plt
import numpy as np
import sys

from importlib import reload

sys.path.append('..')
from googlepicks import kml_to_plane
reload(kml_to_plane)

filename = "../data/dip4.kml"
k1 = kml_to_plane.KmlPoints(filename = filename, debug=True)

k1.add_geotiff("../data/dome_sub_sub_utm.tif")

k1.determine_z_values()

k1.fit_plane_to_all_sets()

k1.point_sets

get_ipython().run_line_magic('matplotlib', 'inline')
k1.stereonet()
# plt.savefig("stereonet_all.pdf")
plt.show()

k1.export_for_geomodeller(filename = 'jebel_madar_ori.csv', formname='Natih', data_type = 'ori')

# take all files in planes directory:
planes_dir = "/Users/flow/sciebo/Karten_und_Profile/planes"
import os

for plane_file in os.listdir(planes_dir):
    filename = os.path.join(planes_dir, plane_file)
    k1 = kml_to_plane.KmlPoints(filename = filename, debug=False)
    k1.add_geotiff("../data/dome_sub_sub_utm.tif")
    k1.determine_z_values()
    k1.latlong_to_utm()
    # extract formation name and construct new filename
    basename = os.path.splitext(plane_file)[0]
    k1.export_for_geomodeller(filename = basename + '_top.csv', formname=basename, data_type = 'planes')




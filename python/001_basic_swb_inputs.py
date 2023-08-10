import os
import numpy as np
import pandas as pd
import gdal

get_ipython().magic('run ../../COMMON/plot_and_table_functions.py')

# lookup table path and filenames
lu_lookup_fname  = '../std_input/Landuse_lookup_maui.txt'

# grid files
lu_gridfile_fname              = '../input/LU2010_w_2_season_sugarcane__simulation_1__50m.asc'
soils_gridfile_fname           = '../input/maui_HYDROLOGIC_SOILS_GROUP__50m.asc'
soil_storage_gridfile_fname    = '../input/maui_SOIL_MOISTURE_STORAGE__50m.asc'

control_file     = '../maui_swb2.ctl'

# open table files
lu_lookup     = pd.read_table( lu_lookup_fname )

# read in grid values - using 'read_raster' function taken from Andy Leaf's
# GISio.py package: https://github.com/aleaf/GIS_utils
lu_data, ly_gt, lu_proj, lu_xy                = read_raster( lu_gridfile_fname )
soils_data, soils_gt, soils_proj, soils_xy    = read_raster( soils_gridfile_fname )
ss_data, ss_gt, ss_proj, ss_xy                = read_raster( soil_storage_gridfile_fname )

make_plot( x=soils_xy[0], y=soils_xy[1], var=soils_data, discrete=True )

get_ipython().magic('run plot_and_table_functions.py')
lu_cmap = discrete_irreg_cmap(discrete_vals=np.unique( lu_data.flatten()), base_cmap='nipy_spectral')
make_plot( x=lu_xy[0], y=lu_xy[1], var=lu_data, discrete=True, cmap=lu_cmap )

make_plot( x=ss_xy[0], y=ss_xy[1], var=ss_data, title='Maximum Soil Storage',
         barlabel='Soil Storage, in inches')

## 


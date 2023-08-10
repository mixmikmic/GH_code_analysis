import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import gdal
import netCDF4 as nc

get_ipython().magic('run plot_and_table_functions.py')

def create_file_reference( component_name ):
    '''
    This is a simple convenience function that will form a path and filename to a
    given water budget component
    '''
    # specify the prefix, path to SWB2 output, timeframe, and resolution
    output_path = '../output/'
    prefix      = 'central_sands_'
    start_year  = '2012'
    end_year    = '2013'
    ncol        = '400'
    nrow        = '346'
    return( output_path + prefix + component_name + '_' + start_year + '_' 
          + end_year + '__' + nrow + '_by_' + ncol + '.nc' )

fname_lu = '../output/Landuse_land_cover__as_read_into_SWB.asc'
dataset_lu = gdal.Open( fname_lu )
if dataset_lu is None:
    print('Could not open landuse grid')
    quit
lu = dataset_lu.ReadAsArray()
lu = ma.masked_where( lu <= 0, lu )
lu_vals=np.unique( lu.flatten() )

# Define the pathname to the SWB landuse lookup table
fname_lu_table = '../std_input/Landuse_lookup_CDL.txt'    
lu_table = pd.read_table( fname_lu_table )    

net_infil           = nc.Dataset( create_file_reference( 'net_infiltration' ) )
irrigation          = nc.Dataset( create_file_reference( 'irrigation' ) )
rainfall            = nc.Dataset( create_file_reference( 'rainfall') )
runoff              = nc.Dataset( create_file_reference( 'runoff' ) )
runon               = nc.Dataset( create_file_reference( 'runon' ) )
runoff_outside      = nc.Dataset( create_file_reference( 'runoff_outside' ) )
snowmelt            = nc.Dataset( create_file_reference( 'snowmelt') )
actual_et           = nc.Dataset( create_file_reference( 'actual_et') )
soil_moisture       = nc.Dataset( create_file_reference( 'soil_storage') )
delta_soil_moisture = nc.Dataset( create_file_reference( 'delta_soil_storage') )
interception        = nc.Dataset( create_file_reference( 'interception') )

rejected_net_infiltration  = nc.Dataset( create_file_reference( 'rejected_net_infiltration') )

net_infil_vals_nc  = net_infil.variables[ 'net_infiltration' ]
irrigation_vals_nc = irrigation.variables[ 'irrigation' ]
rainfall_vals_nc   = rainfall.variables[ 'rainfall' ]
runoff_vals_nc     = runoff.variables[ 'runoff' ]
runon_vals_nc      = runon.variables[ 'runon' ]
runoff_outside_vals_nc     = runoff_outside.variables[ 'runoff_outside' ]
snowmelt_vals_nc   = snowmelt.variables[ 'snowmelt' ]
actual_et_vals_nc  = actual_et.variables[ 'actual_et' ]
soil_moist_vals_nc = soil_moisture.variables[ 'soil_storage' ]
delta_soil_moist_vals_nc = delta_soil_moisture.variables[ 'delta_soil_storage' ]
interception_vals_nc = interception.variables[ 'interception' ]

rejected_net_infil_vals_nc = rejected_net_infiltration.variables[ 'rejected_net_infiltration' ]

x_vals_nc          = net_infil.variables[ 'x' ]
y_vals_nc          = net_infil.variables[ 'y' ]

# create a numpy masked array from the netcdf variable values
net_infil_vals    = ma.masked_where( np.isnan( net_infil_vals_nc ), net_infil_vals_nc ) 
irrigation_vals   = ma.masked_where( np.isnan( irrigation_vals_nc ), irrigation_vals_nc ) 
rainfall_vals     = ma.masked_where( np.isnan( rainfall_vals_nc ), rainfall_vals_nc ) 
runoff_vals       = ma.masked_where( np.isnan( runoff_vals_nc ), runoff_vals_nc ) 
runon_vals        = ma.masked_where( np.isnan( runon_vals_nc ), runon_vals_nc ) 
snowmelt_vals     = ma.masked_where( np.isnan( snowmelt_vals_nc ), snowmelt_vals_nc ) 
actual_et_vals    = ma.masked_where( np.isnan( actual_et_vals_nc ), actual_et_vals_nc ) 
interception_vals = ma.masked_where( np.isnan( interception_vals_nc ), interception_vals_nc ) 
runoff_outside_vals    = ma.masked_where( np.isnan( runoff_outside_vals_nc ), runoff_outside_vals_nc )
delta_soil_moist_vals  = ma.masked_where( np.isnan( delta_soil_moist_vals_nc ), delta_soil_moist_vals_nc )
soil_moist_vals   = ma.masked_where( np.isnan( soil_moist_vals_nc ), soil_moist_vals_nc ) 

rejected_net_infil_vals   = ma.masked_where( np.isnan( rejected_net_infil_vals_nc ), rejected_net_infil_vals_nc ) 

x_vals = np.array( x_vals_nc )
y_vals = np.array( y_vals_nc )

mass_balance = rainfall_vals + snowmelt_vals + irrigation_vals + runon_vals + delta_soil_moist_vals - interception_vals - actual_et_vals - runoff_vals - net_infil_vals - rejected_net_infil_vals

make_plot( x=x_vals, y=y_vals, var=mass_balance[9,:,:], minz=-0.1, maxz=0.1)

np.where( mass_balance[71,:,:]>0.02 )

daynum = 6
np.unique(lu)
lu_descriptions=lu_table['Description']
lu_lookup_values=lu_table['LU_Code']
make_comparison_table( x=actual_et_vals[daynum,:,:], y=mass_balance[daynum,:,:], factor=lu, 
                       description=lu_descriptions,
                       lookup_vals=lu_lookup_values,
                       xlab='Mean actual ET', ylab='Mean mass_balance',
                       calc_difference=False )

# add up the grids for all 731 days in the simulation; divide by the number of years in the
# simulation to obtain the net infiltration sums on a mean annual basis
net_infil_sum          = net_infil_vals.sum(axis=0) / 2.0
irrigation_sum         = irrigation_vals.sum(axis=0) / 2.0
rainfall_sum           = rainfall_vals.sum(axis=0) / 2.0
runoff_sum             = runoff_vals.sum(axis=0) / 2.0
runoff_outside_sum     = runoff_outside_vals.sum(axis=0) / 2.0
runon_sum              = runon_vals.sum(axis=0) / 2.0
snowmelt_sum           = snowmelt_vals.sum(axis=0) / 2.0
actual_et_sum          = actual_et_vals.sum(axis=0) / 2.0
rejected_net_infil_sum = rejected_net_infil_vals.sum(axis=0) / 2.0 

make_plot( x=x_vals, y=y_vals, var=rainfall_sum, maxz=26., title="Rainfall", barlabel="inches per year" )

make_plot( x=x_vals, y=y_vals, var=snowmelt_sum, title="Snowmelt", barlabel="inches per year" )

make_plot( x=x_vals, y=y_vals, var=runon_sum, maxz=50., title="Runon", barlabel="inches per year" )

make_plot( x=x_vals, y=y_vals, var=runoff_sum, maxz=15., title="Runoff", barlabel="inches per year" )

make_plot( x=x_vals, y=y_vals, var=actual_et_sum, title="Actual ET", barlabel="inches per year" )

make_plot( x=x_vals, y=y_vals, title='Irrigation', barlabel='inches per year', var=irrigation_sum, maxz=15. )

make_plot( x=x_vals, y=y_vals, title='Net Infiltration', barlabel='inches per year', var=net_infil_sum, maxz=18. )

make_plot( x=x_vals, y=y_vals, title='Rejected Net Infiltration', barlabel='inches per year', var=rejected_net_infil_sum, maxz=18. )

make_plot( x=x_vals, y=y_vals, title='Runoff Outside', barlabel='inches per year', var=runoff_outside_sum, maxz=15. )




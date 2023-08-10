import os, sys, tarfile, wget

def download_if_absent( dataset_name ):
    "Function that downloads and decompress a chosen dataset"
    if os.path.exists( dataset_name ) is False:
        tar_name = "%s.tar.gz" %dataset_name
        url = "https://github.com/openPMD/openPMD-example-datasets/raw/draft/%s" %tar_name
        wget.download( url, tar_name )
        with tarfile.open( tar_name ) as tar_file:
            tar_file.extractall()
        os.remove( tar_name )

download_if_absent( 'example-2d' )

get_ipython().magic('matplotlib inline')

from opmd_viewer import OpenPMDTimeSeries

ts = OpenPMDTimeSeries('./example-2d/hdf5/')

# One example
rho, info_rho = ts.get_field( iteration=300, field='rho' )
# Another example
Ex, info_Ex = ts.get_field( t=100.e-15,  field='E', coord='x' )

Ex, info_Ex = ts.get_field( t=100.e-15,  field='E', coord='x', plot=True )

get_ipython().magic('pinfo ts.get_field')

get_ipython().magic('pinfo info_rho')

# Look for the available species
print(ts.avail_species)

# One example: extracting several quantities
x, y, z = ts.get_particle( var_list=['x','y', 'z'], iteration=300, species='Hydrogen1+') 
# Another example: extracting 1 quantity 
# (notice the comma after z, so that z is a 1darray, not a list)
z, = ts.get_particle( var_list=['z'], t=150.e-15, species='electrons')

ux = ts.get_particle( var_list=['ux'], iteration=400, species='electrons', plot=True )

z, uz = ts.get_particle( var_list=['z','uz'], iteration=400, species='electrons', plot=True, vmax=3e12 )

get_ipython().magic('pinfo ts.get_particle')


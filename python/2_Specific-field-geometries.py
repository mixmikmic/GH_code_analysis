import os, sys, tarfile, wget

def download_if_absent( dataset_name ):
    "Function that downloads and decompress a chosen dataset"
    if os.path.exists( dataset_name ) is False:
        tar_name = "%s.tar.gz" %dataset_name
        url = "https://github.com/openPMD/openPMD-example-datasets/raw/draft/%s" %tar_name
        wget.download(url, tar_name)
        with tarfile.open( tar_name ) as tar_file:
            tar_file.extractall()
        os.remove( tar_name )

download_if_absent( 'example-3d' )
download_if_absent( 'example-thetaMode' )

get_ipython().magic('matplotlib inline')

from opmd_viewer import OpenPMDTimeSeries

ts_3d = OpenPMDTimeSeries('./example-3d/hdf5/', check_all_files=False )
ts_circ = OpenPMDTimeSeries('./example-thetaMode/hdf5/', check_all_files=False )

# Slice across y (i.e. in a plane parallel to x-z)
rho1, info_rho1 = ts_3d.get_field( field='rho', iteration=500, vmin=-5e6,
                                             slicing_dir='y', plot=True )

# Slice across z (i.e. in a plane parallel to x-y)
rho2, info_rho2 = ts_3d.get_field( field='rho', iteration=500, vmin=-5e6,
                                             slicing_dir='z', plot=True )

# Slice across z, very close to zmin.
rho2, info_rho2 = ts_3d.get_field( field='rho', iteration=500, vmin=-5e6,
                                slicing_dir='z', slicing=-0.9, plot=True )

Ey, info_Ey = ts_circ.get_field( field='E', coord='y', iteration=500, m=0, 
                              plot=True, theta=0.5)

Ey, info_Ey = ts_circ.get_field( field='E', coord='y', iteration=500, m='all', 
                              plot=True, theta=0.5)

Er, info_Er = ts_circ.get_field( field='E', coord='r', iteration=500, m=0, 
                              plot=True, theta=0.5)


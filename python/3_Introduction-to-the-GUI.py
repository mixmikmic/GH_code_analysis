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

download_if_absent( 'example-2d' )
download_if_absent( 'example-3d' )
download_if_absent( 'example-thetaMode' )

get_ipython().magic('matplotlib inline')

from opmd_viewer import OpenPMDTimeSeries

ts_2d = OpenPMDTimeSeries('./example-2d/hdf5/')

ts_2d.slider()

ts_3d = OpenPMDTimeSeries('./example-3d/hdf5/')
ts_3d.slider()

ts_circ = OpenPMDTimeSeries('./example-thetaMode/hdf5/')
ts_circ.slider()


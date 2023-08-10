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

get_ipython().magic('matplotlib inline')

from opmd_viewer.addons import LpaDiagnostics

ts_2d = LpaDiagnostics('./example-2d/hdf5/')

ts_2d.slider()

get_ipython().magic('pinfo ts_2d.get_mean_gamma')

ts_2d.get_mean_gamma(iteration=300, species='electrons', select={'uz' : [0.05, None]})

ts_2d.get_charge(iteration=300, species='electrons')

ts_2d.get_divergence(iteration=300, species='electrons')

ts_2d.get_emittance(iteration=300, species='electrons')

ts_2d.get_current(iteration=300, species='electrons', plot=True);

ts_2d.get_laser_envelope(iteration=300, pol='y');

ts_2d.get_spectrum(iteration=300, pol='y', plot=True);

ts_2d.get_spectrogram(iteration=300, pol='y', plot=True, cmap='YlGnBu_r');

ts_2d.get_main_frequency(iteration=300, pol='y')

ts_2d.get_a0(iteration=300, pol='y')

ts_2d.get_laser_waist(iteration=300, pol='y')

ts_2d.get_ctau(iteration=300, pol='y')


import sys
import os 
import ipywidgets as widgets
from IPython.display import display
import getpass

# Import main BASTet convert tool 
sys.path.insert(0,"/project/projectdirs/openmsi/omsi_processing_status/bastet")
from omsi.tools.convertToOMSI import main as convert_omsi

# Jupyter sets up logging so that log message are not displayed in the notebook, so we need to 
# reload the logging module in order to be able to have log messages appear in the notebook
import logging
reload(logging)
from omsi.shared.log import log_helper
log_helper.setup_logging()
log_helper.set_log_level('DEBUG')

# username = getpass.getuser()
username = 'jiangao'
omsi_original_data = os.path.join("/project/projectdirs/openmsi/original_data", username)

username = getpass.getuser()
omsi_private_data = os.path.join("/project/projectdirs/openmsi/omsi_data_private", username)

get_ipython().magic('system ls -t $omsi_original_data | head -10')

base_filename = '03072017_JG_RootImaging_Brachy_DirectStamp'
in_filename = (os.path.join(omsi_original_data, base_filename))
out_filename = (os.path.join(omsi_private_data, base_filename) + '.h5')

settings = ['convertToOMSI.py', 
            '--no-xdmf',
            '--user', username,
            '--regions', 'merge',
            '--db-server', 'https://openmsi.nersc.gov/openmsi/',
            '--compression',
            '--thumbnail',
            '--auto-chunking',
            '--error-handling', 'terminate-and-cleanup',
            '--add-to-db',
            '--no-fpl',
            '--no-fpg',
            '--no-ticnorm',
            '--no-nmf',
            in_filename,
            out_filename
           ]

print os.path.isfile(out_filename)
print out_filename

convert_omsi(argv=settings)

import sys
import os 

# Import main BASTet convert tool 
sys.path.insert(0,"/project/projectdirs/openmsi/omsi_processing_status/bastet")

from omsi.dataformat.omsi_file import *
# from omsi.analysis.multivariate_stats.omsi_cx import *
# from omsi.analysis.msi_filtering.ticNormalization import *
from omsi.analysis.findpeaks.omsi_findpeaks_global import *

out_filename = '/project/projectdirs/openmsi/omsi_data_private/bpb/20170227MdR_5800_Maldi_sec_met_pathway_screening.h5'

f = omsi_file(out_filename , 'a' )
e = f.get_experiment(0) #All data is organized as experiments and this just gets the first one
d= e.get_msidata(0) #This gets the omsi_file_msidata object for the first raw dataset 




out_filename = '/project/projectdirs/openmsi/omsi_data_private/bpb/20170227MdR_5800_Maldi_sec_met_pathway_screening.h5'

f = omsi_file(out_filename , 'a' )
e = f.get_experiment(0) #All data is organized as experiments and this just gets the first one
d= e.get_msidata(0) #This gets the omsi_file_msidata object for the first raw dataset 
d.shape

peakCube = omsi_findpeaks_global(name_key='Global peak finding on raw data')
# myIons = np.asarray([844.304, 868.288, 852.303, 806.331, 828.312, 804.314])
peakCube.execute( msidata=d, mzdata=d.mz, integration_width=0.1, peakheight=20, smoothwidth=3, slwindow=100)
e.create_analysis(peakCube)
f.close_file()

from omsi.analysis.multivariate_stats.omsi_nmf import *

# nmf_data.get_help_string()

e.create_analysis(nmf_data)
f.close_file()

f = omsi_file(out_filename , 'a' )
e = f.get_experiment(0) #All data is organized as experiments and this just gets the first one
peak_cube = e.get_analysis(0)
nmf_data = omsi_nmf(name_key='NMF')
nmf_data.execute(msidata = peak_cube['peak_cube'],numComponents=20)
e.create_analysis(nmf_data)
f.close_file()

e.create_analysis(nmf_data)
f.close_file()


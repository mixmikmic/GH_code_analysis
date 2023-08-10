get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
from os.path import exists
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

get_ipython().run_line_magic('pylab', 'inline')

from os.path import join
from GWDLR import GWDLR

data_dir = join(data_dir,'GWDLR')
root_name = 'n35w125_wth'

gwdlr = GWDLR(root_name,data_dir)
gwdlr.__dict__

min_width = 25.
gwdlr.to_mask(min_width,overwrite=True,thin=True)

figsize(8,8)
imshow(gwdlr.data,cmap=cm.gray)

mask_file = join(data_dir,root_name+'_center_line_%d.tif'%min_width)
gwdlr.to_gdal(mask_file,gdal_format='GTiff')


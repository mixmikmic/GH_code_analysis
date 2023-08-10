get_ipython().system('mkdir dataset_test')
get_ipython().system('mkdir dataset_test/data')
get_ipython().system('cp -r $CALDB dataset_test/')
get_ipython().system('ls -ltr dataset_test/')

get_ipython().system('cp sim_events_00000* dataset_test/data')

get_ipython().run_cell_magic('bash', '', 'cp dataset_test/caldb/data/cta/prod2/bcf/North_0.5h/irf_file.fits.gz dataset_test/caldb/data/cta/prod2/bcf/North_0.5h/irf_file.fits \ncp dataset_test/caldb/data/cta/prod2/bcf/North_5h/irf_file.fits.gz dataset_test/caldb/data/cta/prod2/bcf/North_5h/irf_file.fits \ncp dataset_test/caldb/data/cta/prod2/bcf/North_50h/irf_file.fits.gz dataset_test/caldb/data/cta/prod2/bcf/North_50h/irf_file.fits \ncp dataset_test/caldb/data/cta/prod2/bcf/South_0.5h/irf_file.fits.gz dataset_test/caldb/data/cta/prod2/bcf/South_0.5h/irf_file.fits \ncp dataset_test/caldb/data/cta/prod2/bcf/South_5h/irf_file.fits.gz dataset_test/caldb/data/cta/prod2/bcf/South_5h/irf_file.fits \ncp dataset_test/caldb/data/cta/prod2/bcf/South_50h/irf_file.fits.gz dataset_test/caldb/data/cta/prod2/bcf/South_50h/irf_file.fits ')

get_ipython().system('ls -l dataset_test/caldb/data/cta/prod2/bcf/*')

get_ipython().system('ls -l dataset_test/data')

import gammalib
import ctools
import cscripts 

import numpy as np
import glob

from astropy.io import fits 
from astropy import units as u
from astropy.coordinates import SkyCoord

get_ipython().system('cp -r $GAMMAPY_EXTRA/test_datasets/cta_1dc .')

obs_ind = 'cta_1dc/hdu-index.fits.gz'
hdulist_obs = fits.open(obs_ind)
paste_header_hdu = hdulist_obs[1].header

paste_header_hdu['DATASET'] = "Dataset for the Sexten School"
#create columns

# count the number of sim_events_000*'fits files
obs_number = len(glob.glob('dataset_test/data/sim_events_*'))

data_1_hdu = np.array([[i]*6 for i in range(1,obs_number+1)]).reshape(6*obs_number)

temp_2 = ['events','gti','aeff','edisp','psf','bkg']
data_2_hdu = np.array([temp_2 for i in range(1,obs_number+1)]).reshape(6*obs_number)

temp_3 = ['events','gti','aeff_2d','edisp_2d','psf_3gauss','bkg_3d']
data_3_hdu = np.array([temp_3 for i in range(1,obs_number+1)]).reshape(6*obs_number)

temp_4 = ['data']*2
temp_4.extend(["caldb/data/cta/prod2/bcf/South_50h"]*4)
data_4_hdu = np.array([temp_4 for i in range(1,obs_number+1)]).reshape(6*obs_number)


temp_5 = ['irf_file.fits']*4
data_5_hdu = np.array([["sim_events_"+str(i).rjust(6,'0') +".fits"] *2 + temp_5 for i in range(1,obs_number+1)]).reshape(6*obs_number)

temp_6 = ['EVENTS','GTI','EFFECTIVE AREA','ENERGY DISPERSION','POINT SPREAD FUNCTION','BACKGROUND']
data_6_hdu = np.array([temp_6 for i in range(1,obs_number+1)]).reshape(6*obs_number)


col1_hdu = fits.Column(name='OBS_ID', format='K', array=data_1_hdu)
col2_hdu = fits.Column(name='HDU_TYPE', format='6A', array=data_2_hdu)
col3_hdu = fits.Column(name='HDU_CLASS', format='10A', array=data_3_hdu)
col4_hdu = fits.Column(name='FILE_DIR', format='39A', array=data_4_hdu)
col5_hdu = fits.Column(name='FILE_NAME', format='26A', array=data_5_hdu)
col6_hdu = fits.Column(name='HDU_NAME', format='21A', array=data_6_hdu)

cols_hdu = fits.ColDefs([col1_hdu, col2_hdu, col3_hdu, col4_hdu, col5_hdu, col6_hdu])
tbhdu_hdu = fits.BinTableHDU.from_columns(cols_hdu, header = paste_header_hdu)

prihdr_hdu = fits.Header()
prihdu_hdu = fits.PrimaryHDU(header=prihdr_hdu)

thdulist_hdu = fits.HDUList([prihdu_hdu, tbhdu_hdu])

fitsname_hdu = "dataset_test/hdu-index.fits.gz"

thdulist_hdu.writeto(fitsname_hdu , overwrite=True)

# Print additional informations regarding the fits file
# Just for testing purposes

obs_ind = 'cta_1dc/obs-index.fits.gz'
hdulist_obs = fits.open(obs_ind)
paste_header_hdu = hdulist_obs[1].header

print hdulist_obs[1].columns.names

data, hdr = fits.getdata('cta_1dc/obs-index.fits.gz', 1, header=True)
test_row = data[0:1]
out_file = "dataset_test/obs-index.fits.gz"

#fits.writeto(out_file, test_row, hdr, overwrite = True)

#get number of file fits from simulations
obs_number = len(glob.glob('dataset_test/data/sim_events_*'))

# these are constant over all the data
test_row['ZEN_PNT']  = 0.0
test_row['ALT_PNT']  = 90.0
test_row['AZ_PNT']   = 0.0
out_list = []

#fits.writeto(out_file, data[1:1],  hdr, overwrite = True)
for i in range(1,obs_number+1):
    
    obs_name = 'data/sim_events_'+str(i).rjust(6,'0')+'.fits'
    obs_ind = 'dataset_test/'+obs_name
    hdulist_obs = fits.open(obs_ind)
    paste_header_hdu = hdulist_obs[1].header

    test_row['OBS_ID']           = i
    test_row['RA_PNT']           = paste_header_hdu["RA_PNT"]
    test_row['DEC_PNT']          = paste_header_hdu["DEC_PNT"]
    c_icrs = SkyCoord(ra=float(test_row['RA_PNT'])*u.degree, dec=float(test_row['DEC_PNT'])*u.degree, frame='icrs')
    
    test_row['GLON_PNT']         = c_icrs.galactic.l.value
    test_row['GLAT_PNT']         = c_icrs.galactic.b.value
    test_row['ONTIME']           = paste_header_hdu["ONTIME"]
    test_row['LIVETIME']         = paste_header_hdu["LIVETIME"]
    test_row['DEADC']            = paste_header_hdu["DEADC"]
    test_row['TSTART']           = paste_header_hdu["TSTART"]
    test_row['TSTOP']            = paste_header_hdu["TSTOP"]
    test_row['DATE_OBS']         = paste_header_hdu["DATE_OBS"]
    test_row['TIME_OBS']         = paste_header_hdu["TIME_OBS"]
    test_row['DATE_END']         = paste_header_hdu["DATE_END"]
    test_row['TIME_END']         = paste_header_hdu["TIME_END"]
    test_row['EVENTS_FILENAME']  = obs_name
    test_row['EVENT_COUNT']      = hdulist_obs[1].header["NAXIS2"]
    test_row['EVENT_TIME_MIN']   = test_row['TSTART']
    test_row['EVENT_TIME_MAX']   = test_row['TSTOP']
    test_row['EVENT_ENERGY_MIN'] = np.min(hdulist_obs[1].data['ENERGY'])
    test_row['EVENT_ENERGY_MIN'] = np.max(hdulist_obs[1].data['ENERGY'])
    out_list.append(test_row.tolist())

test_numpy = np.array(out_list, dtype = test_row.dtype)
fits.writeto(out_file, test_numpy, hdr, overwrite = True)

get_ipython().system('ls dataset_test/')

get_ipython().system('ls dataset_test/data')




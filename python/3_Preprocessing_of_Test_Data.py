import numpy as np
import h5py
import os
import vos

datadir='/home/ubuntu/starnet_data/'  # or "/path/to/my/starnet/directory"

'''
def starnet_download_file(filename):
    vclient = vos.Client()
    vclient.copy('vos:starnet/public/'+filename, datadir+filename)
    print(filename+' downloaded')

starnet_download_file('apStar_combined_main.h5')
'''

#filename = datadir + 'apStar_combined_main.h5'
filename = datadir + 'apStar_combined_main_dr13.h5'
F = h5py.File(filename,'r')
print('Dataset keys in file: \n')
F.keys()

ap_id = F['IDs'][:,0]
combined_snr = F['stacked_snr'][:]
starflag = F['star_flag']
aspcapflag = F['aspcap_flag']
teff = F['TEFF'][:]
logg = F['LOGG'][:]
fe_h = F['FE_H'][:]
vscatter = F['VSCATTER']

print('Obtainined data for '+str(len(list(set(list(ap_id)))))+' stars.')

mean = np.array([np.mean(teff[teff!=-9999.]),np.mean(logg[logg!=-9999.]),np.mean(fe_h[fe_h!=-9999.])])
std = np.array([np.std(teff[teff!=-9999.]),np.std(logg[logg!=-9999.]),np.std(fe_h[fe_h!=-9999.])])
mean_and_std = np.row_stack((mean,std))
np.save(datadir+'mean_and_std', mean_and_std)

print('mean_and_std.npy saved')

teff_min = 4000.
teff_max = 5500.
vscatter_max = 1.
fe_h_min = -3.

indices, cols = np.where((aspcapflag[:]==0.)&(starflag[:]==0.)&(vscatter[:]<vscatter_max)&(fe_h[:]>fe_h_min)&(teff[:]>teff_min)&(teff[:]<teff_max)&(logg[:]!=-9999.).reshape(len(ap_id),1))

ap_id = ap_id[indices]
teff = teff[indices]
logg = logg[indices]
fe_h = fe_h[indices]
combined_snr = combined_snr[indices]

print(str(len(list(set(list(ap_id)))))+' stars remain.')

high_snr_test_ap_ids = np.load(datadir + 'high_snr_test_apids.npy')

indices_high_snr = [i for i, item in enumerate(high_snr_test_ap_ids) if item in ap_id]

high_snr_ap_id = ap_id[indices_high_snr]
high_snr_teff = teff[indices_high_snr]
high_snr_logg = logg[indices_high_snr]
high_snr_fe_h = fe_h[indices_high_snr]
high_snr_combined_snr = combined_snr[indices_high_snr]

indices_high_snr = indices[indices_high_snr] # These indices will be used to index through the spectra

print('High S/N test set includes '+str(len(high_snr_ap_id))+' combined spectra')

# Define edges of detectors
blue_chip_begin = 322
blue_chip_end = 3242
green_chip_begin = 3648
green_chip_end = 6048   
red_chip_begin = 6412
red_chip_end = 8306 

savename = 'high_snr_test_data.h5'
# if path already exist, you must remove it first using os.remove(path) 
#os.remove(datadir + savename)
dt = h5py.special_dtype(vlen=bytes)

with h5py.File(datadir + savename, "a") as f:
    
    # Create datasets for your high S/N data file 
    spectra_ds = f.create_dataset('spectrum', (1,7214), maxshape=(None,7214), dtype="f", chunks=(1,7214))
    error_spectra_ds = f.create_dataset('error_spectrum', (1,7214), maxshape=(None,7214), dtype="f", chunks=(1,7214))
    teff_ds = f.create_dataset('TEFF', high_snr_teff.shape, dtype="f")
    logg_ds = f.create_dataset('LOGG', high_snr_logg.shape, dtype="f")
    fe_h_ds = f.create_dataset('FE_H', high_snr_fe_h.shape, dtype="f")
    combined_snr_ds = f.create_dataset('combined_snr', high_snr_combined_snr.shape, dtype="f")
    ap_id_ds = f.create_dataset('Ap_ID', high_snr_ap_id.shape, dtype="S18")
    
    # Save data to data file
    teff_ds[:] = high_snr_teff
    logg_ds[:] = high_snr_logg
    fe_h_ds[:] = high_snr_fe_h
    combined_snr_ds[:] = high_snr_combined_snr
    ap_id_ds[:] = high_snr_ap_id.tolist()
        
    # Collect spectra
    first_entry=True
    
    for i in indices_high_snr:

        spectrum = F['spectrum'][i:i+1]
        err_spectrum = F['error_spectrum'][i:i+1]


        # NORMALIZE SPECTRUM
        # Separate spectra into chips
        blue_sp = spectrum[0:1,blue_chip_begin:blue_chip_end]
        green_sp = spectrum[0:1,green_chip_begin:green_chip_end]
        red_sp = spectrum[0:1,red_chip_begin:red_chip_end]
        
        blue_sp_med = np.median(blue_sp, axis=1)
        green_sp_med = np.median(green_sp, axis=1)
        red_sp_med = np.median(red_sp, axis=1)

        #Normalize spectra by chips
        blue_sp = (blue_sp.T/blue_sp_med).T
        green_sp = (green_sp.T/green_sp_med).T
        red_sp = (red_sp.T/red_sp_med).T

        # Recombine spectra
        spectrum = np.column_stack((blue_sp,green_sp,red_sp))
        
        # Normalize error spectrum using the same method
        # Separate error spectra into chips

        blue_sp = err_spectrum[0:1,blue_chip_begin:blue_chip_end]
        green_sp = err_spectrum[0:1,green_chip_begin:green_chip_end]
        red_sp = err_spectrum[0:1,red_chip_begin:red_chip_end]

        # Normalize error spectra by chips
        blue_sp = (blue_sp.T/blue_sp_med).T
        green_sp = (green_sp.T/green_sp_med).T
        red_sp = (red_sp.T/red_sp_med).T

        # Recombine error spectra
        err_spectrum = np.column_stack((blue_sp,green_sp,red_sp))
        
        if first_entry:
            spectra_ds[0] = spectrum
            error_spectra_ds[0] = err_spectrum
            first_entry=False
        else:
            spectra_ds.resize(spectra_ds.shape[0]+1, axis=0)
            error_spectra_ds.resize(error_spectra_ds.shape[0]+1, axis=0)

            spectra_ds[-1] = spectrum
            error_spectra_ds[-1] = err_spectrum

print(savename+' has been saved as the high S/N test set to be used in 5_test_Model.ipynb')  

snr_max = 200.

indices_low_snr, cols = np.where((combined_snr[:]<snr_max).reshape(len(ap_id),1))

low_snr_ap_id = ap_id[indices_low_snr]
low_snr_teff = teff[indices_low_snr]
low_snr_logg = logg[indices_low_snr]
low_snr_fe_h = fe_h[indices_low_snr]
low_snr_combined_snr = combined_snr[indices_low_snr]

indices_low_snr = indices[indices_low_snr] # These indices will be used to index through the spectra

print('Low S/N test set includes '+str(len(low_snr_ap_id))+' combined spectra')

savename = 'low_snr_test_data.h5'
# if path already exist, you must remove it first using os.remove(path) 
#os.remove(datadir + savename)
dt = h5py.special_dtype(vlen=bytes)

with h5py.File(datadir + savename, "a") as f:
    
    # Create datasets for your high S/N data file 
    spectra_ds = f.create_dataset('spectrum', (1,7214), maxshape=(None,7214), dtype="f", chunks=(1,7214))
    error_spectra_ds = f.create_dataset('error_spectrum', (1,7214), maxshape=(None,7214), dtype="f", chunks=(1,7214))
    teff_ds = f.create_dataset('TEFF', low_snr_teff.shape, dtype="f")
    logg_ds = f.create_dataset('LOGG', low_snr_logg.shape, dtype="f")
    fe_h_ds = f.create_dataset('FE_H', low_snr_fe_h.shape, dtype="f")
    combined_snr_ds = f.create_dataset('combined_snr', low_snr_combined_snr.shape, dtype="f")
    ap_id_ds = f.create_dataset('Ap_ID', low_snr_ap_id.shape, dtype="S18")
    
    # Save data to data file
    teff_ds[:] = low_snr_teff
    logg_ds[:] = low_snr_logg
    fe_h_ds[:] = low_snr_fe_h
    combined_snr_ds[:] = low_snr_combined_snr
    ap_id_ds[:] = low_snr_ap_id.tolist()
        
    # Collect spectra
    first_entry=True
    
    for i in indices_low_snr:

        spectrum = F['spectrum'][i:i+1]
        err_spectrum = F['error_spectrum'][i:i+1]


        # NORMALIZE SPECTRUM
        # Separate spectra into chips
        blue_sp = spectrum[0:1,blue_chip_begin:blue_chip_end]
        green_sp = spectrum[0:1,green_chip_begin:green_chip_end]
        red_sp = spectrum[0:1,red_chip_begin:red_chip_end]
        
        blue_sp_med = np.median(blue_sp, axis=1)
        green_sp_med = np.median(green_sp, axis=1)
        red_sp_med = np.median(red_sp, axis=1)

        #Normalize spectra by chips
        blue_sp = (blue_sp.T/blue_sp_med).T
        green_sp = (green_sp.T/green_sp_med).T
        red_sp = (red_sp.T/red_sp_med).T

        # Recombine spectra
        spectrum = np.column_stack((blue_sp,green_sp,red_sp))
        
        # Normalize error spectrum using the same method
        # Separate error spectra into chips

        blue_sp = err_spectrum[0:1,blue_chip_begin:blue_chip_end]
        green_sp = err_spectrum[0:1,green_chip_begin:green_chip_end]
        red_sp = err_spectrum[0:1,red_chip_begin:red_chip_end]

        # Normalize error spectra by chips
        blue_sp = (blue_sp.T/blue_sp_med).T
        green_sp = (green_sp.T/green_sp_med).T
        red_sp = (red_sp.T/red_sp_med).T

        # Recombine error spectra
        err_spectrum = np.column_stack((blue_sp,green_sp,red_sp))
        
        
        if first_entry:
            spectra_ds[0] = spectrum
            error_spectra_ds[0] = err_spectrum
            
            first_entry=False
        else:
            spectra_ds.resize(spectra_ds.shape[0]+1, axis=0)
            error_spectra_ds.resize(error_spectra_ds.shape[0]+1, axis=0)

            spectra_ds[-1] = spectrum
            error_spectra_ds[-1] = err_spectrum

print(savename+' has been saved as the low S/N test set to be used in 5_Test_Model.ipynb')  




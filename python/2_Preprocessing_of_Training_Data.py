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
    
starnet_download_file('apStar_visits_main.h5')
'''

filename = datadir + 'apStar_visits_main_dr13.h5'

F = h5py.File(filename,'r')
print('Dataset keys in file: \n')
F.keys()

ap_id = F['IDs'][:,0]
combined_snr = F['stacked_snr']
starflag = F['star_flag']
aspcapflag = F['aspcap_flag']
teff = F['TEFF'][:]
logg = F['LOGG'][:]
fe_h = F['FE_H'][:]
vscatter = F['VSCATTER']

print('Obtained data for '+str(len(ap_id))+' individual visits from '+str(len(list(set(list(ap_id)))))+' stars.')

snr_min = 200.
teff_min = 4000.
teff_max = 5500.
vscatter_max = 1.
fe_h_min = -3.

indices, cols = np.where((aspcapflag[:]==0.)&(starflag[:]==0.)&(combined_snr[:]>=snr_min)&(vscatter[:]<vscatter_max)&(fe_h[:]>fe_h_min)&(teff[:]>teff_min)&(teff[:]<teff_max)&(logg[:]!=-9999.).reshape(len(ap_id),1))

print(str(len(indices))+' visits remain.')

num_ref = 44784 # number of reference spectra

indices_ref = indices[0:num_ref]
np.random.shuffle(indices_ref)

ap_id_ref = ap_id[indices_ref]
teff = teff[indices_ref]
logg = logg[indices_ref]
fe_h = fe_h[indices_ref]

print('Reference set includes '+str(len(ap_id_ref))+' individual visits from '+str(len(set(ap_id_ref)))+' stars.')

indices_test = indices[num_ref:]

ap_id_test = ap_id[indices_test]
ap_id_test = list(set(ap_id_test)-set(ap_id_ref))
np.save(datadir + 'high_snr_test_apids', ap_id_test)
print(str(len(ap_id_test))+' stars to be processed for the High S/N test set.')

# Define edges of detectors
blue_chip_begin = 322
blue_chip_end = 3242
green_chip_begin = 3648
green_chip_end = 6048   
red_chip_begin = 6412
red_chip_end = 8306 

savename = 'training_data.h5'
# if path already exist, you must remove it first using os.remove(path) 
#os.remove(datadir + savename)
dt = h5py.special_dtype(vlen=bytes)

with h5py.File(datadir + savename, "a") as f:
    
    # Create datasets for your reference data file 
    spectra_ds = f.create_dataset('spectrum', (1,7214), maxshape=(None,7214), dtype="f", chunks=(1,7214))
    teff_ds = f.create_dataset('TEFF', teff.shape, dtype="f")
    logg_ds = f.create_dataset('LOGG', logg.shape, dtype="f")
    fe_h_ds = f.create_dataset('FE_H', fe_h.shape, dtype="f")
    ap_id_ds = f.create_dataset('Ap_ID', ap_id_ref.shape, dtype="S18")
    
    teff_ds[:] = teff
    logg_ds[:] = logg
    fe_h_ds[:] = fe_h
    ap_id_ds[:] = ap_id_ref.tolist()
        
    first_entry=True
    
    for i in indices_ref:

        spectrum = F['spectrum'][i:i+1]


        # NORMALIZE SPECTRUM
        # Separate spectra into chips
        blue_sp = spectrum[0:1,blue_chip_begin:blue_chip_end]
        green_sp = spectrum[0:1,green_chip_begin:green_chip_end]
        red_sp = spectrum[0:1,red_chip_begin:red_chip_end]

        # Normalize spectra by chips

        blue_sp = (blue_sp.T/np.median(blue_sp, axis=1)).T
        green_sp = (green_sp.T/np.median(green_sp, axis=1)).T
        red_sp = (red_sp.T/np.median(red_sp, axis=1)).T 

        # Recombine spectra

        spectrum = np.column_stack((blue_sp,green_sp,red_sp))
        if first_entry:
            spectra_ds[0] = spectrum
            first_entry=False
        else:
            spectra_ds.resize(spectra_ds.shape[0]+1, axis=0)

            spectra_ds[-1] = spectrum

print(savename+' has been saved as the reference set to be used in 4_Train_Model.ipynb')  




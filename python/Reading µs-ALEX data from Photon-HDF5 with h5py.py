from __future__ import division, print_function  # only needed on py2
get_ipython().magic('matplotlib inline')
import numpy as np
import h5py
import matplotlib.pyplot as plt

def print_children(group):
    """Print all the sub-groups in `group` and leaf-nodes children of `group`.

    Parameters:
        data_file (h5py HDF5 file object): the data file to print
    """
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            content = '(Group)'
        else:
            content = value[()]
        print(name)
        print('    Content:     %s' % content)
        print('    Description: %s\n' % value.attrs['TITLE'].decode())

filename = '../data/0023uLRpitc_NTP_20dT_0.5GndCl.hdf5'

h5file = h5py.File(filename)

print_children(h5file)

print_children(h5file['sample'])

photon_data = h5file['photon_data']

photon_data['measurement_specs']['measurement_type'][()].decode()

timestamps = photon_data['timestamps'][:]
timestamps_unit = photon_data['timestamps_specs']['timestamps_unit'][()]
detectors = photon_data['detectors'][:]

print('Number of photons: %d' % timestamps.size)
print('Timestamps unit:   %.2e seconds' % timestamps_unit)
print('Detectors:         %s' % np.unique(detectors))

h5file['setup']['excitation_wavelengths'][:]

donor_ch = photon_data['measurement_specs']['detectors_specs']['spectral_ch1'][()]
acceptor_ch = photon_data['measurement_specs']['detectors_specs']['spectral_ch2'][()]
print('Donor CH: %d     Acceptor CH: %d' % (donor_ch, acceptor_ch))

alex_period = photon_data['measurement_specs']['alex_period'][()]
donor_period = photon_data['measurement_specs']['alex_excitation_period1'][()]
offset = photon_data['measurement_specs']['alex_offset'][()]
acceptor_period = photon_data['measurement_specs']['alex_excitation_period2'][()]
print('ALEX period:     %d  \nOffset:         %4d      \nDonor period:    %s      \nAcceptor period: %s' %       (alex_period, offset, donor_period, acceptor_period))

timestamps_donor = timestamps[detectors == donor_ch]
timestamps_acceptor = timestamps[detectors == acceptor_ch]

fig, ax = plt.subplots()
ax.hist((timestamps_acceptor - offset) % alex_period, bins=100, alpha=0.8, color='red', label='donor')
ax.hist((timestamps_donor - offset) % alex_period, bins=100, alpha=0.8, color='green', label='acceptor')
ax.axvspan(donor_period[0], donor_period[1], alpha=0.3, color='green')
ax.axvspan(acceptor_period[0], acceptor_period[1], alpha=0.3, color='red')
ax.set_xlabel('(timestamps - offset) MOD alex_period')
ax.set_title('ALEX histogram')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False);

timestamps_mod = (timestamps - offset) % alex_period
donor_excitation = (timestamps_mod < donor_period[1])*(timestamps_mod > donor_period[0])
acceptor_excitation = (timestamps_mod < acceptor_period[1])*(timestamps_mod > acceptor_period[0])
timestamps_Dex = timestamps[donor_excitation]
timestamps_Aex = timestamps[acceptor_excitation]

fig, ax = plt.subplots()
ax.hist((timestamps_Dex - offset) % alex_period, bins=np.arange(0, alex_period, 40), alpha=0.8, color='green', label='D_ex')
ax.hist((timestamps_Aex - offset) % alex_period, bins=np.arange(0, alex_period, 40), alpha=0.8, color='red', label='A_ex')
ax.set_xlabel('(timestamps - offset) MOD alex_period')
ax.set_title('ALEX histogram (selected periods only)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False);

#plt.close('all')




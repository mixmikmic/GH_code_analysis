from __future__ import division, print_function  # only needed on py2
get_ipython().magic('matplotlib inline')
import numpy as np
import tables
import matplotlib.pyplot as plt

def print_children(group):
    """Print all the sub-groups in `group` and leaf-nodes children of `group`.

    Parameters:
        group (pytables group): the group to be printed.
    """
    for name, value in group._v_children.items():
        if isinstance(value, tables.Group):
            content = '(Group)'
        else:
            content = value.read()
        print(name)
        print('    Content:     %s' % content)
        print('    Description: %s\n' % value._v_title.decode())

filename = '../data/0023uLRpitc_NTP_20dT_0.5GndCl.hdf5'

h5file = tables.open_file(filename)

print_children(h5file.root)

print_children(h5file.root.sample)

photon_data = h5file.root.photon_data

photon_data.measurement_specs.measurement_type.read().decode()

timestamps = photon_data.timestamps.read()
timestamps_unit = photon_data.timestamps_specs.timestamps_unit.read()
detectors = photon_data.detectors.read()

print('Number of photons: %d' % timestamps.size)
print('Timestamps unit:   %.2e seconds' % timestamps_unit)
print('Detectors:         %s' % np.unique(detectors))

h5file.root.setup.excitation_wavelengths.read()

donor_ch = photon_data.measurement_specs.detectors_specs.spectral_ch1.read()
acceptor_ch = photon_data.measurement_specs.detectors_specs.spectral_ch2.read()
print('Donor CH: %d     Acceptor CH: %d' % (donor_ch, acceptor_ch))

alex_period = photon_data.measurement_specs.alex_period.read()
offset = photon_data.measurement_specs.alex_offset.read()
donor_period = photon_data.measurement_specs.alex_excitation_period1.read()
acceptor_period = photon_data.measurement_specs.alex_excitation_period2.read()
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




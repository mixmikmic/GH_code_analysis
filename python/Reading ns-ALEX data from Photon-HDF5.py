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

filename = '../data/Pre.hdf5'

h5file = tables.open_file(filename)

print_children(h5file.root)

print_children(h5file.root.sample)

photon_data = h5file.root.photon_data

photon_data.measurement_specs.measurement_type.read().decode()

timestamps = photon_data.timestamps.read()
timestamps_unit = photon_data.timestamps_specs.timestamps_unit.read()
detectors = photon_data.detectors.read()
nanotimes = photon_data.nanotimes.read()
tcspc_num_bins = photon_data.nanotimes_specs.tcspc_num_bins.read()
tcspc_unit = photon_data.nanotimes_specs.tcspc_unit.read()

print('Number of photons: %d' % timestamps.size)
print('Timestamps unit:   %.2e seconds' % timestamps_unit)
print('TCSPC unit:        %.2e seconds' % tcspc_unit)
print('TCSPC number of bins:    %d' % tcspc_num_bins)
print('Detectors:         %s' % np.unique(detectors))

h5file.root.setup.excitation_wavelengths.read()

donor_ch = photon_data.measurement_specs.detectors_specs.spectral_ch1.read()
acceptor_ch = photon_data.measurement_specs.detectors_specs.spectral_ch2.read()
print('Donor CH: %d     Acceptor CH: %d' % (donor_ch, acceptor_ch))

laser_rep_rate = photon_data.measurement_specs.laser_repetition_rate.read()
donor_period = photon_data.measurement_specs.alex_excitation_period1.read()
acceptor_period = photon_data.measurement_specs.alex_excitation_period2.read()
print('Laser repetion rate: %5.1f MHz \nDonor period:    %s      \nAcceptor period: %s' %       (laser_rep_rate*1e-6, donor_period, acceptor_period))

nanotimes_donor = nanotimes[detectors == donor_ch]
nanotimes_acceptor = nanotimes[detectors == acceptor_ch]

bins = np.arange(0, tcspc_num_bins + 1)
hist_d, _ = np.histogram(nanotimes_donor, bins=bins)
hist_a, _ = np.histogram(nanotimes_acceptor, bins=bins)

fig, ax = plt.subplots(figsize=(10, 4.5))
scale = tcspc_unit*1e9
ax.plot(bins[:-1]*scale, hist_d, color='green', label='donor')
ax.plot(bins[:-1]*scale, hist_a, color='red', label='acceptor')
ax.axvspan(donor_period[0]*scale, donor_period[1]*scale, alpha=0.3, color='green')
ax.axvspan(acceptor_period[0]*scale, acceptor_period[1]*scale, alpha=0.3, color='red')
ax.set_xlabel('TCSPC Nanotime (ns) ')
ax.set_title('TCSPC Histogram')
ax.set_yscale('log')
ax.set_ylim(10)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

#plt.close('all')




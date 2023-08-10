get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import astropy.units as u

hitran_raw_path = '/Users/bmmorris/Desktop/whoseline/hitran/custom/5a0c0d48.out.txt'
molecule_key_path = '/Users/bmmorris/Desktop/whoseline/hitran/key.txt'
simplified_output_path = '../data/hitran_simplified.txt'
delimiter = ' '
wavelength_column = 3
strength_column = 4
einstein_column = 5
molecular_index_column = 0

hitran = np.genfromtxt(hitran_raw_path, autostrip=True)

molecule_key = ascii.read(molecule_key_path, 
                          delimiter='\t', format='no_header')

molecules = {i: j for i, j in zip(molecule_key['col1'], molecule_key['col2'])}

molecule_labels = list(map(lambda x: molecules[x], hitran[:, molecular_index_column]))

def wavenumber_to_wavelength(wn):
    """
    Convert wavenumber in cm^-1 to wavelength in Angstrom
    """
    return (u.cm/wn).to(u.Angstrom)

wavelength = wavenumber_to_wavelength(hitran[:, wavelength_column]).value
strengths = hitran[:, einstein_column]

with open(simplified_output_path, 'w') as w:
    for i, j, k in zip(wavelength, np.array(molecule_labels), strengths):
        w.write(delimiter.join(map(str, [i, j, k])) + '\n')




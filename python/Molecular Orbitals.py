from chemview import MolecularViewer

from chemlab.io import remotefile
from chemlab.core import guess_bonds
# pull the file
df = remotefile('https://raw.githubusercontent.com/cclib/cclib/master/data/GAMESS/basicGAMESS-US2012/Trp_polar_tdhf.out', 'gamess')

mo_coefficients = df.read('mocoeffs')
gbasis = df.read('gbasis')
molecule = df.read('molecule')
molecule.bonds = guess_bonds(molecule.r_array, molecule.type_array, threshold=0.05)

from chemlab.qc import molecular_orbital

f = molecular_orbital(molecule.r_array, mo_coefficients[0][-1], gbasis)

mv = MolecularViewer(molecule.r_array, {'atom_types': molecule.type_array,
                                        'bonds': molecule.bonds})
mv.wireframe()
mv.add_isosurface(f, isolevel=0.3, color=0xff0000)
mv.add_isosurface(f, isolevel=-0.3, color=0x0000ff)

mv


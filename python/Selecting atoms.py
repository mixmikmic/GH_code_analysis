from __future__ import print_function
import warnings
warnings.filterwarnings("ignore") 
get_ipython().magic('pylab inline')

from magres.atoms import MagresAtoms

atoms = MagresAtoms.load_magres('../samples/ethanol-all.magres')

atoms

len(atoms)

for atom in atoms:
    print(atom)

atoms.C1

atoms.C1.position

atoms.species("H")


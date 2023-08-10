from __future__ import print_function
import warnings
warnings.filterwarnings("ignore") 
get_ipython().magic('pylab inline')

from magres.atoms import MagresAtoms

atoms = MagresAtoms.load_magres('../samples/glycine-relaxed.magres')

print("We have", len(atoms), "atoms")

for atom in atoms.species('H'):
    print(atom, atom.ms.iso, atom.ms.aniso)

print("atom\tiso\taniso")
for atom in atoms.species('H'):
    print("{}\t{:.2f}\t{:.2f}".format(atom, atom.ms.iso, atom.ms.aniso))

atoms.species('H').set_reference(40.0)

print("atom\tcs\taniso")
for atom in atoms.species('H'):
    print("{}\t{:.2f}\t{:.2f}".format(atom, atom.ms.cs, atom.ms.aniso))


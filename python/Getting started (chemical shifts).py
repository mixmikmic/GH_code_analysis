from __future__ import print_function

from magres.atoms import MagresAtoms
from numpy import mean

atoms = MagresAtoms.load_magres('../samples/ethanol-all.magres')

print("We have", len(atoms), "atoms")

for atom in atoms:
    print(atom, atom.ms.iso)

atoms.species('H').set_reference(10.0)

for atom in atoms.species('H'):
    print(atom, atom.ms.cs)

for atom in atoms.within(atoms.C1, 2.0):
    print(atom, atom.ms.iso, atom.ms.aniso)

atoms.species('H').ms.iso

print("C1 H mean ms iso = ", mean(atoms.C1.bonded.species('H').ms.iso))
print("C2 H mean ms iso = ", mean(atoms.C2.bonded.species('H').ms.iso))
print("O1 H mean ms iso = ", mean(atoms.O1.bonded.species('H').ms.iso))

print("Magnetic shielding tensor, sigma")
print(atoms.C1.ms.sigma)

print()
print("Eigenvectors of sigma")
print(atoms.C1.ms.evecs)

print()
print("Eigenvalues of sigma")
print(atoms.C1.ms.evals)


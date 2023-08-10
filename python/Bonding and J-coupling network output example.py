from __future__ import print_function

from magres.atoms import MagresAtoms, MagresAtomsView
ethanol_atoms = MagresAtoms.load_magres('../samples/ethanol-all.magres')

castep_file = open('../samples/ethanol.castep').read()

ethanol_atoms.calculate_bonds()

(ethanol_atoms.C1 + ethanol_atoms.C1.bonded)

alanine_atoms = MagresAtoms.load_magres('../samples/alanine-jc-all.magres')

#castep_file = open('../samples/alanine.castep').read()

alanine_atoms.calculate_bonds()

alanine_atoms

alanine_atoms.N1.bonded + alanine_atoms.N2.bonded + alanine_atoms.O1.bonded + alanine_atoms.O3.bonded + alanine_atoms.N1 + alanine_atoms.N2 + alanine_atoms.O1 + alanine_atoms.O3

def get_molecule(atom):
    atoms = set()

    def _get_molecule(atom1):
        for atom2 in atom1.bonded:
            if atom2 not in atoms:
                atoms.add(atom2)
                _get_molecule(atom2)
                
    _get_molecule(atom)
    
    return MagresAtomsView(list(atoms), atom.bonded.lattice)

get_molecule(alanine_atoms.N1)


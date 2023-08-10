from utilities.wikiMagics import JinjaMagics
from IPython.core.magic import register_cell_magic
ip = get_ipython()
ip.register_magics(JinjaMagics)

# load model into memory
import os
import re
from rmgpy.chemkin import loadChemkinFile, getSpeciesIdentifier
from rmgpy.molecule.molecule import Molecule
from rmgpy.molecule.draw import MoleculeDrawer

mech = 'pdd_scratch_add18'
path = os.path.abspath('./')
chemkinPath= path + '/data/' + mech + '/chem.inp'
dictionaryPath = path + '/data/' + mech + '/species_dictionary.txt'
species_list, reactions_list = loadChemkinFile(chemkinPath, dictionaryPath, readComments = False)

#generate species images
mech_path = path + '/data/' + mech
speciesPath = mech_path + '/species/'
if not os.path.isdir(speciesPath):
    os.makedirs(speciesPath)

species = species_list[:]
re_index_search = re.compile(r'\((\d+)\)$').search
for spec in species:
    match = re_index_search(spec.label)
    if match:
        spec.index = int(match.group(0)[1:-1])
        spec.label = spec.label[0:match.start()]
    # Draw molecules if necessary
    fstr = os.path.join(mech_path, 'species', '{0}.png'.format(spec))
    if not os.path.exists(fstr):
        try:
            MoleculeDrawer().draw(spec.molecule[0], 'png', fstr)
        except IndexError:
            raise OutputError("{0} species could not be drawn!".format(getSpeciesIdentifier(spec)))

species_target = 'C=CC=C'

# search the target species in model
mol_tgt = Molecule().fromSMILES(species_target)

for spc in species_list:
    if spc.isIsomorphic(mol_tgt):
        print '{} is found in model with spc name {}'.format(mol_tgt, getSpeciesIdentifier(spc))
        break

target_spc_index = 30
from IPython.display import display as disp
for spc in species_list:
    if spc.index == target_spc_index:
        print "The spcies with index {} is found {} and displayed below:".format(target_spc_index, getSpeciesIdentifier(spc))
        disp(spc)
        print "And its SMILES is {}.".format(spc.molecule[0].toSMILES())
        break
else:
    print "Cound not find species with index {}.".format(target_spc_index)

target_spc_label = 'C8H8'
for spc in species_list:
    if spc.label == target_spc_label:
        print "The spcies with label {} is found and displayed below:".format(target_spc_label)
        disp(spc)
        print "And its SMILES is {}.".format(spc.molecule[0].toSMILES())
        break
else:
    print "Cound not find species with label {}.".format(target_spc_label)

rxns_spc = []
for rxn in reactions_list:
    for spec_rxn in (rxn.reactants + rxn.products):
        if spec_rxn.index == spc.index:
            for spec2_rxn in (rxn.reactants + rxn.products):
                if spec2_rxn.index in range(5,17):
                    rxns_spc.append(rxn)

disp_num = min(100, len(rxns_spc))

get_ipython().run_cell_magic('jinja', 'html', 'rxn_with_spc.html')


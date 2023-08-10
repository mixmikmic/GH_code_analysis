import os
from rmgpy.chemkin import loadChemkinFile, getSpeciesIdentifier
from pychemkin.chemkin import getMoleFraction
from utilities.wikiMagics import JinjaMagics
from IPython.core.magic import register_cell_magic
ip = get_ipython()
ip.register_magics(JinjaMagics)

# load chemkin model
mech = 'pdd_scratch_add11'
path = os.path.abspath('../')
chemkinPath= path + '/data/' + mech + '/chem.inp'
dictionaryPath = path + '/data/' + mech + '/species_dictionary.txt'
species_list, reactions_list = loadChemkinFile(chemkinPath, dictionaryPath, readComments = False)

import re
from rmgpy.molecule.draw import MoleculeDrawer

mech_path = path + '/data/' + mech
if not os.path.isdir(os.path.join(mech_path,'species')):
    os.makedirs(os.path.join(mech_path,'species'))
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

# load ckcsv info
species_id_list = []
for spc in species_list:
    species_id_list.append(getSpeciesIdentifier(spc))

ckcsvPath= path + '/data/' + mech + '/CKSoln.ckcsv'
tempData, spcData = getMoleFraction(ckcsvPath, species_id_list)

# get top radicals and saturates list
spc_sorted = sorted(spcData.items(), key=lambda tup: -tup[1][0][-1]) # tup:(spc,[array,..])

top_rad_list = []
top_sat_list = []
for tup in spc_sorted:
    spc_id = tup[0]
    final_mf = tup[1][0][-1]
    for spc in species_list:
        if getSpeciesIdentifier(spc) == spc_id:
            if spc.molecule[0].isRadical():
                top_rad_list.append((spc, final_mf))
            else:
                top_sat_list.append((spc, final_mf))

disp_num = 10

get_ipython().run_cell_magic('jinja', 'html', 'top_spc_with_mf.html')


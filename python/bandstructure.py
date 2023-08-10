from datetime import datetime, timedelta
from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()

from aiida.work.run import run
from aiida.orm.data.base import Str
from aiida.tools.dbimporters import DbImporterFactory
from aiida.workflows.user.workchain.quantumespresso.pw.bands import PwBandsWorkChain

# Loading the COD importer so we can directly import structure from COD id's
CodImporter = DbImporterFactory('cod')
importer = CodImporter()

# Make sure here to define the correct codename that corresponds to the pw.x code installed on your machine of choice
codename = Str('pw-5.1@localhost')

# Al COD ID(9008460)
structure_Al = importer.query(id='9008460')[0].get_aiida_structure()
structure_Al.get_formula()

# GaAs COD ID(9008845)
structure_GaAs = importer.query(id='9008845')[0].get_aiida_structure()
structure_GaAs.get_formula()

# CaF2 COD ID(1000043)
structure_CaF2 = importer.query(id='1000043')[0].get_aiida_structure()
structure_CaF2.get_formula()

# h-BN COD ID(9008997)
structure_hBN = importer.query(id='9008997')[0].get_aiida_structure()
structure_hBN.get_formula()

# This will take approximately 3 minutes on the tutorial AWS
results_Al = run(
    PwBandsWorkChain,
    codename=codename,
    structure=structure_Al,
)

fermi_energy = results_Al['scf_parameters'].dict.fermi_energy
results_Al['bandstructure'].show_mpl(y_origin=fermi_energy, plot_zero_axis=True)

print """Final crystal symmetry: {spacegroup_international} (number {spacegroup_number})
Extended Bravais lattice symbol: {bravais_lattice_extended}
The system has inversion symmetry: {has_inversion_symmetry}""".format(
    **results_Al['final_seekpath_parameters'].get_dict())

# This will take approximately 5 minutes on the tutorial AWS
results_GaAs = run(
    PwBandsWorkChain,
    codename=codename,
    structure=structure_GaAs,
)

fermi_energy = results_GaAs['scf_parameters'].dict.fermi_energy
results_GaAs['bandstructure'].show_mpl(y_origin=fermi_energy, plot_zero_axis=True)

print """Final crystal symmetry: {spacegroup_international} (number {spacegroup_number})
Extended Bravais lattice symbol: {bravais_lattice_extended}
The system has inversion symmetry: {has_inversion_symmetry}""".format(
    **results_GaAs['final_seekpath_parameters'].get_dict())

# This will take approximately 9 minutes on the tutorial AWS
results_CaF2 = run(
    PwBandsWorkChain,
    codename=codename,
    structure=structure_CaF2,
)

fermi_energy = results_CaF2['scf_parameters'].dict.fermi_energy
results_CaF2['bandstructure'].show_mpl(y_origin=fermi_energy, plot_zero_axis=True)

print """Final crystal symmetry: {spacegroup_international} (number {spacegroup_number})
Extended Bravais lattice symbol: {bravais_lattice_extended}
The system has inversion symmetry: {has_inversion_symmetry}""".format(
    **results_CaF2['final_seekpath_parameters'].get_dict())

# This will take approximately 28 minutes on the tutorial AWS
results_hBN = run(
    PwBandsWorkChain,
    codename=codename,
    structure=structure_hBN,
)

fermi_energy = results_hBN['scf_parameters'].dict.fermi_energy
results_hBN['bandstructure'].show_mpl(y_origin=fermi_energy, plot_zero_axis=True)

print """Final crystal symmetry: {spacegroup_international} (number {spacegroup_number})
Extended Bravais lattice symbol: {bravais_lattice_extended}
The system has inversion symmetry: {has_inversion_symmetry}""".format(
    **results_hBN['final_seekpath_parameters'].get_dict())




from pymatgen.ext.matproj import MPRester

mpr = MPRester()  # If this gives you an error, please do mpr = MPRester("your API key") instead.

# Here, we use the high-level interface to the Materials Project (MPRester) to get all entries from 
# the Materials Project with formula Li4GeS4.

entries = mpr.get_entries("Na3PS4", inc_structure=True)
print(len(entries))  # There should be two entries.

# Usually, we want the minimum energy structure
min_entry = min(entries, key=lambda e: e.energy_per_atom)
Na3PS4 = min_entry.structure

# Let us now automatically assign oxidation states to the elements.
Na3PS4.add_oxidation_state_by_guess()

print(Na3PS4)
print("Spacegroup of lowest energy Na3PS4 is %s (%d)" % Na3PS4.get_space_group_info())

Na3PS4Cl = Na3PS4.copy()
Na3PS4Cl.make_supercell([2, 2, 2])

# Remove one Na.
del Na3PS4Cl[0]

# Replace one S2- with Cl-. Here, we create a disordered site with the right ratios,
# which will then be used to find symmetrically distinct orderings subsequently.
Na3PS4Cl["S2-"] = {"S2-": 63/64, "Cl-": 1/64}
print("Overall charge of %s is %f" % (Na3PS4Cl.formula, Na3PS4Cl.charge))

# Generates a crystallographic information format file that can be viewed in most 
# crystal visualization software.
Na3PS4Cl.to(filename="Na3PS4Cl.cif")  

from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

trans = OrderDisorderedStructureTransformation()
ordered_Na3PS4Cl = trans.apply_transformation(Na3PS4Cl, return_ranked_list=100)
print("# of structures generated = %s" % len(ordered_Na3PS4Cl))

from pymatgen.analysis.structure_matcher import StructureMatcher

matcher = StructureMatcher()
groups = matcher.group_structures([d["structure"] for d in ordered_Na3PS4Cl])
distinct_Na3PS4Cl = [g[0] for g in groups]
print(len(distinct_Na3PS4Cl))

from pymatgen.io.vasp.sets import MPRelaxSet, batch_write_input

# batch_write_input(distinct_Na3PS4Cl, include_cif=True, output_dir="Cl_doped_Na3PS4")

import json
from pymatgen.entries.computed_entries import ComputedEntry

with open("entry_Na47P16S63Cl.json") as f:
    entry = ComputedEntry.from_dict(json.load(f))
print(entry.entry_id)
print(entry)

from pymatgen.analysis.phase_diagram import PhaseDiagram

npscl_entries = mpr.get_entries_in_chemsys(["Na", "P", "S", "Cl"])

from pymatgen.entries.compatibility import MaterialsProjectCompatibility

compat = MaterialsProjectCompatibility()
processed_entries = compat.process_entries(npscl_entries + [entry])

pd = PhaseDiagram(processed_entries)
ehull = pd.get_e_above_hull(entry)
print("Defect formation energy = %.3f eV and Ehull = %.3f eV/atom" % (ehull * entry.composition.num_atoms, ehull))

get_ipython().run_line_magic('matplotlib', 'inline')
from pymatgen.analysis.phase_diagram import PDPlotter
plotter = PDPlotter(pd)
plt = plotter.plot_element_profile("Na", entry.composition)




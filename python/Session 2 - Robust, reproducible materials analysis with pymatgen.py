from pymatgen import Structure

Li4GeS4 = Structure.from_file("ICSD_95649.cif")
print(Li4GeS4)

from pymatgen.ext.matproj import MPRester

mpr = MPRester()  # If this gives you an error, please do mpr = MPRester("your API key") instead.

# Here, we use the high-level interface to the Materials Project (MPRester) to get all entries from 
# the Materials Project with formula Li4GeS4.

entries = mpr.get_entries("Li4GeS4", inc_structure=True)
print(len(entries))
Li4GeS4 = entries[0].structure
print(Li4GeS4)

# Another example of getting more than one structure from Materials 
# Project.

for e in mpr.get_entries("CsCl", inc_structure=True):
    print(e.structure)
    print(e.structure.get_space_group_info())

Li4SnS4 = Li4GeS4.copy()
Li4SnS4["Ge"] = "Sn"
print(Li4SnS4)

# Generates a crystallographic information format file that can be viewed in most 
# crystal visualization software.
Li4SnS4.to(filename="Li4SnS4.cif")  

from pymatgen.io.vasp.sets import MPRelaxSet

input_set = MPRelaxSet(Li4SnS4)
print(input_set.incar)
print(input_set.kpoints)
# print(input_set.poscar)
# print(input_set.potcar)

# Do not run the line below. This is to show that you can write all the input files to a directory.
# input_set.write_input("Li4SnS4")

from pymatgen.io.vasp import Vasprun

vasprun = Vasprun("Li4SnS4/vasprun.xml")
print(vasprun.final_structure)
print("Final energy = %.3f" % vasprun.final_energy)
Li4SnS4_entry = vasprun.get_computed_entry()

# Get all the entries in the chemical system
Li_Sn_S_entries = mpr.get_entries_in_chemsys(["Li", "Sn", "S"])

from pymatgen.entries.compatibility import MaterialsProjectCompatibility

compat = MaterialsProjectCompatibility()
all_entries = compat.process_entries(Li_Sn_S_entries + [Li4SnS4_entry])

# This line allows the plots to show up within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter

pd = PhaseDiagram(all_entries)
plotter = PDPlotter(pd)
plotter.show()

cdos = vasprun.complete_dos

from pymatgen.electronic_structure.plotter import DosPlotter

dos_plotter = DosPlotter(stack=True)
dos_plotter.add_dos_dict(cdos.get_spd_dos())
plt = dos_plotter.get_plot()
plt.xlim((-15, 10))  # Limit the range of energies for easier visualization.
text = plt.title("Band gap of Li4SnS4 = %.2f eV" % cdos.get_gap(), fontsize=24)


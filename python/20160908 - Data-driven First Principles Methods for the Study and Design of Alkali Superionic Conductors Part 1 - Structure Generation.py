from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
from pymatgen.io.vasp.sets import batch_write_input, MPRelaxSet

structure = Structure.from_file("aimd_data/EntryWithCollCode418490.cif")
print(structure)

# loop over all sites in the structure
for i, site in enumerate(structure):
    # change the occupancy of Li+ disordered sites to 0.5
    if not site.is_ordered:
        structure[i] = {"Li+": 0.5}
print("The composition after adjustments is %s." % structure.composition.reduced_formula)

analyzer = SpacegroupAnalyzer(structure)
prim_cell = analyzer.find_primitive()
print(prim_cell)

enum = EnumerateStructureTransformation()
enumerated = enum.apply_transformation(prim_cell, 100)  # return no more than 100 structures
structures = [d["structure"] for d in enumerated]  
print("%d structures returned." % len(structures))

batch_write_input(structures, vasp_input_set=MPRelaxSet, output_dir="Li6PS5Cl_orderings")


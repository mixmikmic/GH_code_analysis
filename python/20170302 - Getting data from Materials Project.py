from pymatgen import MPRester, Composition
import re
import pprint

# Make sure that you have the Materials API key. Put the key in the call to
# MPRester if needed, e.g, MPRester("MY_API_KEY")
mpr = MPRester()

comp = Composition("Fe2O3")
anon_formula = comp.anonymized_formula
# We need to convert the formula to the dict form used in the database.
anon_formula = {m.group(1): int(m.group(2)) 
                for m in re.finditer(r"([A-Z]+)(\d+)", anon_formula)}

data = mpr.query({"anonymous_formula": anon_formula}, 
                 properties=["task_id", "pretty_formula", "structure"])
print(len(data))  #Should show ~600 data.

# data now contains a list of dict. This shows you what each dict has.
# Note that the mp id is named "task_id" in the database itself.
pprint.pprint(data[0])  

bs = mpr.get_bandstructure_by_material_id("mp-20470")

from pymatgen.electronic_structure.plotter import BSPlotter
get_ipython().magic('matplotlib inline')

plotter = BSPlotter(bs)
plotter.show()

elastic_data = mpr.query({"elasticity": {"$exists": True}}, 
                         properties=["task_id", "pretty_formula", "elasticity"])

print(len(elastic_data))
pprint.pprint(elastic_data[0])

from pymatgen.analysis.structure_matcher import StructureMatcher

m = StructureMatcher() # You can customize tolerances etc., but the defaults usually work fine.

s1 = data[0]["structure"]
print(s1)
s2 = s1.copy()
s2.apply_strain(0.1)
print(s2)

print(m.fit(s1, s2)) 

matches = []
for d in data:
    if m.fit_anonymous(d["structure"], s1):
        matches.append(d)

# The above fitting took a few seconds. We have 32 similar structures.
print(len(matches))

# Let's see a few of the matches.
pprint.pprint(matches[0])
pprint.pprint(matches[1])
pprint.pprint(matches[2])




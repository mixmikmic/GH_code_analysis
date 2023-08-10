h = open('../data/1ehz.pdb')
pdb_content = h.read()
h.close()

from pyrna.parsers import parse_pdb
tertiary_structures = parse_pdb(pdb_content)
print tertiary_structures[0].residues[1]

from pyrna.db import PDB
from pyrna.parsers import parse_pdb
pdb = PDB()
tertiary_structures = parse_pdb(pdb.get_entry('1EHZ'))

from pyrna.computations import Rnaview
secondary_structures = []
for ts in tertiary_structures:
    secondary_structure, tertiary_structure = Rnaview().annotate(ts)
    secondary_structures.append(secondary_structure)

ss = secondary_structures[0]
ts = tertiary_structures[0]
ss.find_junctions()
ss.find_stem_loops() #if no junctions are available in the secondary structure, the function find_junctions() is called automatically

from pyrna.parsers import to_pdb
from pyrna.features import Location

for index, stem_loop in enumerate(ss.stem_loops):
    with open('/Users/fjossinet/tmp/stem_loop_%i.pdb'%index, 'w') as f: 
        f.write(to_pdb(ts, location = Location(nested_lists = stem_loop['location'])))

from IPython.display import Image
Image("../data/1EHZ_stem_loops.png")

for index, junction in enumerate(ss.junctions):
    with open('/Users/fjossinet/tmp/junction_%i.pdb'%index, 'w') as f:
        location = Location(nested_lists = junction['location'])
        f.write(to_pdb(ts, location = location))

from IPython.display import Image
Image("../data/1EHZ_junctions.png")

ts = parse_pdb(pdb.get_entry('3Q1Q'))[0]
ss, ts = Rnaview().annotate(ts)
ss.find_junctions()

for index, helix in enumerate(ss.helices):
    with open('/Users/fjossinet/tmp/helix_%i.pdb'%index, 'w') as f:
        location = Location(nested_lists = helix['location'])
        f.write(to_pdb(ts, location = location))

for index, junction in enumerate(ss.junctions):
    with open('/Users/fjossinet/tmp/junction_%i.pdb'%index, 'w') as f:
        location = Location(nested_lists = junction['location'])
        f.write(to_pdb(ts, location = location))

from IPython.display import Image
Image("../data/3Q1Q.png")


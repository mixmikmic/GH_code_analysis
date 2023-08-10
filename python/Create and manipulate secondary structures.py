import json #to have a better output for dict describing pyrna.features.RNA objects (interactions, helices, single-strands,..)

from pyrna.parsers import parse_bn
rna = RNA(name = 'my_rna', sequence = 'GGGGGACAACCCC')
bn = '(({(.....))))'
base_pairs = parse_bn(bn)
print base_pairs

from pyrna.parsers import base_pairs_to_secondary_structure
ss = base_pairs_to_secondary_structure(rna, base_pairs)

for helix in ss.helices:
    print json.dumps(helix, indent = 2)
    print "\nAny non canonical base pairs in this helix?\n"
    for interaction in helix['interactions']:
        print json.dumps(interaction, indent = 2)    

for single_strand in ss.single_strands:
    print json.dumps(single_strand, indent = 2)

from pyrna.parsers import secondary_structure_to_base_pairs
print secondary_structure_to_base_pairs(ss)

rna = RNA(name = 'my_rna', sequence = 'GGGGGACGCAGTAACCCC')
bn = '(({(.(.....)..))))'
base_pairs = parse_bn(bn)
print base_pairs

ss = base_pairs_to_secondary_structure(rna, base_pairs)
for tertiary_interaction in ss.tertiary_interactions:
    print json.dumps(tertiary_interaction, indent = 2)

print secondary_structure_to_base_pairs(ss, keep_tertiaries=True)

print secondary_structure_to_base_pairs(ss)

from pyrna.parsers import parse_vienna

h = open('../data/ft3100_2D_with_bracket_notation.fasta')
vienna_content = h.read()
h.close()

print vienna_content

all_molecules, all_base_pairs = parse_vienna(vienna_content)

for base_pairs in all_base_pairs:
    print base_pairs

from pyrna.db import Rfam
rfam = Rfam(use_website = True)

gapped_rnas, organism_names_2_nse, consensus_2d = rfam.get_entry(rfam_id='RF00059')

for gapped_rna in gapped_rnas[:10]:
    print "sequence for %s: %s"%(gapped_rna.name, gapped_rna.sequence)

from pyrna.parsers import consensus2d_to_base_pairs, to_bn

for gapped_rna in gapped_rnas[:10]:
    rna = RNA(name = gapped_rna.name, sequence = gapped_rna.sequence.replace('-',''))
    print rna.name
    print rna.sequence
    print to_bn(consensus2d_to_base_pairs(gapped_rna, consensus_2d), len(rna))

from pyrna.computations import Rnafold
from pyrna.features import RNA
rna = RNA(name = 'my_rna', sequence = 'GGGGTAGGGACGGTAGGGGGACGCAGTGCAGTAACGTACCCGGTAGGGGGTAGGGGGACGCAGTAACCCCGGGGACGCAGTAACCCCACGCAGTAACCCC')
rnafold = Rnafold() #the algorithm is launched locally, using a Docker image
base_pairs = rnafold.fold(rna)
print base_pairs

from pyrna.computations import Rnaplot
rnaplot = Rnaplot()
plot = rnaplot.plot(secondary_structure = base_pairs, rna = rna)
print plot

from pyrna.db import PDB
from pyrna.parsers import parse_pdb
pdb = PDB()
tertiary_structures = parse_pdb(pdb.get_entry('1HR2'))

secondary_structures = []
for ts in tertiary_structures:
    #the function annotate() from Rnaview returns a pyrna.features.SecondaryStructure object and its 3D counterpart as a pyrna.features.TertiaryStructure object
    secondary_structure, tertiary_structure = Rnaview().annotate(ts)
    secondary_structures.append(secondary_structure)
    #the function secondary_structure_to_base_pairs() transform a pyrna.features.SecondaryStructure object into a list of base pairs stored in a pandas Dataframe
    print "Molecular chain %s"%secondary_structure.rna.name
    print secondary_structure_to_base_pairs(secondary_structure, keep_tertiaries = True)

base_pairs = secondary_structure_to_base_pairs(secondary_structures[0], keep_tertiaries = True)
print base_pairs[(base_pairs['edge1'] == '[') | (base_pairs['edge2'] == ']')]

junctions = []
for ss in secondary_structures:
    ss.find_junctions()
    junctions += ss.junctions

print junctions[0]

import re
for junction in junctions:
    if re.match('G[AUGC][AG]A',junction['description']):
        print "Apical loop with sequence %s"%junction['description']

for junction in junctions:
    if len(junction['location']) == 3:
        print "Three-way junctions with sequence %s"%junction['description']


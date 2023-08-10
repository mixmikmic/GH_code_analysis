from pyrna.features import DNA, RNA
rna = RNA(name = 'my_rna', sequence = 'AGGGGATTAACCCC')
print "%s: %s"%(rna.name, rna.sequence)
dna = DNA(name = 'my_dna', sequence = 'GGTTGGATTAACCCC')
print "%s: %s"%(dna.name, dna.sequence)

print "slice: %s"%rna[0:2]
print "length: %i"%len(rna)

print rna[3]

rna +'AAA'
print rna.sequence

rna-3
print rna.sequence

for index, residue in enumerate(rna):
    print "residue n%i: %s"%(index+1, residue)

h = open('../data/1ehz.pdb')
pdb_content = h.read()
h.close()

from pyrna.parsers import parse_pdb
tertiary_structures = parse_pdb(pdb_content)

for ts in tertiary_structures:
    print ts.rna.name
    print ts.rna.sequence
    print ts.rna.modified_residues

h = open('../data/telomerases.fasta')
fasta_content = h.read()
h.close()

from pyrna.parsers import parse_fasta
#the default type is RNA
for rna in parse_fasta(fasta_content):
    print "sequence of %s:"%rna.name
    print "%s\n"%rna.sequence

h = open('../data/ft3100_from_FANTOM3_project.fasta')
fasta_content = h.read()
h.close()

for dna in parse_fasta(fasta_content, 'DNA'):
    print "sequence as a DNA:"
    print "%s\n"%dna.sequence

for rna in parse_fasta(fasta_content):
    print "sequence as an RNA:"
    print rna.sequence

parse_fasta(fasta_content)[0]

from pyrna.db import PDB
pdb = PDB()
pdb_content = pdb.get_entry('1GID')

from pyrna.parsers import parse_pdb

for tertiary_structure in parse_pdb(pdb_content):
    print "molecular chain %s: %s"%(tertiary_structure.rna.name, tertiary_structure.rna.sequence)




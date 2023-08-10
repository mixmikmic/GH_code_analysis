import networkx as nx
import custom_funcs as cf
import pandas as pd
import matplotlib.pyplot as plt
import dendropy

from Levenshtein import distance
from collections import defaultdict, Counter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment
from itertools import product

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('matplotlib inline')

G = nx.read_gpickle('20150902_all_ird Final Graph.pkl')
G = cf.clean_host_species_names(G)
G = cf.impute_reassortant_status(G)
G = cf.impute_weights(G)

G.nodes(data=True)[1286]

# Get all of the host species with TOL and BOLD links
hosts_with_coi = pd.read_csv('host_species.csv', index_col=0)
hosts_with_coi

# Compile COI sequences into a FASTA file to do multiple sequence alignment.

coi_sequences = []
for r, d in hosts_with_coi.iterrows():
    if not pd.isnull(d['sequence']):
        seq = Seq(d['sequence'])
        seqrecord = SeqRecord(seq, id='{0}.{1}'.format(d['host_species'].replace(' ', '_'), d['TOL_species_name'].replace(' ', '_')))
        coi_sequences.append(seqrecord)
SeqIO.write(coi_sequences, 'host_coi_unaligned.fasta', 'fasta')

# After aligning using clustal omega (default parameters), load back the alignment.
coi_aligned = AlignIO.read('host_coi_aligned.fasta', 'fasta')
coi_aligned

# To identify where to trim the alignment, look at the number of gaps.

num_gaps = dict()
for i in range(coi_aligned.get_alignment_length()):
    num_gaps[i] = Counter(coi_aligned[:,i])['-']
plt.plot(list(num_gaps.keys()), list(num_gaps.values()))
plt.xlabel('position in alignment')
plt.ylabel('number of gap characters')

# Given this distribution of gaps, we will use a cut-off of 3 gaps to trim the alignment.
# i.e. if there are more than 3 gaps at that position, we trim that position out.

coi_df = pd.DataFrame([s for s in coi_aligned])
index = [s.id for s in coi_aligned]

for i in range(coi_aligned.get_alignment_length()):
    num_gaps = Counter(coi_aligned[:,i])['-']
    if num_gaps > 3:
        coi_df = coi_df.drop(i, axis=1)
coi_df.index = index
coi_df

# Now, I have to concatenate the sequences back into a single string.
trimmed_coi = []
for host_name, letters in coi_df.iterrows():
    sequence = ''
    for letter in coi_df.ix[host_name]:
        sequence += letter
    seq = Seq(sequence)
    seqrecord = SeqRecord(seq, description='', id=host_name, name='')
    trimmed_coi.append(seqrecord)
SeqIO.write(trimmed_coi, 'host_coi_trimmed.fasta', 'fasta')

trimmed_coi

# Filter out COI sequences such that only those without gaps are left. This is so that we can do a phylogenetic tree.
no_gaps = []
for s in trimmed_coi:
    if '-' not in s.seq:
        no_gaps.append(s)
no_gaps

SeqIO.write(no_gaps, 'host_coi_nogaps.fasta', 'fasta')
SeqIO.write(no_gaps, 'host_coi_nogaps.phylip', 'phylip')

# Get the distribution of hamming distances.
from itertools import combinations
from Levenshtein import distance
distances = []
for s1, s2 in combinations(no_gaps, 2):
    s1 = str(s1.seq)
    s2 = str(s2.seq)
    distances.append(distance(s1,s2))

plt.hist(distances)

get_ipython().getoutput(' raxmlHPC -p 100 -# 3 -m GTRGAMMA -s host_coi_nogaps.fasta -n host_coi_nogaps.tree -T 2')

get_ipython().getoutput(' ls *.tree')

from dendropy import Tree
from dendropy.calculate.treemeasure import PatristicDistanceMatrix
coi_tree = Tree.get(file=open('RAxML_bestTree.host_coi_nogaps.tree', 'r'),
                    schema='newick')


coi_pds = PatristicDistanceMatrix(coi_tree)
coi_pds.sum_of_distances()

taxon1 = coi_tree.leaf_nodes()[0].taxon
taxon2 = coi_tree.leaf_nodes()[1].taxon

taxon2 = coi_tree.leaf_nodes()[1].taxon.__str__()
taxon2.replace("'","")

coi_pds.__call__(taxon1, taxon2)

patristic_distances = nx.Graph()
pds = []
for taxon1, taxon2 in product(coi_tree.leaf_nodes(), coi_tree.leaf_nodes()):
    taxon1 = taxon1.taxon
    taxon2 = taxon2.taxon
    
    pd = coi_pds.__call__(taxon1, taxon2)
    t1 = taxon1.__str__().replace("'","").split('.')[0]
    t2 = taxon2.__str__().replace("'","").split('.')[0]
    patristic_distances.add_edge(t1, t2, pd=pd)
    pds.append(pd)

plt.hist(pds)

max(pds)

min(pds)

nx.write_gpickle(patristic_distances, 'supp_data/patristic_distances.pkl')

patristic_distances.edge['Mallard']




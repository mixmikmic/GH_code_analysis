
### This parses a fasta file and searches for a particular set of ids (names_to_search), and printing

names_to_search = ['hsa-let-7g-5p', 'let-7b-5p']
from Bio import SeqIO
handle = open("/projects/ps-yeolab/genomes/mirbase/release_21/mature.fa", "rU")
for record in SeqIO.parse(handle, "fasta"):
    for name in names_to_search:
        if name in record.id:
            print(record.id)
            print(record.seq)
handle.close()

### This parses a fasta file and searches for a particular set of ids (names_to_search), and writing to a separate file

from Bio import SeqIO
names_to_search = ['rna', 'RNA', 'alu']
handle = open("/projects/ps-yeolab/genomes/RepBase18.05.fasta/all.ref", "rU")
outfile = "/home/bay001/projects/encode/analysis/tests/eclip_tests/small_repelements/small_repelements.fa"
records = []
x = 0
for record in SeqIO.parse(handle, "fasta"):
    for name in names_to_search:
        if name in record.id:
            x = x + 1
            print('[{}]'.format(x)),
            records.append(record)
SeqIO.write(records,outfile,"fasta")
handle.close()
            

### This parses a fasta file and searches for a particular set of ids (names_to_search), and printing
from Bio import SeqIO
import pandas as pd

lengths = {}
handle = open("/projects/ps-yeolab/genomes/mirbase/release_21/mature.fa", "rU")
for record in SeqIO.parse(handle, "fasta"):
    lengths[record.id] = len(record.seq)
handle.close()
pd.DataFrame(lengths,index=['len']).T

### This parses a fasta file renames the ID

from Bio import SeqIO
handle = open("/projects/ps-yeolab/genomes/ce11/ws245_genes.ucsctable.fa", "rU")
outfile = "/projects/ps-yeolab/genomes/ce11/ws245_genes.ucsctable.fix_genenames.fa"
records = []
x = 0
for record in SeqIO.parse(handle, "fasta"):
    record.id = record.id.split('_')[2]
    records.append(record)
SeqIO.write(records,outfile,"fasta")
handle.close()
            

### This gets len of each sequence

from Bio import SeqIO
handle = open("/projects/ps-yeolab/genomes/ce10/ce10.fa", "rU")

for record in SeqIO.parse(handle, "fasta"):
    print len(record.seq), record.name
handle.close()
            

### This gets len of each sequence

from Bio import SeqIO
handle = open("/projects/ps-yeolab/genomes/ce10/chromosomes/all.fa", "rU")

for record in SeqIO.parse(handle, "fasta"):
    print len(record.seq), record.name
handle.close()
            

from Bio import SeqIO
def get_seq_dict_from_file(f, seq_ids=[], file_type='fasta', equal=True):
    """
    Returns dictionary of {name : sequence}
     
    Parameters
    ----------
    
    f : basestring
        file location of a fasta file
    seq_ids : list
        list of sequence ids to search. Empty field returns all sequences
    equal : bool
        True if seq_id needs to be identical
        False if we just have partial identifier
    file_type : basestring
        default "fasta" type file
    Returns
    -------
    records : dict
        {name:sequence}
    """
    records = {}
    for record in SeqIO.parse(f, file_type):
        if len(seq_ids) > 0:
            for name in seq_ids:
                if equal:
                    if name == record.id:
                        records[record.id] = record.seq
                else:
                    if name in record.id:
                        records[record.id] = record.seq
        else:
            records[record.id] = record.seq
    return records

def get_seq_sizes(f, seq_ids=[], file_type='fasta', equal=True):
    """
    Returns dictionary of {name : seqsize}
    
    Parameters
    ----------
    f
    seq_ids
    equal
    file_type

    Returns
    -------

    """
    lengths = {}
    records = get_seq_dict_from_file(f, seq_ids, file_type, equal)
    
    for seq_id, sequence in records.iteritems():
        lengths[seq_id] = len(sequence)
    return lengths

f = "/projects/ps-yeolab/genomes/hg19/chromosomes/all.fa"
seq_ids = []

### this subsets the fasta file to just contain XYZ number of bases

from Bio import SeqIO
handle = open("/projects/ps-yeolab3/cellrangerdatasets/chr19.fa", "rU")
outfile = "/projects/ps-yeolab3/cellrangerdatasets/hg19chr19kbp550_CELLRANGER_REFERENCE/chr19.1M.fa"
records = []
for record in SeqIO.parse(handle, "fasta"):
    record.seq = record.seq[:1000000]
    records.append(record)
    print(len(record.seq))
SeqIO.write(records,outfile,"fasta")
handle.close()






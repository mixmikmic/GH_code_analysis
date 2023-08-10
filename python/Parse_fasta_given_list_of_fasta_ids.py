names_to_search = ["unmapped-49-contig_list_contig_67081-0"]



### This parses a fasta file and searches for a particular set of ids (names_to_search), and printing

# names_to_search = ['TRINITY_DN68727_c0_g1','TRINITY_DN73280_c0_g1']
from Bio import SeqIO
handle = open("/home/bay001/projects/kes_20160307/data/kestrel5-reclustered.no-aerv.no-mtdna.no-vec.no-virus.no-bac.200.fasta", "rU")
for record in SeqIO.parse(handle, "fasta"):
    for name in names_to_search:
        if name in record.id:
            print(record.id)
            print(record.seq)
handle.close()

### This parses a fasta file and searches for a particular set of ids (names_to_search), and writing to a separate file

from Bio import SeqIO
handle = open("/home/bay001/projects/kes_20160307/analysis/final_scaffolds/Trinity/Trinity.fixheader.fasta", "rU")
outfile = "/home/bay001/projects/kes_20160307/analysis/final_scaffolds/Trinity/diffexp/tbbpa-unannotated.fasta"
records = []
x = 0
for record in SeqIO.parse(handle, "fasta"):
    for name in names_to_search:
        if name in record.id:
            x = x + 1
            print('[{}]'.format(x)),
            record.id = record.id.split(' ')[0]
            records.append(record)
SeqIO.write(records,outfile,"fasta")
handle.close()




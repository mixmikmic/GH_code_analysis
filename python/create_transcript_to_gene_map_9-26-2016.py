import pandas as pd
import numpy as np

current = '/home/bay001/projects/kes_20160307/data/kestrel5-reclustered.no-aerv.no-mtdna.no-vec.no-virus.no-bac.200.fasta'
blast_file = '/home/bay001/projects/kes_20160307/data/diamond/all.blast'
blast_head = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch',
              'gapopen', 'qstart', 'qend', 'sstart', 'send',
              'evalue', 'bitscore']
blast = pd.read_table(blast_file,names=blast_head)
blast.head()

from Bio import SeqIO
records = {'header':list()}
i = 0
handle = open(current, "rU")
for record in SeqIO.parse(handle, "fasta"):
    records['header'].append(record.id)
    i = i + 1
handle.close()
print('{} records parsed.'.format(i))
records = pd.DataFrame(records)

genes_from_blast = pd.concat([blast['qseqid'],blast['sseqid']],axis=1)
genes_from_blast.drop_duplicates(inplace=True)
genes_from_blast.head()

gene_to_trans = pd.concat([records['header'],records['header']],axis=1)
gene_to_trans.to_csv('/home/bay001/projects/kes_20160307/data/RSEM/gene_to_trans.map',
                    sep='\t',
                    header=None,
                    index=None)




import pandas as pd
import numpy as np
import os
import glob
from Bio import SeqIO

wd = '/home/bay001/projects/kes_20160307/org/03_output/assembly/gene_sequences'
all_genes = glob.glob(os.path.join(wd,'*.clustered.fasta'))

genelist = []
for gene in all_genes:
    genelist.append(os.path.basename(gene).replace('.clustered.fasta',''))

df = pd.DataFrame(genelist)
df.columns = ['Gene ID']
ensembl2gene = pd.read_table('/home/bay001/projects/kes_20160307/org/00_data/references/biomart/galgal4_biomart.txt').fillna('NONAME')

merged = pd.merge(df,ensembl2gene,how='left',on=['Gene ID'])
merged.to_csv('/home/bay001/projects/kes_20160307/data/ensembl-genename-description.tsv',sep='\t',index=None)
merged

goi = ['ENSGALG00000010736','ENSGALG00000026005','ENSGALG00000002550','ENSGALG00000007809',
       'ENSGALG00000013672','ENSGALG00000004528','ENSGALG00000015917','ENSGALG00000020180',
      'ENSGALG00000006098','ENSGALG00000011171','ENSGALG00000010725','ENSGALG00000028600']
goi = pd.DataFrame(goi)
goi.columns = ['Gene ID']
merged2 = pd.merge(goi,merged,how='left',on='Gene ID')
merged2.to_csv('/home/bay001/projects/kes_20160307/data/ensembl-genename-description-goi.tsv',sep='\t',index=None)

merged


def get_seq(df):
    records = {}
    i = 0
    for gene in all_genes:
        ensg = os.path.basename(gene).replace('.clustered.fasta','')
        if ensg in list(df['Gene ID']):
            gene_name = df[(df['Gene ID'].str.contains(ensg))]['Associated Gene Name'].to_string(index=False)
            handle = open(gene, "rU")
            for record in SeqIO.parse(handle,"fasta"):
                records[i] = (ensg, gene_name, record.id)
                i = i + 1
    df = pd.DataFrame(records).T
    df.columns = ['ensembl','name','contig_name']
    return df

get_seq(merged).to_csv('/home/bay001/projects/kes_20160307/org/03_output/csvs/contig_and_gene_name.txt',sep='\t',index=None)

get_ipython().system(' head /home/bay001/projects/kes_20160307/org/03_output/csvs/contig_and_gene_name.txt')

all_genes[:5]




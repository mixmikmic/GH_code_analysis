import pandas as pd
import numpy as np
import os

wd = '/home/bay001/projects/kes_20160307/data/'
blast_head = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch',
              'gapopen', 'qstart', 'qend', 'sstart', 'send',
              'evalue', 'bitscore']
counts = '/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/counts.RSEM.txt'
get_ipython().system(' wc -l $counts')

evalue_threshold = 1e-10
blast = pd.read_table(os.path.join(wd,'blast/chicken.blastx'),names=blast_head)
print("number of starting hits: {}".format(blast.shape[0]))
blast = blast[blast['evalue'] < evalue_threshold]
print("number of filtered for e-value hits: {}".format(blast.shape[0]))
blast.sort_values(by='evalue',inplace=True)
blast.drop_duplicates('qseqid',inplace=True, keep='first')
print("number of filtered hits for de-duplicated contigs: {}".format(blast.shape[0]))
blast.set_index('qseqid',inplace=True)
blast.head()

contig2chicken = blast.loc[:,['sseqid']]
contig2chicken.rename(columns={'sseqid':'ensembl'},inplace=True)
contig2chicken.head()

uniref = pd.read_table(os.path.join(wd,'diamond/all.blast'),names=blast_head)
print("number of starting hits: {}".format(uniref.shape[0]))
uniref = uniref[uniref['evalue'] < 1e-10]
print("number of filtered for e-value hits: {}".format(uniref.shape[0]))
uniref.sort_values(by='evalue',inplace=True)
uniref.drop_duplicates('qseqid',inplace=True, keep='first')
print("number of filtered hits for de-duplicated contigs: {}".format(uniref.shape[0]))
uniref.head()

uniref_translation = pd.read_table('/home/bay001/projects/kes_20160307/data/uniref90.headers',names=['uniref'])
uniref_translation.head(2)

uniref_translation = uniref_translation['uniref'].str.replace('>','')

uniref2gene = uniref_translation.str.extract('(^[\w\d-]+).+ RepID=([\w-]+)$')

uniref2gene.reset_index(inplace=True)
uniref2gene.head()

# save this intermediate step because this takes a long time.
uniref2gene.to_csv('/home/bay001/projects/kes_20160307/data/uniref2gene.txt',sep='\t',header=None,index=None)

uniref2gene = pd.read_table('/home/bay001/projects/kes_20160307/data/uniref2gene.txt',names=[0,1])
uniref2gene.head(2)

# make sure we capture all annotations, that the regex expression is correct. This list should be empty.
uniref2gene[uniref2gene.isnull().any(axis=1)]

uniref2blast = pd.merge(uniref,uniref2gene,how='left',left_on='sseqid',right_on=0)
uniref2blast.head()



contig2uniref = uniref2blast.loc[:,['qseqid','sseqid']]
contig2uniref.set_index('qseqid',inplace=True)
contig2uniref.rename(columns={'sseqid':'uniref'},inplace=True)
print(contig2uniref.shape)
contig2uniref.drop_duplicates() # Sanity check.
print(contig2uniref.shape)
contig2uniref.head()

uniref2blast[uniref2blast[0]=='UniRef90_Q9H3D4'] # more sanity check. Make sure this is mapping to just one

countsdf = pd.read_table(counts,index_col=0)
print(countsdf.shape[0])
countsdf.head(2)

merged_ensembl = pd.merge(countsdf,contig2chicken,how='left',left_index=True,right_index=True)
merged_ensembl.head(2)

# contig2chicken.ix['EC-4AK111_TAGCTT_R1_(paired)_contig_1003-0']
contig2chicken.drop_duplicates().shape

# Print some basic stats after annotating with ensembl blast hits
print("number of total contigs: {}".format(merged_ensembl.shape[0]))
print("number of annotated contigs: {}".format(merged_ensembl.shape[0] - 
                                               merged_ensembl[merged_ensembl.isnull().any(axis=1)].shape[0]))
print("number of still missing annotated contigs: {}".format(merged_ensembl[merged_ensembl.isnull().any(axis=1)].shape[0]))

merged_ensembl_uniref = pd.merge(merged_ensembl,contig2uniref,how='left',left_index=True,right_index=True)
merged_ensembl_uniref.head(2)

merged_ensembl_uniref['annotation'] = merged_ensembl_uniref['ensembl']
merged_ensembl_uniref['annotation'].fillna(merged_ensembl_uniref['uniref'],inplace=True)
merged_ensembl_uniref.head()

# Print some basic stats after annotating with ensembl blast hits
print("number of total contigs: {}".format(merged_ensembl_uniref.shape[0]))
print("number of annotated contigs: {}".format(merged_ensembl_uniref.dropna(subset=['annotation']).shape[0]))
print("number of uniref annotations not ensembl: {}".format(merged_ensembl_uniref[merged_ensembl_uniref['annotation'].str.contains('(UniRef)')==True].shape[0]))
print("number of ensembl annotations: {}".format(merged_ensembl_uniref[merged_ensembl_uniref['annotation'].str.contains('(ENSG)')==True].shape[0]))

# translate counts into an annotated counts table.
merged_ensembl_uniref.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/annotations.txt',sep='\t')
new_counts_df = merged_ensembl_uniref.reset_index()
new_counts_df['annotation'].fillna(new_counts_df['gene_id'],inplace=True)
del new_counts_df['ensembl']
del new_counts_df['uniref']
del new_counts_df['gene_id']
cols = new_counts_df.columns.tolist()
cols.insert(0,cols.pop(cols.index('annotation')))
new_counts_df = new_counts_df.reindex(columns = cols)
new_counts_df.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/counts.RSEM.txt.annotated',sep='\t',
                    index=None)
new_counts_df.head()

# give the counts table gene names
merged_ensembl_uniref['uniref'].dropna().to_csv('/home/bay001/projects/kes_20160307/data/uniref_ids.txt',index=None)

ensembl2name = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/ensembl_to_genename.txt')
ensembl2name.head(2)

new_counts_df2 = pd.merge(new_counts_df,ensembl2name,how='left',left_on='annotation',right_on='Ensembl Gene ID')
new_counts_df2['Associated Gene Name'].fillna(new_counts_df2['annotation'],inplace=True)
del new_counts_df2['Ensembl Gene ID']
del new_counts_df2['annotation']
cols = new_counts_df2.columns.tolist()
cols.insert(0,cols.pop(cols.index('Associated Gene Name')))
new_counts_df2 = new_counts_df2.reindex(columns = cols)
# new_counts_df2.rename(columns={'Associated Gene Name':'annotation'},inplace=True)
new_counts_df2.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/counts.RSEM.txt.gene-name.annotated',sep='\t',
                    index=None)
new_counts_df2.head()

allcontigs = pd.DataFrame(countsdf.reset_index()['gene_id'])
allcontigs.rename(columns = {'gene_id':'qseqid'},inplace=True)
allcontigs['gene'] = np.nan

gene_to_trans = pd.concat([contig2chicken.rename(columns={'ensembl':'gene'}),
                           contig2uniref.rename(columns={'uniref':'gene'}),
                           allcontigs.set_index('qseqid')])

gene_to_trans = gene_to_trans.reset_index().drop_duplicates(subset='qseqid')
gene_to_trans.head()

# reorder gene-to-trans map
cols = gene_to_trans.columns.tolist()
cols.insert(0,cols.pop(cols.index('gene')))
gene_to_trans = gene_to_trans.reindex(columns = cols)
gene_to_trans.head()

# Uh oh. Looks like the gene to trans map contains more transcripts that don't exist in the current assembly.
print(gene_to_trans.shape[0])
get_ipython().system(' wc -l /home/bay001/projects/kes_20160307/data/kestrel.headers')

X = pd.read_table('/home/bay001/projects/kes_20160307/data/kestrel.headers',names=['headers'])
X['qseqid'] = X['headers'].str.replace('>','')
del X['headers']
Y = pd.merge(X,gene_to_trans,how='left', on='qseqid')
Y['gene'].fillna(Y['qseqid'],inplace=True)
# reorder gene-to-trans map
cols = Y.columns.tolist()
cols.insert(0,cols.pop(cols.index('gene')))
Y = Y.reindex(columns = cols)
Y.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/gene_to_trans.map',
                    sep='\t',header=None,index=None)

Y.shape # now contains the exact number of contigs (transcripts) to corresponding genes. 

# How many gene names are captured?
new_counts_df3 = pd.merge(new_counts_df,ensembl2name,how='left',left_on='annotation',right_on='Ensembl Gene ID')
len(set(new_counts_df3['Associated Gene Name'].dropna()))

testdf = new_counts_df3[new_counts_df3['annotation'].duplicated()==True].sort_values('annotation').drop_duplicates('annotation')
del testdf['Ensembl Gene ID']
del testdf['Associated Gene Name']
testdf.to_csv('/home/bay001/projects/kes_20160307/data/counts.TEST.txt',sep='\t',index=None)

cdf = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/counts.RSEM.txt')

cdf[cdf['gene_id'].str.contains('ENSG')].shape




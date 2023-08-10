get_ipython().system(' wget http://mbe.oxfordjournals.org/content/suppl/2008/09/25/msn214.DC1/mbe-08-0522-File008_msn214.xls')

import pandas as pd

gene_age = pd.read_excel('mbe-08-0522-File008_msn214.xls', sheetname='Table S3a', skiprows=1)
gene_age.head()

gene_age = gene_age.set_index('Gene_ID')
gene_age = gene_age.rename(columns={'Phylostratum': 'domazetloso2008_phylostratum'})
gene_age.head()

gene_age.to_csv('')


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob

rnaseq_dir = '/projects/ps-yeolab3/bay001/hundley_rnae_20160210/analysis/rnae_v6/'
editing_calls_dir = '/projects/ps-yeolab3/bay001/hundley_rnae_20160210/analysis/editing_calls_v18/ws257/'
wd = '/projects/ps-yeolab3/bay001/hundley_rnae_20160210/analysis/editor_response_minor_1/ws257/'
old_wd = '/projects/ps-yeolab3/bay001/hundley_rnae_20160210/analysis/editing_calls_v15_final/ws254/'

region = 'CDS'

# load the previous annotations
old_ws254 = pd.read_table(
    os.path.join(old_wd,'supplemental_doc_1.txt')
)
ws254_regions = old_ws254[old_ws254['Region']==region.replace('_',' ').replace('utr','UTR')]
print("number of ws254 {} called: {}".format(region, ws254_regions.shape[0]))


# load current annotations
new_ws257 = pd.read_table(
    os.path.join(wd, 'supplemental_doc_1.txt')
)
ws257_regions = new_ws257[new_ws257['Region']==region]
print("number of ws257 {} called: {}".format(region, ws257_regions.shape[0]))
ws257_regions.head()

merged = pd.merge(
    old_ws254, new_ws257, 
    how='outer', 
    left_on=['Chromosome','Position'],
    right_on=['Chromosome','Position']
)

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']!='three prime UTR') & (merged['Region_y']=='three_prime_utr')]

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']=='three prime UTR') & (merged['Region_y']!='three_prime_utr')]

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']!='five prime UTR') & (merged['Region_y']=='five_prime_utr')]

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']=='five prime UTR') & (merged['Region_y']!='five_prime_utr')]

# regions that were introns using the old annotations, but not called intronic in the new
region = 'intron'
merged[(merged['Region_x']!=region) & (merged['Region_y']==region)]

# regions that were introns using the old annotations, but not called intronic in the new
region = 'intron'
merged[(merged['Region_x']==region) & (merged['Region_y']!=region)]

# regions that were introns using the old annotations, but not called intronic in the new
region = 'CDS'
merged[(merged['Region_x']!=region) & (merged['Region_y']==region)]

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']==region) & (merged['Region_y']!=region)]

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']!='downstream from gene') & (merged['Region_y']=='downstream_gene')]

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']=='downstream from gene') & (merged['Region_y']!='downstream_gene')]

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']!='upstream from gene') & (merged['Region_y']=='upstream_gene')]

# regions that were introns using the old annotations, but not called intronic in the new
merged[(merged['Region_x']=='upstream from gene') & (merged['Region_y']!='upstream_gene')]




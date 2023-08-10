import pandas as pd
import seaborn as sb
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

sra = pd.read_csv('../../output/chip/20171103_s2cell_chip-seq.csv')
sra.head()

modENCODE = pd.read_table('../../data/modENCODE_sampletable.tsv')
modENCODE.head()

len(modENCODE.srx.unique())

merged = sra.merge(modENCODE, on='srx', how='inner')

merged.shape

merged.modENCODE_id.unique()

mymod = pd.read_table('../../output/chip/modENCODE_big.bed', header=None, names=['chrom', 'start','stop','score','strand','modENCODE_id','type','stage','otherid'])

check = mymod.merge(merged, on='modENCODE_id', how='inner')

check.srr_x.unique()

check.geo.unique()

other_data = check[['chrom','start','stop','srx']]
other_data.to_csv('../../output/chip/modE_s2.bed', header=None, index=False, sep='\t')

SRX016158_narrow = pd.read_table('../../chipseq-wf/data/chipseq_peaks/macs2/CTCF-1-narrow/peaks.bed', header=None)

coverage = pd.read_table('../../output/chip/peakintersecttest', header=None)[[0,1,2,3,12]]
coverage.columns = ['chrom','start','end','name','overlap']

sb.distplot(coverage['overlap'])

filtered = coverage[coverage.overlap > 0.01]

filtered['chrom'] = ['chr'+str(x) for x in filtered.chrom]

filtered[['chrom','start','end','name']].to_csv('../../output/chip/filteredpeaks', header=None, index=False, sep='\t')

h3k9 = pd.read_csv('../../output/chip/results-table.csv')

h3k9bed = h3k9[['Binding Site > Chromosome > DB identifier',
       'Binding Site > Chromosome Location > Start',
       'Binding Site > Chromosome Location > End',]]

h3k9bed.to_csv('../../output/chip/h3k9.bed', header=None, index=False, sep='\t')

fixbed = pd.read_table('../../output/chip/h3k9.bed', header=None)

fixbed[0] = ['chr'+str(x) for x in fixbed[0]]

fixbed.to_csv('../../output/chip/liftmeover.bed', header=None, index=False, sep='\t')

coverage2 = pd.read_table('../../output/chip/peakcoverage_test2', header=None)

coverage2.shape

sb.distplot(coverage2[12])

test_coverage = pd.read_table('../../output/chip/test_coverage.bed', header=None)[[3,4]]

test_coverage['gene_id'] = [x.split(';')[0].split()[1].strip('"')for x in test_coverage[3]]
test_coverage['gene_symbol'] = [x.split(';')[1].split()[1].strip('"')for x in test_coverage[3]]
test_coverage['count'] = test_coverage[4]
test_coverage = test_coverage[['gene_id','gene_symbol','count']]

mod_coverage = pd.read_table('../../output/chip/modENCODE_coverage.bed', header=None)

mod_coverage['gene_id'] = [x.split(';')[0].split()[1].strip('"')for x in mod_coverage[3]]
mod_coverage['gene_symbol'] = [x.split(';')[1].split()[1].strip('"')for x in mod_coverage[3]]
mod_coverage['count'] = mod_coverage[4]
mod_coverage = mod_coverage[['gene_id','gene_symbol','count']]

mod_coverage.head()

test_coverage.head()

sb.distplot(mod_coverage['count'])

sb.distplot(test_coverage['count'])

merged = test_coverage.merge(mod_coverage, how='outer', on=['gene_id', 'gene_symbol'])

merged['difference'] = abs(merged.count_x - merged.count_y)

merged.head()

merged.difference.describe()

merged['count_x'].corr(merged['count_y'], method='spearman')

merged.shape

import matplotlib.pyplot as plt

sb.distplot(merged.difference)

from math import sqrt
 
def euclidean_distance(x,y):
 
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

euclidean_distance(merged.count_x, merged.count_y)

results = []
for bs in range(10000):
    sub1 = merged.count_x.sample(n=17728, replace=True)
    sub2 = merged.count_y.sample(n=17728, replace=True)
    metric = euclidean_distance(sub1, sub2)
    results.append(metric)

sb.distplot(results)




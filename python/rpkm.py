

import pandas

counts_file = 'counts.txt'

def counts_to_rpkm(feature_counts_table):
    counts = feature_counts_table.ix[:,5:]
    lengths = feature_counts_table['Length']
    mapped_reads = counts.sum()
    return (counts * pow(10,9)).div(mapped_reads, axis=1).div(lengths, axis=0)

counts_table = pandas.read_table(counts_file, index_col=0, skiprows=1)

counts_to_rpkm(featureCountsTable)




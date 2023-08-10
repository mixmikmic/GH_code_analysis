get_ipython().magic('run notebook.config.ipy')

statement = '''select * from featurecounts'''

counts = DB.fetch_DataFrame(statement, db)
count_table = counts.pivot(columns="track", index="gene_id", values="counts")

print count_table.shape

DB.write_DataFrame(count_table,"count_table",ipydb)
#count_table.to_csv("count_table.txt",sep="\t")

# e.g. select cells:
# (i) expressing more than 3000 genes and
# (ii) where <50% reads map to spike-in sequences and
# (iii) whic have less than 7 million reads.

statement = '''select sample_id 
               from qc_summary q
               where q.cufflinks_no_genes_pc > 3000
                 and q.fraction_spike < 0.5
                 and q.total_reads < 7000000'''

good_samples = DB.fetch_DataFrame(statement, db)["sample_id"].values

print len(good_samples)

sample_stat = '"' + '", "'.join(good_samples) + '"'

# fetch a pandas dataframe containing the "good" cells (samples)
statement = '''select gene_id, %(sample_stat)s
               from count_table''' % locals()

count_table_filtered = DB.fetch_DataFrame(statement, ipydb)
print count_table_filtered.shape

# write a new frame containing the filtered data
DB.write_DataFrame(count_table_filtered, "count_table_filtered", ipydb)


import pandas as pd
import numpy as np
import os

results = get_ipython().getoutput('ls /home/bay001/projects/kes_20160307/data/RSEM/*.genes.results')

results = results[9:]+results[:9]
results

MASTER = pd.read_table(results[0],index_col=0)
master = pd.DataFrame(MASTER['expected_count'])
master.rename(columns={'expected_count':os.path.basename(results[0]).replace('.genes.results','')}, inplace=True)
for i in range(1,len(results)):
    X = pd.read_table(results[i],index_col=0)
    x = pd.DataFrame(X['expected_count'])
    x.rename(columns={'expected_count':os.path.basename(results[i]).replace('.genes.results','')}, inplace=True)
    master = pd.merge(master, x, how='outer', left_index=True, right_index=True)
master.head()

master.round().astype(int).to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/counts.RSEM.txt',
                                 sep='\t',
                                 )

test = master.reset_index()

test[test['gene_id']=='ENSGALG00000000019']




get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
from tqdm import tnrange, tqdm_notebook
import pysam

jxc_dir = '/home/elvannostrand/data/ENCODE/RNAseq/scripts/exon_junction_counts/encode_rnaseq_psis_20170321/'
jxc_i = glob.glob(os.path.join(jxc_dir,'*.bam.namesort.bam.int_jxn_counts'))
jxc = glob.glob(os.path.join(jxc_dir,'*.bam.namesort.bam.jxn_counts'))
bam = '/home/bay001/projects/encode/analysis/tests/eric_retained_intron/ENCFF385UCY.sorted.bam'

all_controls = pd.read_table('/projects/ps-yeolab3/encode/k562_brenton-graveley_ambiguous_bams_for_integrated_analysis.txt')
all_controls_r1 = set(all_controls['control_rep1'])
all_controls_r2 = set(all_controls['control_rep2'])
ac = all_controls_r1.union(all_controls_r2)
ac = set(ac)

stop_at = 10
bad = []
ctrl_bams = ac

c = 0
ac = ['ENCFF157TKC.bam']
progress = tnrange(len(ac))
for b in ac:
    original = '/projects/ps-yeolab3/encode/rnaseq/shrna_knockdown_graveley_tophat/{}'.format(b)
    bam = '/home/bay001/projects/encode/analysis/tests/eric_retained_intron/{}'.format(b)
    if not os.path.exists(bam):
        get_ipython().system(' ln -s $original $bam')
    if not os.path.exists(bam + '.bai'):
        get_ipython().system(' samtools index $bam')
    i = '/home/elvannostrand/data/ENCODE/RNAseq/scripts/exon_junction_counts/encode_rnaseq_psis_20170321/{}.namesort.bam.int_jxn_counts'.format(b)
    samfile = pysam.AlignmentFile(bam, "rb", )
    # for jct in dfi['jct'].head():
    dfi = pd.read_table(i, names = ['chrom','strand','jct','num'])
    # dfi = pd.DataFrame(dfi.ix[3]).T
    for col, row in dfi.iterrows():
        c += 1
        names = []
        for read in samfile.fetch(row['chrom'], row['jct'], row['jct']+1):
            # print(read.get_reference_positions()[0], row['jct'])
            if 'N' not in read.cigarstring:
                if row['jct'] - read.get_reference_positions()[0] > 10 and read.get_reference_positions()[-1] - row['jct'] > 10: # sufficient overhang near read start
                    names.append(read.query_name)
        if c > stop_at:
            break
        if (len(set(names)) != row['num']):
            bad.append([b,set(names), row['num'], row['jct']])
        
        
    if c > stop_at:
        break
    # print("{}:{} ".format(b,len(bad))),
    
    progress.update(1)

bad

def make_bedtool(eric, offset=0):
    chrom, strand, pos = eric.split(':')
    start, end = pos.split('-')
    return pybedtools.create_interval_from_list([
            chrom, str(int(start) + offset), str(int(start) + offset + 1), 'start', '0', strand
        ])
def count_jcts(jct_df):
    bts = []
    for junc in jct_df['jct']:
        bts.append(make_bedtool(junc))
    return pybedtools.BedTool(bts)

bts = count_jcts(df).sort()

b = pybedtools.BedTool(bam)
bts[0]

btx = bts.intersect(b, sorted=True, wa=True, wb=True)

dfx = btx.to_dataframe()



single_junction_site = pybedtools.create_interval_from_list(['chr1','40875508','40875509','name','0','+'])
single_junction_site = pybedtools.BedTool(single_junction_site)
single_junction_site

dfy = single_junction_site.intersect(b, wa=True, wb=True)
dfy


samfile = pysam.AlignmentFile(bam, "rb", )
c = 0
for read in samfile.fetch('chr1', 40875507, 40875508):
     if 'N' not in read.cigarstring:
        # print(read.get_reference_positions()[0] - 40875508, read.query_name, read.flag, read.cigarstring, read.get_reference_positions()[0])
        c+=1
        # print(read.query_name)
print(c)

i = '/home/elvannostrand/data/ENCODE/RNAseq/scripts/exon_junction_counts/encode_rnaseq_psis_20170321/ENCFF475GCB.bam.namesort.bam.int_jxn_counts'

dfi = pd.read_table(i, names = ['chrom','strand','jct','num'])
dfi.head()





set(names)

pd.DataFrame(dfi)

i




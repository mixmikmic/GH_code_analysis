import pandas as pd
import seaborn as sb
import glob
import os
import pybedtools
from pybedtools.featurefuncs import gff2bed

#dropping this dataset -- 4 of the samples have 0 reads after cutadapt
throw_out = ['ERX242704', 'ERX242706', 'ERX242707','ERX242709', 'ERX242710','ERX242711','ERX242712','ERX242713',
             'ERX242715', 'ERX242716', 'ERX242717', 'ERX242718','ERX242722', 'ERX242723', 'ERX242724', 'ERX242726',
             'ERX242727', 'ERX242728', 'ERX242729']

#look into rerunning these with different parameters, 'SRX326969' shouldn't come up again 
empty = ['SRX495277','SRX495278','SRX495290','SRX495289']

#all peaks would be glob.glob('../chipseq-wf/data/chipseq_peaks/*/[S,E]RX*/peaks.bed')
#macs2 peaks
concat = []
for fname in glob.glob('../chipseq-wf/data/chipseq_peaks/macs2/[S,E]RX*/peaks.bed'):
    name = fname.split('../chipseq-wf/data/chipseq_peaks/macs2/')[1].split('/peaks.bed')[0]
    if name not in throw_out:
        if name not in empty:
            df = pd.read_table(fname, header=None)
            df['srx'] = name
            df['caller'] = 'macs2'
            concat.append(df)
macs2_full = pd.concat(concat)

macs2 = macs2_full[[0,1,2,'srx',8,'caller']]

#spp peaks
glob.glob('../chipseq-wf/data/chipseq_peaks/spp/[S,E]RX*/peaks.bed')
concat = []
for fname in glob.glob('../chipseq-wf/data/chipseq_peaks/spp/[S,E]RX*/peaks.bed'):
    name = fname.split('../chipseq-wf/data/chipseq_peaks/spp/')[1].split('/peaks.bed')[0]
    if name not in throw_out:
        if os.path.getsize(fname) != 0:
            df = pd.read_table(fname, header=None)
            df['srx'] = name
            df['caller'] = 'spp'
            concat.append(df)
spp_full = pd.concat(concat)

spp = spp_full[[0,1,2,'srx',8,'caller']]
spp[[1,2]]= spp[[1,2]].astype(int)

#Count number of peaks per dataset macs2 
peakcount_macs = macs2.groupby('srx')[[1]].count()
sb.distplot(peakcount_macs)

#Count number of peaks per dataset spp
peakcount_spp = spp.groupby('srx')[[1]].count()
sb.distplot(peakcount_spp)

both = pd.concat([macs2, spp])
#note: score is -log10 qvalue
both.columns = ['chrom','start','end','name','score','caller']
both.head()

#log 10 peak count distribution for all peak data
import numpy

peakcount = both.groupby('name')[['start']].count()
sb.distplot(numpy.log10(peakcount + 1))

peakcount.describe()

phantompeaks = pd.read_excel(
    '../output/chip/gkv637_Supplementary_Data/Supplementary_table_3__List_of_Phantom_Peaks.xlsx')
phantompeaks = phantompeaks[['chr ','start','end','Name']]

# Get peak data in bed format
both = both[~both['start'].astype(str).str.contains('-')]
bed = both[['chrom','start','end','name']]

intersect = pybedtools.BedTool.from_dataframe(bed).intersect(pybedtools.BedTool.from_dataframe(phantompeaks), 
                                                             wo=True).to_dataframe()

#filter for an overlap of at least 50bp based on Jain et al 2014
filtered = intersect[intersect.itemRgb >= 50]

filtered.head()

outermerge = both.merge(filtered, how='outer', on=['chrom','start','end','name'], indicator=True)

no_phantom = outermerge[outermerge._merge == 'left_only'][['chrom','start','end','name','score_x','caller']]
no_phantom.rename(columns={'score_x': 'score'}, inplace=True)
no_phantom.head()

no_phantom.shape

nophtm_peakcount = no_phantom.groupby('name').start.count()
nophtm_peakcount.describe()

sb.distplot(nophtm_peakcount)

spreadsheet = pd.read_csv('../output/chip/20171103_s2cell_chip-seq.csv')

#For now we are excluding datasets with no input: 
spreadsheet = spreadsheet[spreadsheet.input != 'no input?']
spreadsheet.head()

all_chromatin = spreadsheet[spreadsheet.chromatin == 1]
no_chromatin = spreadsheet[spreadsheet.chromatin == 0]

chromatin_nophantom = no_phantom[no_phantom.name.isin(all_chromatin.srx.values)]
tf_nophantom = no_phantom[no_phantom.name.isin(no_chromatin.srx.values)]

chr_peakcount = chromatin_nophantom.groupby('name').start.count()
chr_peakcount.describe()

sb.distplot(chr_peakcount)

tf_peakcount = tf_nophantom.groupby('name').start.count()
tf_peakcount.describe()

sb.distplot(tf_peakcount)

tf_peakcount[tf_peakcount < 1000]

chr_peakcount[chr_peakcount < 1000]

i1 = tf_peakcount[tf_peakcount < 1000].index
i2 = chr_peakcount[chr_peakcount < 1000].index

filtered_tf_np = tf_nophantom[~tf_nophantom.name.isin(i1)]
filtered_chr_np = chromatin_nophantom[~chromatin_nophantom.name.isin(i2)]

filtered_tf_np.groupby('name').count().describe()

filtered_chr_np.groupby('name').count().describe()

filtered_tf_np.to_csv('../output/chip/ALL_TF_CHIP_filtered.bed', header=False,index=False, sep='\t')

filtered_chr_np.to_csv('../output/chip/ALL_HIST_CHIP_filtered.bed', header=False,index=False, sep='\t')



spp_empty = ['SRX149192', 'SRX885700', 'ERX402137', 'ERX402138','SRX885698', 'SRX883604','SRX1179573','SRX054533',
'SRX495789', 'SRX1389384','SRX2055961','SRX2055966','SRX2055958', 'ERX402108','SRX330269','ERX402133','SRX306190',
'ERX402112','SRX359797','SRX1433400', 'SRX306193','ERX1403350', 'SRX1179572','SRX1433401','SRX018632','SRX1389387',
 'SRX326970','SRX2055964','SRX885702','SRX2055945','SRX326969', 'SRX447393','SRX330270','SRX495270','SRX2055944',
'SRX097620','SRX359798','SRX883605','SRX018631','SRX306196','SRX018629','SRX2055953','SRX149189','SRX1389388',
'SRX018630','SRX1433397','ERX402114','SRX495269','SRX1433399']

emptydf = pd.DataFrame(spp_empty, columns=['srx'])

spreadsheet.merge(emptydf, how='inner')[['srx','geo','target','srr','input']]




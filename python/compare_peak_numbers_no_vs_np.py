get_ipython().magic('matplotlib inline')
import pandas as pd
from matplotlib_venn import venn2

no = pd.read_table(
    '/home/bay001/projects/nazia_clipseq_20170627/permanent_data/eCLIP-0.1.5/idr/GT_T1.compressed.bed.entropy.full',
    names=['chrom','start','end','name','ipreads','inpreads',
           '.','.','.','.','l10p','l2fc','.']
)
no.head()

no = no[(no['l10p']>=3) & (no['l2fc']>=3)]
no.sort_values(by=['chrom','start','end']).head()

np = pd.read_table(
    '/home/bay001/projects/nazia_clipseq_20170627/permanent_data/eCLIP-0.1.5/clip_analysis/20171121/GT.IP_T1.---.r-.fqTrTrU-SoMaSoCoSoMeV2ClNpCoFc3Pv3.bed.thickStartEnd.bed',
    names=['chrom','start','end','name','score','strand','tstart','tend']
)
np.shape

np.sort_values(by=['chrom','start','end']).head()

npstart = set(np['start'])
nostart = set(no['start'])

venn2([npstart, nostart], ['Gabe','Eric'])




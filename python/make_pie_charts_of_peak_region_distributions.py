get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

ipnorm_manifest = pd.read_table('/home/bay001/projects/tbos_clipseq_20160809/analysis/recall_clips/recall_with_old/input_norm/input_norm_manifest.tsv')
uid2rbp = dict(zip(ipnorm_manifest.uID, ipnorm_manifest.RBP))
ipnorm_manifest



l10pv = 5
l2fcv = 3
wd = '/home/bay001/projects/tbos_clipseq_20160809/analysis/recall_clips/recall_with_old/'
input_norm_bed_head = ['chrom','start','end','l10p','l2fc','strand','annotation','gene']
allpeaks = get_ipython().getoutput('ls $wd/*.bed.annotated')
allpeaks[:1]

def get_region(row):
    return 'intergenic' if row['annotation'] == 'intergenic' else row['annotation'].split('|')[0]
def get_containment(row):
    return 'intergenic' if row['annotation'] == 'intergenic' else row['annotation'].split('|')[1]

for peak in allpeaks:
    df = pd.read_table(peak,names=input_norm_bed_head)
    name = uid2rbp[re.findall('([\w\d_]+)_[\d].$',os.path.basename(peak).split('.')[0])[0]]
    df['region'] = df.apply(get_region,axis=1)
    df['containment'] = df.apply(get_containment,axis=1)
    dfx = df[(df['l10p'] > l10pv) & (df['l2fc'] > l2fcv)]
    num_peaks = dfx.shape[0]

    regions = dfx['region'].value_counts().to_dict()
    labels = []
    sizes = []

    colors = sns.color_palette("hls", len(regions))

    for region, count in regions.iteritems():
        labels.append(region)
        sizes.append(count)

    """
    Plot pie
    """
    # The slices will be ordered and plotted counter-clockwise.

    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title(name,y=1.1)
    plt.tight_layout()

    plt.savefig(peak.replace('.annotated','.svg'))
    
    plt.close()
    












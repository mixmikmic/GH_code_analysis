get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pybedtools
import os
import functools
import numpy as np

def filter_input_norm_as_df(file_name, out_file, l2fc, pval):
    """
    Convenience method for me.

    Filters an "input norm"-formatted file given
    log2 fold change and log10 pvalue thresholds.
    See data/input_norm_bed.bed file for an example.

    Parameters
    ----------
    file_name : basename
    out_file : basename
    l2fc : float
    pval : float

    Returns
    -------
    filtered : pandas.DataFrame()

    """
    return filter_input_norm(file_name, l2fc, pval).saveas(out_file).to_dataframe()


def filter_input_norm(file_name, l2fc, pval):
    """
    Filters an "input norm"-formatted file given
    log2 fold change and log10 pvalue thresholds.
    See data/input_norm_bed.bed file for an example.

    Parameters
    ----------
    file_name : basename
    l2fc : float
    pval : float

    Returns
    -------
    filtered : pybedtools.BedTool()

    """
    try:
        bedtool = pybedtools.BedTool(file_name)
        filter_data_inst = functools.partial(filter_data, l2fc=l2fc, pval=pval)
        return bedtool.filter(filter_data_inst)
    except Exception as e:
        return 1


def filter_data(interval, l2fc, pval):
    """
    col4 is -log10 p-val
    col5 is -log2 fold enrichment

    Expects the standard input norm file format.

    Parameters
    ----------
    interval : pybedtools.Interval
    l2fc : float
    pval : float

    Returns
    -------

    """

    return (float(interval[4]) >= pval) and (float(interval[3]) >= l2fc)

clip_peak = '/home/bay001/projects/codebase/bfx/pyscripts/data/input_norm_bed.annotated'
l2fc = 3
pval = 3
out_file = '/home/bay001/projects/codebase/bfx/pyscripts/data/input_norm_bed.filtered.bed'

filter_input_norm_as_df(clip_peak, out_file, l2fc, pval).head()

# ip_l2fc = '/projects/ps-yeolab3/bay001/tbos/input_norm/rbfox2_input_norm/204_RBFOX2_ReadsByLoc_combined.csv.l2fcwithpval_enr.csv'
ip_l2fc = '/projects/ps-yeolab3/bay001/tbos/input_norm/input_norm_latest/A1_01_DAZAP1_ReadsByLoc_combined.csv.l2fcwithpval_enr.csv'
df = pd.read_table(
    ip_l2fc,
    index_col=0,
)
df = df.head()

def split_single_cols(df, col, sep='|'):
    """
    Splits a df['col'] into two separated by 'sep'
    """
    df["{} l2fc".format(col.split(sep)[1])],     df["{} l10p".format(col.split(sep)[1])] = zip(
        *df[col].map(lambda x: x.split(sep))
    )
    return df

def split_l2fcwithpval_enr(df, discard = True):
    """
    Splits a dataframe into its l2fc and log10 pvalue
    """
    for col in df.columns:
        df = split_single_cols(df, col)
        if discard:
            del df[col]
    return df

split_cols(df)


import pandas as pd

READSBYLOC_COMBINED_CSV_L2FCWITHPVAL_ENR_HEADERS_1IP1IN = [
    'ENSG','CDS','CDS-pvalue','5utr','5utr-pvalue','3utr','3utr-pvalue',
    '5_and_3_utr','5_and_3_utr-pvalue','intron','intron-pvalue',
    'intergenic','intergenic-pvalue','noncoding_exon','noncoding_exon-pvalue',
    'noncoding_intron','noncoding_intron-pvalue'
]

READS_BY_LOC_HEADER = [
    'ENSG', 'CDS', '3utr', '5utr', '5utr|3utr', 'intron', 
    'intergenic', 'noncoding_exon', 'noncoding_intron'
]

palette = sns.color_palette("hls", 8)
utr3 = palette[5] # blue
cds = palette[0] # red
intron = palette[4] # light blue

def scatter_matrix(ip_l2fc, inp_reads_by_loc):
    """
    returns a merged matrix between ip l2fc and input reads by loc.
    NOTE: columns MUST be in this 
    """
    plot_x = pd.read_table(
        inp_reads_by_loc, 
        index_col=0,
    )
    """
    plot_y = pd.read_table(
        ip_l2fc,
        sep='[\t\|]+',
        engine='python',
        index_col=0,
        skiprows=1,
        names=READSBYLOC_COMBINED_CSV_L2FCWITHPVAL_ENR_HEADERS_1IP1IN
    )"""
    
    plot_y = pd.read_table(
        ip_l2fc,
        index_col=0,
    )
    plot_y = split_l2fcwithpval_enr(plot_y)
    
    x = pd.merge(plot_x, plot_y, how='inner', left_index=True, right_index=True)
    return x

def plot_ip_foldchange_over_input_reads(
    ip_l2fc, inp_reads_by_loc, field_list={'CDS':'blue','3utr':'red','intron':'yellow'}
):
    
    
    df = scatter_matrix(ip_l2fc, inp_reads_by_loc)
    
    f, ax = plt.subplots(figsize=(10,10))
    for region, color in field_list.iteritems():
        ax.scatter(np.log2(df[region]+1), df["{} l2fc".format(region)], color=color, alpha=0.3)

    ax.set_title("Region-based analysis of genes enriched")
    ax.set_xlabel("Reads in SMInput (log2)")
    ax.set_ylabel("Fold Enrichment (log2)")
    plt.legend()

# ip_l2fc = '/projects/ps-yeolab3/bay001/tbos/input_norm/rbfox2_input_norm/204_RBFOX2_ReadsByLoc_combined.csv.l2fcwithpval_enr.csv'
# inp_reads_by_loc = '/projects/ps-yeolab3/bay001/tbos/input_norm/rbfox2_input_norm/RBFOX2-204-INPUT_S2_R1.unassigned.adapterTrim.round2.rmRep.rmDup.sorted.r2.bam.reads_by_loc.csv'
ip_l2fc = '/projects/ps-yeolab3/bay001/tbos/input_norm/input_norm_latest/A1_01_DAZAP1_ReadsByLoc_combined.csv.l2fcwithpval_enr.csv'
inp_reads_by_loc = '/projects/ps-yeolab3/bay001/tbos/input_norm/input_norm_latest/A1_A_IN_S28_L003_R1_001.unassigned.adapterTrim.round2.rmRep.rmDup.sorted.r2.bam.reads_by_loc.csv'
plot_figure(ip_l2fc, inp_reads_by_loc)
# scatter_matrix(ip_l2fc, inp_reads_by_loc)

ANNOTATED_BED_HEADERS = [
    'chrom','start','end','pv','fc','strand','annotation','gene'
]
REGIONS = ['noncoding_exon','3utr','5utr','intron','noncoding_intron','CDS', 'intergenic', '5utr_and_3utr']

import glob

def return_region(row):
    """
    Given a row of a inputnormed bedfile, return region
    Row must be in the same format as a line in Eric's 
    *.annotated file. 
    
    """
    try:
        if row['annotation']=='intergenic':
            return 'intergenic'
        region = row['annotation'].split('|')[0]

        return region
    except Exception as e:
        print(e)
        print(row)
        
def get_counts(wd, l2fc, l10p, suffix = '.annotated'):
    """
    Returns the number of peak counts for all regions 
    annotated by eric's pipeline.
    
    Parameters
    ----------
    wd : string
        directory where the input_norm output is kept 
        (where to look for .annotated files)
    """
    samples = {}
    
    for f in glob.glob(os.path.join(wd,'*{}'.format(suffix))):
        df = filter_input_norm_as_df(f, f.replace('{}'.format(suffix),'{}.filtered'.format(suffix)), l2fc, l10p)
        df.columns = ANNOTATED_BED_HEADERS
        basename = os.path.basename(f)

        samples[basename] = {}
        df['region'] = df.apply(return_region,axis=1)
        for key, value in df['region'].value_counts().iteritems():
            samples[basename][key] = value
        for region in REGIONS:
            if region not in samples[basename]:
                samples[basename][region] = 0
    return pd.DataFrame(samples)

s = get_counts('/projects/ps-yeolab3/bay001/tmp', 3, 3)

from itertools import izip

def plot_region_distribution(
    wd, out_file, l10p, l2fc, 
    ax=None, trim_suffix=".peaks.l2inputnormnew.bed.compressed.bed.annotated"
):
    df = get_counts(wd, l10p, l2fc)
    
    dfdiv = df/df.sum()
    cumsum_events = dfdiv.cumsum()

    num_rows = 1
    num_cols = 1
    if ax is None:
        f, ax = plt.subplots(figsize=(10,10))
        
        # ax = plt.gca()    
    legend_builder = []
    legend_labels = []
    for splice_type, color in izip(
        reversed(cumsum_events.index), 
        sns.color_palette("Set2", len(cumsum_events.index))
    ):
        if trim_suffix != None:
            cumsum_events.columns = [
                event.replace(trim_suffix, '') for event in cumsum_events.columns
            ]
            
        names = np.array(
            ["".join(item) for item in cumsum_events.columns]
        )

        sns.barplot(
            names, 
            y=cumsum_events.ix[splice_type], color=color, ax=ax
        )

        legend_builder.append(
            plt.Rectangle((0,0),.25,.25, fc=color, edgecolor = 'none')
        )
        legend_labels.append(splice_type)

    sns.despine(ax=ax, left=True)

    ax.set_ylim(0,1)

    l = ax.legend(legend_builder, 
                  legend_labels, loc=1, ncol = 1, 
                  prop={'size':12}, 
                  bbox_to_anchor=(1.4, 0.8))
    l.draw_frame(False)
    [tick.set_rotation(90) for tick in ax.get_xticklabels()]

    ax.set_ylabel("Fraction of Peaks", fontsize=14)
    [tick.set_fontsize(12) for tick in ax.get_xticklabels()]
    ax.set_title(
        "Fraction of Peaks among RBPs \
        \n-log10(p-value):{}, log2(fold-change):{}".format(
        l10p, l2fc)
    )
    f.savefig(out_file)
# plot_region_distribution('/projects/ps-yeolab3/bay001/tmp', 3, 3)

plot_region_distribution('/projects/ps-yeolab3/bay001/tmp','/projects/ps-yeolab3/bay001/tmp/distribution.png', 3, 3, )




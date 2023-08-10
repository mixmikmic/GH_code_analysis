get_ipython().magic('matplotlib inline')
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from collections import defaultdict
from tqdm import tnrange, tqdm_notebook
import numpy as np
from encode import manifest_helpers
from density import ReadDensity

MIN_EVENTS = 1000

def read_filter(fn, p=3, f=3):
    """
    Reads and filters the input normalized peak files
    """
    df = pd.read_table(fn, names=['chrom','start','stop','name','score','strand'])
    df = df[(df['name']>=p) & (df['score']>=f)]
    print(df.shape)

pickle_loc = '/projects/ps-yeolab3/encode/analysis/encode_idr_clip_analysis/'
pickles = glob.glob(os.path.join(pickle_loc, '*IDR.out.0102merged.bed.annotated.clip_formatted.bed.clip_analysis.pickle'))
pickles[:3]

manifest = '/home/bay001/projects/maps_20160420/permanent_data/ALLDATASETS.txt'
# manifest = '/home/bay001/projects/encode/analysis/conservation_analysis/uid_to_rbpinfo_from_gabe_manifest.txt'
manifest_df = pd.read_table(manifest)

def uid_to_label(uid, df=manifest_df):
    """
    From a UID, return a nice label containing 
    the RBP name and cell type
    """
    _, _, _, rbp, cell = manifest_helpers.get_clip_file_from_uid(
        manifest_df,
        uid, 
        uid_col='uID',
        rbp_col='RBP',
        clip_r1_col='CLIP',
        clip_r2_col='CLIP',
        input_col='INPUT'
    )
    if cell is None and rbp is None:
        return ''
    return '{}-{}-{}'.format(uid, rbp, cell)
uid_to_label('534')

def format_filename_to_nicename(fn):
    """
    Returns a shorter but still unique label given a pickle filename
    """
    fil1 = os.path.basename(fn)
    return fil1

def get_dataframes(p):
    """
    From a clip_analysis pickle file, return two lists of dataframes
    containing the avg phastcon scores for each peak, for each
    region. For both real and random conditions, these will be lists
    even though the len(real_dfs) will always be one.
    
    returns :
        real_dfs : {'cds':[phastcon dataframe1], 'three_prime_utr':[phastcon dataframe1], ...}
        rand_dfs : {'cds':[df1, df2, df3], 'three_prime_utr':[df1, df2, df3], ...}
        
    """
    loaded = pickle.load(open(p, 'rb'))
    
    real_dfs = defaultdict(list)
    rand_dfs = defaultdict(list)
    
    for category, df in loaded['phast_values'].groupby(level=0):
        for real_rand, new_df in df.groupby(level=1):
            if real_rand == 'real':
                real_dfs[category].append(new_df)
            elif real_rand == 'rand':
                rand_dfs[category].append(new_df)
            else:
                print("warning: labels don't match what we expect: {}".format(real_rand))
                
    return real_dfs, rand_dfs

def get_mean_phastcon(lst, v=False, min_events=MIN_EVENTS):
    """
    returns mean phastcon score for all dataframes in list using 'mean' column.
    
    lst : list[df1, df2, df3, ...]
    v : boolean
        if True, verbose
    returns :
        mean : average phastcon value over all dataframes in list or nan if it doesn't pass threshold.
    """
    summed = 0
    num_events = 0
    for l in lst: # for each dataframe 
        summed = summed + l['mean'].sum()  # add mean values to the running sum 
        num_events = num_events + l.shape[0]  # add the number of events to the running total
    if v:
        print(num_events)
    if num_events < min_events:
        return np.nan
    return summed/float(num_events)

def l2_fold_change(real, rand):
    """
    Returns the log2 fold change of real 
    (real mean phastcon)/rand (mean phastcon over random event)
    """
    return np.log2(real/rand)

merged_fold_changes = pd.DataFrame()
merged_means_real = pd.DataFrame()
merged_means_rand = pd.DataFrame()

to_exclude = ['243','298','544','553','516'] # LARP7, DKC1, BOP1

progress = tnrange(len(pickles))

for p in pickles:  # for each pickle file/RBP
    ### this block corresponds to a single line in merged dataframes, each line is a sample ###
    uid = os.path.basename(p).split('.')[0] # assumes a specific IDR filename structure... 
    label = uid_to_label(uid)
    if label != '':  # we found a submitted dataset with proper rbp/cell line annotations
        mean_real_dict = defaultdict()
        mean_rand_dict = defaultdict()
        fold_changes_dict = defaultdict()

        real, rand = get_dataframes(p) # get list of dataframes for each real/rand
        
        ### this block checks the number of events for the RBPs we want to exclude.
        if uid in to_exclude:
            print('{}, {}, '.format(uid, label)),
            get_mean_phastcon(real['all'], v=True)
        for region in rand.keys():  # for each cds, three_prime_utr, etc.

            ### get mean phastcon scores over list of dataframes in real/rand
            mean_real = get_mean_phastcon(real[region])
            mean_rand = get_mean_phastcon(rand[region])
            
                
            ### store values into dictionary 
            mean_real_dict[region] = mean_real
            mean_rand_dict[region] = mean_rand
            fold_changes_dict[region] = l2_fold_change(mean_real, mean_rand)

        ### Concatenate current line into merged
        merged_fold_changes = pd.concat(
            [merged_fold_changes, pd.DataFrame(fold_changes_dict, index=[label])], 
            axis=0
        )
        merged_means_real = pd.concat(
            [merged_means_real, pd.DataFrame(mean_real_dict, index=[label])]
        )
        merged_means_rand = pd.concat(
            [merged_means_rand, pd.DataFrame(mean_rand_dict, index=[label])]
        )
    progress.update(1)

merged_fold_changes

merged_fold_changes['sum'] = 0  # incase I need to re-run and don't want the sum to include old sums
merged_fold_changes['sum'] = merged_fold_changes.sum(axis=1)
merged_fold_changes.sort_values(by='sum', ascending=False, inplace=True)

merged_fold_changes.to_csv(
    '/home/bay001/projects/encode/analysis/conservation_analysis/analysis_from_pickle/l2_fold_change_phastcons_from_pickle.txt', 
    sep='\t'
)
merged_means_real.to_csv(
    '/home/bay001/projects/encode/analysis/conservation_analysis/analysis_from_pickle/mean_phastcons_from_real_pickle.txt', 
    sep='\t'
)
merged_means_rand.to_csv(
    '/home/bay001/projects/encode/analysis/conservation_analysis/analysis_from_pickle/mean_phastcons_from_rand_pickle.txt', 
    sep='\t'
)

img_dir = '/home/bay001/projects/encode/analysis/conservation_analysis/analysis_from_pickle/'

regions = ['all','three_prime_utrs','five_prime_utrs','cds','distintron500','proxintron500']
num_to_plot = 50

# merged_fold_changes.index = [i.replace('.clip_formatted.bed.clip_analysis.pickle','') for i in merged.index] # cleans up the labels a bit more.
def plot_all_regions(df, regions, img_dir):
    for region in regions:
        # sort by the region specified (highest first), then take first num_to_plot
        df.sort_values(by=region, ascending=False, inplace=True)
        subset = df[:num_to_plot][region] 
        subset = subset.iloc[::-1] # flip so that highest is on top
        subset.plot(
            kind='barh', 
            figsize=(15,25),
        )
        plt.title('Mean phastcon fold changes at least {} peaks: {}'.format(MIN_EVENTS, region))
        plt.tight_layout()
        plt.savefig(
            os.path.join(img_dir, '{}_mean_phastcon_l2_foldchanges.svg'.format(region))
        )
        
        plt.clf()
        plt.cla()
    return 0

plot_all_regions(merged_fold_changes, regions, img_dir)

p = '/projects/ps-yeolab3/encode/analysis/encode_idr_clip_analysis/495.01v02.IDR.out.0102merged.bed.annotated.clip_formatted.bed.clip_analysis.pickle'
loaded = pickle.load(open(p, 'rb'))

mean_real = loaded['phast_values'].ix['all'].ix['real'].ix[1]['mean'].mean()
phastsum_rand1 = loaded['phast_values'].ix['all'].ix['rand'].ix[0]['mean'].sum()
sum_rand1 = loaded['phast_values'].ix['all'].ix['rand'].ix[0].shape[0]
phastsum_rand2 = loaded['phast_values'].ix['all'].ix['rand'].ix[1]['mean'].sum()
sum_rand2 = loaded['phast_values'].ix['all'].ix['rand'].ix[1].shape[0]
phastsum_rand3 = loaded['phast_values'].ix['all'].ix['rand'].ix[2]['mean'].sum()
sum_rand3 = loaded['phast_values'].ix['all'].ix['rand'].ix[2].shape[0]

sum_all = phastsum_rand1 + phastsum_rand2 + phastsum_rand3
n_all = sum_rand1 + sum_rand2 + sum_rand3

mean_rand = sum_all/float(n_all)
print(mean_real/mean_rand)
print(np.log2((mean_real/mean_rand)))
merged_fold_changes.ix['495-PPIG-HepG2']

fig, ax = plt.subplots(figsize=(10,65))
to_plot = merged_fold_changes[['all','cds','distintron500','proxintron500','three_prime_utrs','five_prime_utrs']]
sns.heatmap(to_plot, ax=ax)
plt.savefig('/home/bay001/projects/encode/analysis/conservation_analysis/analysis_from_pickle/fold_change_phastcons_heatmap.svg')

# look at distal intron enriched conservation for splicing factors U2AF2/1, SF3B4, PRPF8
# look at 3' UTR enriched conservation for TDP43
# plot histogram of conservation scores
# convert to zscore

merged_fold_changes['all'].plot(kind='hist', bins=25, label='all', alpha=0.75)

merged_fold_changes['cds'].plot(kind='hist', bins=25, label='cds', alpha=0.75)

merged_fold_changes['distintron500'].plot(kind='hist', bins=25, label='distintron', alpha=0.75)

merged_fold_changes['proxintron500'].plot(kind='hist', bins=25, label='proxintron', alpha=0.75)

merged_fold_changes['three_prime_utrs'].plot(kind='hist', bins=25, label='3UTR', alpha=0.75)

merged_fold_changes['five_prime_utrs'].plot(kind='hist', bins=25, label='5UTR', alpha=0.75)
plt.legend()

merged_fold_changes_z = (merged_fold_changes - merged_fold_changes.mean())/merged_fold_changes.std()

fig, ax = plt.subplots(figsize=(10,65))
to_plot = merged_fold_changes_z[['all','cds','distintron500','proxintron500','three_prime_utrs','five_prime_utrs']]
sns.heatmap(to_plot, ax=ax)
plt.savefig('/home/bay001/projects/encode/analysis/conservation_analysis/analysis_from_pickle/fold_change_phastcons_heatmap_zscore.svg')

regions = ['all','three_prime_utrs','five_prime_utrs','cds','distintron500','proxintron500']
num_to_plot = 50

# merged_fold_changes.index = [i.replace('.clip_formatted.bed.clip_analysis.pickle','') for i in merged.index] # cleans up the labels a bit more.
def plot_zscores(df, regions, img_dir):
    for region in regions:
        df.sort_values(by=region, ascending=False, inplace=True)
        subset = df[:num_to_plot][region]
        subset = subset.iloc[::-1] # flip so that highest is on top
        subset.plot(
            kind='barh', 
            figsize=(15,25),
        )
        plt.title('Mean phastcon fold changes at least {} peaks: {}'.format(MIN_EVENTS, region))
        plt.tight_layout()
        plt.savefig(
            os.path.join(img_dir, '{}_mean_phastcon_l2_foldchanges_zscores.svg'.format(region))
        )
        plt.clf()
        plt.cla()
        
plot_zscores(merged_fold_changes_z, regions, img_dir)

import pybedtools

tdp43_peaks = pd.read_table(
    '/home/elvannostrand/data/clip/CLIPseq_analysis/ENCODE_FINALforpapers_20170325/IDR/340.01v02.IDR.out.0102merged.bed.annotated',
    names=['chrom','start','end','l10p','l2fc','strand','annotation','genes']
)
utr3_bed = pybedtools.BedTool('/projects/ps-yeolab3/bay001/annotations/data/regions/hg19_three_prime_utrs.bed')
phastcon_bw = '/projects/ps-yeolab/genomes/hg19/hg19.100way.phastCons.bw'

conservation_density = ReadDensity.Phastcon(phastcon_bw)
"""
def get_region(row):
    if 'intergenic' in row['annotation']:
        return 'intergenic'
    else:
        return row['annotation'].split('|')[0]

tdp43_peaks['region'] = tdp43_peaks.apply(get_region, axis=1)
tdp43_3utr_peaks = tdp43_peaks[tdp43_peaks['region']=='3utr'][['chrom','start','end','l10p','l2fc','strand']]
peaks = pybedtools.BedTool.from_dataframe(tdp43_3utr_peaks)"""

tdp_pickle = '/projects/ps-yeolab3/encode/analysis/encode_idr_clip_analysis/340.01v02.IDR.out.0102merged.bed.annotated.clip_formatted.bed.clip_analysis.pickle'
dataframe = get_dataframes(tdp_pickle)[0]['three_prime_utrs'][0].reset_index()
dataframe.head()

dataframe['pos'] = dataframe['level_3'].apply(lambda x: x.split('_')[-1])
dataframe['chrom'] = dataframe['pos'].apply(lambda x: x.split(':')[0])
dataframe['start'] = dataframe['pos'].apply(lambda x: int(x.split(':')[1].split('-')[0]))
dataframe['end'] = dataframe['pos'].apply(lambda x: int(x.split(':')[1].split('-')[1]))
dataframe = dataframe[['chrom','start','end','sum','mean']]
dataframe['strand'] = '+' # ignore strand for now
dataframe.head()

hg19chromsizes = '/projects/ps-yeolab/genomes/hg19/hg19.chrom.sizes'
peaks = pybedtools.BedTool.from_dataframe(dataframe)
flanking_regions = peaks.flank(g=hg19chromsizes, b=100).to_dataframe()

def return_avg_phastcon(row, density=conservation_density):
    lst = density.values(
        row['chrom'], row['start'], row['end'], row['strand']
    )
    return sum(lst) / float(len(lst))

flanking_regions['score'] = flanking_regions.apply(return_avg_phastcon, axis=1)

add_left_flank = pd.merge(flanking_regions, dataframe, how='right', left_on='end', right_on='start')
add_left_flank.columns = ['left_chrom','left_start','left_end','left_name','left_score','left_strand',
                         'chrom','start','end','name','score','strand']
add_right_flank = pd.merge(add_left_flank, flanking_regions, how='left', left_on='end', right_on='start')

add_right_flank['what'] = add_right_flank['left_score'] + add_right_flank['score_y'] - add_right_flank['score_x']

out_file = '/home/bay001/projects/encode/analysis/conservation_analysis/analysis_from_pickle/340_tdp43_conserved_3utr_candidates.bed'
add_right_flank.sort_values(by='what').head(30)[['chrom_x','start_x','end_x']].to_csv(out_file, sep='\t', index=False, header=False)




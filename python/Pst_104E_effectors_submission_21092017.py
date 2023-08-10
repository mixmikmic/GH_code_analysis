get_ipython().run_line_magic('matplotlib', 'inline')

import os
from Bio import SeqIO
import pandas as pd
import re
from pybedtools import BedTool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import statsmodels.sandbox.stats.multicomp
#import rcParams and make it outfit also when saving files
from matplotlib import rcParams
import scipy.stats as stats
rcParams.update({'figure.autolayout': True})

#define your input folders
#define your input folders updated for haplotigs
CLUSTER_FOLDER = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/Pst_104E_genome/gene_expression/Pst104_p_SecretomeClustering'
EFFECTORP_FOLDER = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/Pst_104E_genome/Secretomes/EffectorP'
GFF_FOLDER = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/032017_assembly'
PROTEIN_ANNO_FOLDER = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/enrichment_analysis/pa_26062017'
OUT_FOLDER = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/enrichment_analysis/lists'
OUT_FOLDER_FIG = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/enrichment_analysis/figures'
TMP_FIG_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/downstream_analysis_2017/scripts/tmp_figures'

###make a function out of the 5' and 3' prime distance
def _5_to_3_df(ref_bed_fn, query_bed_fn, io=False ):
    """Takes two bed6 filenames and returns dataframe with 5' and 3' distances."""
    from pybedtools import BedTool
    ref = BedTool(ref_bed_fn)
    query = BedTool(query_bed_fn)

    sub_3 = ref.closest(query ,io=io,  N=True, iu=True, D='a' ).to_dataframe()
    sub_3.rename(columns={12:'3_distance', 3:'query', 9:'3_target', 0:'contig'},                  inplace=True)
    sub_5 = ref.closest(query,                                io=io,N=True, id=True, D='a' ).to_dataframe()
    sub_5.rename(columns={12:'5_distance', 3:'query', 9:'5_target', 0:'contig'}, inplace=True)

    #merging them
    sub_merged = sub_3.merge(sub_5, on=['query', 'contig'])

    #needs to be fixed to take boundaries into account
    #remove genes on the edges
    sub_merged = sub_merged [((sub_merged['5_target'] != '.') & (sub_merged['3_target'] != '.') )  ]
    sub_merged['5_distance'] = sub_merged['5_distance'].apply(lambda x: np.absolute(x))
    return sub_merged

#makeing a function out of 5' and 3' distance bining
def _5_to_3_chains(_5_prime_df, _3_prime_df, max_distance=15000, label=''):
    """This function takes two dataframes one from _5_prime_ nearest neighbour and one from
    _3_prime_nearest neighbour. max_distance and label for the df columns can be added as well.
    Those should be generated with pybedtools and converted to dataframe
    without subsetting and with selecting the distance.
    Columns should be renamed as
    12:'3_distance', 3:'query', 9:'3_target', 0:'contig' for _3_prime
    and
    12:'5_distance', 3:'query', 9:'5_target', 0:'contig' for _5_prime
    The output will be new dataframe that contains the merged dataframes, the linked information, the linkage group,
    and the frequency of each linkage group = group member count.
    And a dataframe that summarizes the bin size, number within each bin and number of each bin.
    member_count_label	bin_size_label	number_of_bins_label
    """
    five_df = _5_prime_df.copy()
    three_df = _3_prime_df.copy()
    max_distance = max_distance
    
    #getting 5' and 3' distance
    distance_df = three_df.merge(five_df, on = ['query','contig'])

    #convert negative -1 from bedtools closest to nan and make values absolute
    tmp_index = distance_df[distance_df['5_target'] == '.'].index
    distance_df.loc[tmp_index, '5_distance'] = np.nan
    distance_df['5_distance'] = abs(distance_df['5_distance'])
    distance_df['5_distance'].fillna(-1, inplace = True) 
    #convert -1 from bedtools closest to nan in 3_distance
    tmp_index = distance_df[distance_df['3_target'] == '.'].index
    distance_df.loc[tmp_index, '3_distance'] = -1

    #subset the df and get the index first by distance and than by linkage (remember everything at the edges got a  -1 as distance)
    sub_distance_df = distance_df[                            ((distance_df['3_distance'] <max_distance)&(distance_df['3_distance'] > - 1))                                            | 
                            ((distance_df['5_distance'] <max_distance) &(distance_df['5_distance'] > -1))  ]

    #get the max distance of two consective genes in teh distance_df. It could be to have A-B close and C-D close but not B-C currently
    #this would not have gotten caught. #to be illustrated

    sub_distance_df['next_distance'] = abs(sub_distance_df.shift(-1)['1_x'] - sub_distance_df['2_x'])

    #now set the distance to the next gene to max_distance +1 if the next contig is different
    #from the current one
    next_contig_index = sub_distance_df[sub_distance_df.shift(-1)['contig'] != sub_distance_df['contig']].index.values

    sub_distance_df.loc[next_contig_index, 'next_distance'] = max_distance +1

    #get the index values and not the series
    sub_distance_df_index = sub_distance_df.index.values

    #transfer the next_distance of the linked once the main datframe and make everything else max_distance +1

    distance_df['next_linked_distance'] = max_distance +1

    distance_df.loc[sub_distance_df_index, 'next_linked_distance'] = sub_distance_df.next_distance


    #introduce new column 'linked' and make this 1 were the genes are linked (e.g. less than max distance apart)
    distance_df['linked'] =0
    distance_df.loc[sub_distance_df_index, 'linked']  = 1
    #get a new columns linkage_group that is 0 for now
    distance_df['linkage_group'] = 0

    #get linkage groups first filtered by consecutive index
    tmp_linkage_groups = ((distance_df[distance_df.linked == 1].linked                           != distance_df[distance_df.linked == 1].linked.index.to_series().diff().eq(1))).cumsum()    
    #this also adds together genes that are not really linked because they are on a different contig or A-B close and
    #C-D close but not B-C. We need to take care of this later on using the next_linked_distance column
    distance_df.loc[distance_df[distance_df.linked == 1].index, 'tmp_lg']  = tmp_linkage_groups
    
    #generate a new sub_distance_df that has all colmuns as the main distance df
    sub_distance_df = distance_df[distance_df.linked == 1]
    
    #the indexes are consectutive as indicated by the tmp linkage_group. 
    #now identify where linked sequences are separated by more than the max_distance. This includes intercontigs breaks.
    
    unlinked_lg_index = (sub_distance_df[(sub_distance_df.tmp_lg.shift(-1) == sub_distance_df.tmp_lg)]                                                  ['next_linked_distance'] > max_distance)
    
    #combine this remove unlinked_lg_indexs from initial linkage group by making everything Flase that is not linked.
    #this requires to 'add' the unlinked_lg_index boolean array to the consecutive boolean array using an or |
    #meaning only the Trues are transfered and this needs to be shiffted one downward (could have also done previous distance and not
    #next)
    
    tmp_linkage_groups = ((distance_df[distance_df.linked == 1].linked                           != distance_df[distance_df.linked == 1].linked.index.to_series().diff().eq(1))    | unlinked_lg_index.shift(1)).cumsum()
    
    distance_df.loc[distance_df[distance_df.linked == 1].index, 'linkage_group']  = tmp_linkage_groups
    
    distance_df = distance_df.loc[:,['contig', 'query', '3_target',                                            '3_distance', '5_target', '5_distance', 'linked', 'linkage_group']]
    #add a frequency columns to the dataframe
    distance_df['lg_freq'] = distance_df.groupby('linkage_group')['linkage_group'].transform('count')
    
    #now make a bin count dataframe
    
    #get the counts for each lg_freq == total number of genes in a bin of size lg_freq
    bins = distance_df[distance_df.linked !=0 ].groupby('lg_freq').count()
    
    #now get unlinked total number of genes ina bin size of 1
    bin_one = distance_df[distance_df.linked ==0 ].groupby('lg_freq').count().reset_index(drop= True)

    bin_one.index = [1] 
    #combine both
    all_bins = bins.append(bin_one)
    #use the index which represents the bin size
    all_bins['bin_size'] = all_bins.index
    
    all_bins = all_bins.sort_values('bin_size').reset_index(drop=True).loc[:, ['linked', 'bin_size']]

    all_bins.rename(columns={'linked': 'member_count'}, inplace=True)

    all_bins['number_of_bins'] = all_bins['member_count'] / all_bins['bin_size']

    #new_cnames = ['%s_%s' % (x,label) for x in all_bins.columns]

    #all_bins.rename(columns=dict(zip(all_bins.columns, new_cnames)), inplace=True)
    
    all_bins['label'] = label
    
    #all_bins['bin_size'] = all_bins['bin_size_' + label]
    return distance_df, all_bins

#define a function that subsets a dataframe to the inner quantil residual columnwise
def quant_cut_df(dataframe):
    nn_df = dataframe.copy()
    iqr_df_low = nn_df.apply(lambda x: x.quantile(0.25) - 1.5*(x.quantile(0.75) - x.quantile(0.25)) )
    iqr_df_low.name ='low'
    iqr_df_high = nn_df.apply(lambda x: x.quantile(0.75) + 1.5*(x.quantile(0.75) - x.quantile(0.25)) )
    iqr_df_high.name = 'high'

    iqr_df = pd.concat([iqr_df_low, iqr_df_high], axis=1).T

    iqr_nn_df = nn_df.apply(lambda x: x[(x > iqr_df.loc['low', x.name]) & (x  < iqr_df.loc['high', x.name])], axis=0)
    return iqr_nn_df 

if not os.path.exists(OUT_FOLDER_FIG):
    os.mkdir(OUT_FOLDER_FIG)

#get some empty list to fill them with data
genome = 'Pst_104E_v12_'
p_effector_list = []
h_effector_list = []
p_effector_seq_list = []
h_effector_seq_list = []
p_effectorp_list = []
h_effectorp_list = []
p_effectorp_seq_list = []
h_effectorp_seq_list = []

#define what you want to take clusters are from the expression analysis
#in the manuscript we use the expression data from cluster 2,3,8
clusters = [ 2,3,8]
clusters_files = [os.path.join(CLUSTER_FOLDER, x) for x in os.listdir(CLUSTER_FOLDER)                 if x.startswith('Cluster') and x.endswith('_DEs.fasta') and                  any(str(y) in x for y in clusters) ] #fixed to check if any of the clusters are
                                    #in the file header
#we also read in all the effectorp headers generated by Jana Speerschneider
effectorp_files = [os.path.join(EFFECTORP_FOLDER, x) for x in os.listdir(EFFECTORP_FOLDER)                  if x.endswith('effectors.fasta') and x.startswith(genome)]

#get all the sequence names into a list from the fasta headers 
for file in clusters_files:
    fh = open(file, 'r')
    for seq in SeqIO.parse(fh, 'fasta'):
        if 'hcontig' in seq.id:
            h_effector_list.append(seq.id)
            h_effector_seq_list.append(seq)
        if 'pcontig' in seq.id:
            p_effector_list.append(seq.id)
            p_effector_seq_list.append(seq)
    fh.close()

for file in effectorp_files:
    fh = open(file, 'r')
    for seq in SeqIO.parse(fh, 'fasta'):
        if 'hcontig' in seq.id and seq.id not in h_effector_list:
            h_effector_list.append(seq.id)
            h_effector_seq_list.append(seq)
        if 'pcontig' in seq.id and seq.id not in p_effector_list:
            p_effector_list.append(seq.id)
            p_effector_seq_list.append(seq)
    fh.close()
    
for file in effectorp_files:
    fh = open(file, 'r')
    for seq in SeqIO.parse(fh, 'fasta'):
        if 'hcontig' in seq.id:
            h_effectorp_list.append(seq.id)
            h_effectorp_seq_list.append(seq)
        if 'pcontig' in seq.id:
            p_effectorp_list.append(seq.id)
            p_effectorp_seq_list.append(seq)
    fh.close()

#define the effector file name
#these were defined previously by the effectorp output and the outer union of this and
#the expression cluster 2, 3, 8
p_effector_file = os.path.join(OUT_FOLDER, genome + 'p_effector.list')
p_effectorp_file = os.path.join(OUT_FOLDER, genome + 'p_effectorp.list')

#now get BUSCO list in order to remove the two BUSCOs from the effector candidate list
p_busco_file = [os.path.join(PROTEIN_ANNO_FOLDER, x) for x in os.listdir(PROTEIN_ANNO_FOLDER) if x.startswith(genome+'p_ctg') and 'busco' in x][0]
p_busco_list = pd.read_csv(p_busco_file, header=None, sep='\t')[0].tolist()

#write out effectors without BUSCOs
effector_busco_overlap = [x for x in p_effector_list if x in p_busco_list]
print(effector_busco_overlap)
#remove those two from the effector list and update the effectors
#one is a peptidase and the other an ER cargo protein both are fairly conserved
print("This is the number of effector candidates before removal: %i" % len(p_effector_list))
updated_effector_seq_list = []
for x in effector_busco_overlap:
    p_effector_list.remove(x)
for seq in p_effector_seq_list:
    if seq.id not in effector_busco_overlap:
        updated_effector_seq_list.append(seq)
print("This is the number of effector candidates after removal of BUSCOs: %i" % len(p_effector_list))

#write out effectorps without BUSCOs
effectorp_busco_overlap = [x for x in p_effectorp_list if x in p_busco_list]
print(effectorp_busco_overlap)
#remove those two from the effector list and update the effectors
#one is a peptidase and the other an ER cargo protein both are fairly conserved
print("This is the number of effector candidates before removal: %i" % len(p_effectorp_list))
updated_effectorp_seq_list = []
for x in effectorp_busco_overlap:
    p_effectorp_list.remove(x)
for seq in p_effectorp_seq_list:
    if seq.id not in effectorp_busco_overlap:
        updated_effectorp_seq_list.append(seq)
print("This is the number of effectorp candidates after removal of BUSCOs: %i" % len(p_effectorp_list))

#subset the gff files as well and write those out
p_gff_file = [os.path.join(GFF_FOLDER, x) for x in os.listdir(GFF_FOLDER)                 if x.startswith(genome+'p_ctg') and x.endswith('anno.gff3') ][0]

#get repeat gff files
p_repeat_gff_fn = [os.path.join(GFF_FOLDER, x) for x in os.listdir(GFF_FOLDER)                 if x.startswith(genome+'p_ctg') and x.endswith('REPET.gff') ][0]

#get repeat gff files
p_repeat_superfamily_gff_fn = [os.path.join(OUT_FOLDER, x) for x in os.listdir(OUT_FOLDER)                 if x.startswith(genome+'p_ctg') and x.endswith('REPET.sorted.superfamily.gff') ][0]

#gff header 
gff_header = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']

#now read in the gff file. This can be used in future versions for subsetting to make
#it all more flexible
p_gff_df = pd.read_csv(p_gff_file, header = None, sep='\t', names= gff_header)
p_gff_df['ID'] = p_gff_df.attributes.str.extract(r'ID=([^;]*);', expand=False)
p_gff_df.sort_values(by=['seqid', 'start'], inplace = True)

#now sort REPET gff and write out again
p_repeat_gff_df = pd.read_csv(p_repeat_gff_fn, header=None, sep='\t', names=gff_header, comment='#')
p_repeat_gff_fn = os.path.join(OUT_FOLDER,p_repeat_gff_fn.split('/')[-1] )
p_repeat_gff_df.sort_values(by=['seqid', 'start']).to_csv(p_repeat_gff_fn, header=None, index=None, sep='\t')

#now define the filenames
p_effector_bed_fn = p_effector_file.replace('.list', '.gene.bed')
p_effector_gff_fn = p_effector_file.replace('.list', '.gene.gff3')    
p_noeffector_bed_fn = p_effector_file.replace('p_effector.list', 'p_noeffector.gene.bed')
p_noeffector_gff_fn = p_effector_file.replace('p_effector.list', 'p_noeffector.gene.gff3')    
p_noeffector_list_fn = p_effector_file.replace('p_effector.list', 'p_noeffector.list')
#effectorp
p_effectorp_bed_fn = p_effectorp_file.replace('.list', '.gene.bed')
p_effectorp_gff_fn = p_effectorp_file.replace('.list', '.gene.gff3')    
p_noeffectorp_bed_fn = p_effectorp_file.replace('p_effectorp.list', 'p_noeffectorp.gene.bed')
p_noeffectorp_gff_fn = p_effectorp_file.replace('p_effectorp.list', 'p_noeffectorp.gene.gff3')    
p_noeffectorp_list_fn = p_effectorp_file.replace('p_effectorp.list', 'p_noeffectorp.list')
#get BUSCO filenames
p_busco_list_fn = p_effector_file.replace('effector.list', 'busco.list')
p_busco_gff_fn = p_effector_file.replace('effector.list', 'busco.gene.gff3')
p_busco_bed_fn = p_effector_file.replace('effector.list', 'busco.gene.bed')

#get no buscos afiles
p_non_busco_list_fn = p_effector_file.replace('effector.list', 'non_busco.list')
p_non_busco_gff_fn = p_effector_file.replace('effector.list', 'non_busco.gene.gff3')
p_non_busco_bed_fn = p_effector_file.replace('effector.list', 'non_busco.gene.bed')
#get file names for no buscos and no effectors
p_noeffector_nobusco_bed_fn = p_effector_file.replace('effector.list', 'no_busco_no_effector.gene.bed')    
p_noeffectorp_nobusco_bed_fn = p_effectorp_file.replace('effectorp.list', 'no_busco_no_effectorp.gene.bed')
p_gene_bed_fn = p_effector_file.replace('effector.list', 'all.gene.bed') 

#here subset the repeat superfamily GFF to filder out smaller repeats if wanted
tmp_REPET = pd.read_csv(p_repeat_superfamily_gff_fn, header=None, sep='\t', names=gff_header)
tmp_REPET['distance'] = tmp_REPET.end - tmp_REPET.start
tmp_REPET = tmp_REPET[tmp_REPET.source != 'Pst79p_anno_REPET_SSRs'].copy()
min_length = 0
tmp_fn = p_repeat_superfamily_gff_fn.replace('superfamily.gff', 'g%i_superfamily.bed' % min_length)
tmp_REPET[tmp_REPET.distance > min_length].iloc[:,[0,3,4,8,7,6]]    .to_csv(tmp_fn, header=None, sep='\t', index=None)
#read this in as repeat_df
p_repeats_bed = BedTool(tmp_fn)

#read in files and generate bedtool objects. Issue might be that same variable are used for the bed files
#while those are sometimes gff and sometimes bed. Needs changing
p_effector_bed = BedTool(p_effector_bed_fn)
p_noeffector_bed = BedTool(p_noeffector_bed_fn)
p_busco_bed = BedTool(p_busco_bed_fn)
p_non_busco_bed = BedTool(p_non_busco_bed_fn)
p_gene_bed = BedTool(p_gene_bed_fn)
p_no_eb_bed = BedTool(p_noeffector_nobusco_bed_fn)


#get closest repeat and make a df out of it
p_closest_rep_to_eff = p_effector_bed.closest(p_repeats_bed, d=True)
p_closest_rep_to_eff_df = p_closest_rep_to_eff.to_dataframe()

#bed closest header when using gff as input files
bed_repeat_closest_header = [x +'_gene' for x in gff_header] + [x +'_repeat' for x in gff_header] + ['distance']

#ignore the warnings because of all the bedtools to datafarme warning
import warnings
warnings.filterwarnings('ignore')

#get the distances of TEs to different gene categories
min_dist_TE_to_e = p_effector_bed.closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_b = p_busco_bed.closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_noe = p_noeffector_bed.closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_nob = p_non_busco_bed.closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_g = p_gene_bed.closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_noeb = p_no_eb_bed.closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values

_, p_TE_g_vs_e = scipy.stats.ranksums(min_dist_TE_to_g, min_dist_TE_to_e)
_, p_TE_g_vs_b = scipy.stats.ranksums(min_dist_TE_to_g, min_dist_TE_to_b)
_, p_TE_b_vs_e = scipy.stats.ranksums(min_dist_TE_to_b, min_dist_TE_to_e)
statsmodels.sandbox.stats.multicomp.multipletests([p_TE_g_vs_e, p_TE_g_vs_b, p_TE_b_vs_e],    alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)

#capture the corrected errors
corrected_errors = statsmodels.sandbox.stats.multicomp.multipletests([p_TE_g_vs_e, p_TE_g_vs_b, p_TE_b_vs_e],    alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)

TE_dist_df_no_io = pd.concat([pd.Series(x) for x in [min_dist_TE_to_g, min_dist_TE_to_b,            min_dist_TE_to_e, min_dist_TE_to_nob, min_dist_TE_to_noe, min_dist_TE_to_noeb]], axis=1)

TE_dist_df_no_io.rename(columns=dict(zip(TE_dist_df_no_io.columns,        ['All genes', 'BUSCOs', 'Candidate effectors', 'No BUSCOs', 'No effectors', 'No BUSCOs no effectors'])), inplace=True) 

TE_dist_df_no_io.describe()

from matplotlib.font_manager import FontProperties
df = TE_dist_df_no_io.iloc[:,[0,1,2]].melt()
sns.set_style("white")
#do a boxplot and swarmplot on the same data
fig, ax = plt.subplots(figsize=(10, 10))
#ax.set_xscale("log")
sns.violinplot(y="value", x="variable", data=df, cut=0, 
          whis=np.inf, palette=sns.color_palette('colorblind'))
plt.setp(ax.artists, alpha=.01)
#sns.swarmplot(y="value", x="variable", data=df,
 #           size=2, color=".3", linewidth=0)
#set the labels
plt.ylim(0, 35000)

#add the title
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')
font.set_size(20)

title = 'Minimum distance to Transposable Elements'
plt.title(title.replace('to', 'to\n'), fontproperties=font)

#set font size for labels and such
fs = 20

#fontsize labels
font0 = FontProperties()
font_axis = font0.copy()
font_axis.set_size(20)

plt.xlabel("Gene categories", fontproperties = font_axis)
plt.ylabel('Distance in bp', fontproperties = font_axis)


#add the stats to it as well with numbers and lines
ax.text(0.15, 32000, 'p~%.2E' % corrected_errors[1][0], color='k', fontsize=fs)
ax.plot([-0.1, 1.1], [31500, 31500],color ='k' ,lw=1)
ax.text(1.25, 29500, 'p~%.2E' % corrected_errors[1][1], color='k', fontsize=fs)
ax.plot([0.1, 2.1], [29000, 29000],color ='k' ,lw=1)
ax.text(1.25, 27000, 'p~%.2E' % corrected_errors[1][2], color='k', fontsize=fs)
ax.plot([0.9, 2.1], [26500, 26500],color ='k' ,lw=1)
sns.despine(offset=10, trim=True)

#fontsize of ticks
ax.tick_params(labelsize=fs)

#save the file as well
out_file_name = "_".join(title.split(' '))
fig.savefig(os.path.join(OUT_FOLDER_FIG, out_file_name+'_v3.png'), dpi=600)

sub_set = len(p_busco_bed)

#get the distances of TEs to different gene categories
min_dist_TE_to_e = p_effector_bed.random_subset(sub_set).closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_b = p_busco_bed.random_subset(sub_set).closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_noe = p_noeffector_bed.random_subset(sub_set).closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_nob = p_non_busco_bed.random_subset(sub_set).closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_g = p_gene_bed.random_subset(sub_set).closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
min_dist_TE_to_noeb = p_no_eb_bed.random_subset(sub_set).closest(p_repeats_bed,d=True,t='last')                    .to_dataframe().iloc[:,12].dropna().values
TE_dist_df = pd.concat([pd.Series(x) for x in [min_dist_TE_to_g, min_dist_TE_to_b,            min_dist_TE_to_e, min_dist_TE_to_nob, min_dist_TE_to_noe, min_dist_TE_to_noeb]], axis=1)
TE_dist_df.rename(columns=dict(zip(TE_dist_df.columns,        ['All genes', 'BUSCOs', 'Effectors', 'No BUSCOs', 'No effectors', 'No BUSCOs no effectors'])), inplace=True) 

df = TE_dist_df.iloc[:,[0,1,2]].melt()
#do a boxplot and swarmplot on the same data
f, ax = plt.subplots(figsize=(15, 10))
#ax.set_xscale("log")
sns.violinplot(x="value", y="variable", data=df, cut=0,
          whis=np.inf)
plt.setp(ax.artists, alpha=.01)
#sns.swarmplot(x="value", y="variable", data=df,
 #             size=2, color=".3", linewidth=0)
#plt.xlim(0, 120000)
plt.ylabel("Comparisons")
plt.xlabel('Distance in bp')
#ax.text(118000, 0.5, '*\n*\n*', color='k')
#ax.plot([117000, 117000],[-0.1, 0.8], color ='k' )
#ax.text(118000, 2.5, '*\n*\n*', color='k')
#ax.plot([117000, 117000],[1.9, 2.8], color ='k' )
sns.despine(offset=10, trim=True)

TE_dist_df.describe()

_, p_TE_g_vs_e = scipy.stats.ranksums(min_dist_TE_to_g, min_dist_TE_to_e)
_, p_TE_g_vs_b = scipy.stats.ranksums(min_dist_TE_to_g, min_dist_TE_to_b)
_, p_TE_b_vs_e = scipy.stats.ranksums(min_dist_TE_to_b, min_dist_TE_to_e)
statsmodels.sandbox.stats.multicomp.multipletests([p_TE_g_vs_e, p_TE_g_vs_b, p_TE_b_vs_e],    alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)

sample_size = [len(p_busco_bed), len(p_effector_bed)]

sample_size.sort()

#set the size of the subset here
sub_set = sample_size[0]

#read in bed files and subset
p_effector_bed = BedTool(p_effector_bed_fn)
p_allgene_bed = BedTool(p_gene_bed_fn)
p_busco_bed = BedTool(p_busco_bed_fn)
p_allall_rand_sub = p_allgene_bed.random_subset(sub_set)
p_effector_bed_rand_sub = p_effector_bed.random_subset(sub_set)
p_busco_bed_rand_sub = p_busco_bed.random_subset(sub_set)

#get the distances with nearest gene effectors vs effectors
p_eself = p_effector_bed_rand_sub.closest(p_effector_bed_rand_sub, d=True,  N=True).to_dataframe().iloc[:,12]
p_eself = p_eself[p_eself > -1].copy()
p_eself.name = 'Candidate effectors'
p_eall = p_effector_bed_rand_sub.closest(p_allall_rand_sub, d=True,  N=True).to_dataframe().iloc[:,12]
p_eall= p_eall[p_eall > -1].copy()
print(p_eself.describe())
p_eself.plot(kind='box')

#get the distances with nearest gene all vs all subsampled
p_allall = p_allall_rand_sub.closest(p_allall_rand_sub, d=True, N=True).to_dataframe().iloc[:,12]
p_allall = p_allall[p_allall > -1]
p_allall.name = 'All_genes'
print(p_allall.describe())
p_allall.plot(kind='box')

#now with buscos
p_bself = p_busco_bed_rand_sub.closest(p_busco_bed_rand_sub, d=True,  N=True).to_dataframe().iloc[:,12]
p_bself = p_bself[p_bself > -1]
p_bself.name = 'BUSCO'
print(p_bself.describe())
p_bself.plot(kind='box')

#non_effectors
p_noeffector_bed= BedTool(p_noeffector_bed_fn)
p_noeffector_rand_sub = p_noeffector_bed.random_subset(sub_set)
p_neself = p_noeffector_rand_sub.closest(p_noeffector_rand_sub, d=True,  N=True).to_dataframe().iloc[:,12]
p_neself = p_neself[p_neself > -1]
p_neself.name = 'No_effectors'
print(p_neself.describe())
p_neself.plot(kind='box')

print('effectors distance to closest Busco')
p_effector_bed_rand_sub.closest(p_busco_bed_rand_sub, d=True,t='last', io=True).to_dataframe().boxplot(column=12)
effector_busco_c_df = p_effector_bed_rand_sub.closest(p_busco_bed_rand_sub, d=True,t='last', io=True).to_dataframe().iloc[:,12]
effector_busco_c_df = effector_busco_c_df[effector_busco_c_df>-1]
effector_busco_c_df.name = "Closest Busco to effector"
effector_busco_c_df.describe()

print('Closest effector to busco')
p_busco_bed_rand_sub.closest(p_effector_bed_rand_sub, d=True,t='last', io=True).to_dataframe().boxplot(column=12)
busco_effector_c_df = p_busco_bed_rand_sub.closest(p_effector_bed_rand_sub, d=True,t='last', io=True).to_dataframe().iloc[:,12]
busco_effector_c_df = busco_effector_c_df[busco_effector_c_df>-1]
busco_effector_c_df.name = 'Closest effector to busco'
busco_effector_c_df.describe()

print('effectors distance to closest rand')
p_effector_bed_rand_sub.closest(p_allall_rand_sub, d=True,t='last', io=True).to_dataframe().boxplot(column=12)
effector_randsubset_c_df = p_effector_bed_rand_sub.closest(p_allall_rand_sub, d=True,t='last', io=True).to_dataframe().iloc[:,12]
effector_randsubset_c_df = effector_randsubset_c_df[effector_randsubset_c_df>-1]
effector_randsubset_c_df.name =  "Closest randsubset to effector"
effector_randsubset_c_df.describe()

print('busco distance to closest rand')
p_busco_bed_rand_sub.closest(p_allall_rand_sub, d=True,t='last', io=True).to_dataframe().boxplot(column=12)
busco_rand_sub_c_df = p_busco_bed_rand_sub.closest(p_allall_rand_sub, d=True,t='last', io=True).to_dataframe().iloc[:,12]
busco_rand_sub_c_df = busco_rand_sub_c_df[busco_rand_sub_c_df>-1]
busco_rand_sub_c_df.name = "Closest rand sub to BUSCO"
busco_rand_sub_c_df.describe()

print('rand to closest busco')
p_allall_rand_sub.closest(p_busco_bed_rand_sub, d=True,t='last', io=True).to_dataframe().boxplot(column=12)
rand_sub_busco_c_df = p_allall_rand_sub.closest(p_busco_bed_rand_sub, d=True,t='last', io=True).to_dataframe().iloc[:,12]
rand_sub_busco_c_df = rand_sub_busco_c_df[rand_sub_busco_c_df>-1]
rand_sub_busco_c_df.name = 'Closest BUSCO to rand subset'
rand_sub_busco_c_df.describe()

print('Rand to closest effector')
p_allall_rand_sub.closest(p_effector_bed_rand_sub, d=True,t='last', io=True).to_dataframe().boxplot(column=12)
rand_subset_effector_c_df = p_allall_rand_sub.closest(p_effector_bed_rand_sub, d=True,t='last', io=True).to_dataframe().iloc[:,12]
rand_subset_effector_c_df = rand_subset_effector_c_df[rand_subset_effector_c_df>-1]
rand_subset_effector_c_df.name ='Closest effector to rand subset'
rand_subset_effector_c_df.describe()

#all against others both ways to see if effectors are closer to buscos compared to random subset
all_vs_others_c_df =  pd.concat([busco_effector_c_df,rand_subset_effector_c_df,effector_busco_c_df,rand_sub_busco_c_df,               effector_randsubset_c_df, busco_rand_sub_c_df ], axis=1)

fig, ax = plt.subplots(figsize=(15,6))
sns.violinplot(data=all_vs_others_c_df, cut=0)
plt.xticks(rotation=-45)

#plot the violine plots for distances once the iqr is present only
all_vs_others_iqr_df = quant_cut_df(all_vs_others_c_df)
fig, ax = plt.subplots(figsize=(15,6))
sns.violinplot(data=all_vs_others_iqr_df, cut=0)
plt.xticks(rotation=-45)
#same as above only normalized for inner quantile residual 

all_vs_others_c_df.describe()

#do some stats on it look first into ranksum test
_, error_e = scipy.stats.ranksums(all_vs_others_c_df['Closest effector to busco'], all_vs_others_c_df['Closest effector to rand subset'])

_, error_b = scipy.stats.ranksums(all_vs_others_c_df['Closest Busco to effector'], all_vs_others_c_df['Closest BUSCO to rand subset'])

#rename figures for paper
new_c_names = ['Candidate effectors \n-> BUSCOs', 'Candidate effectors \n-> All genes subset',               'BUSCOs \n-> Candidate effectors',               'BUSCOs \n-> All genes subset' ]

all_vs_others_iqr_df.rename(columns=dict(zip(all_vs_others_iqr_df.columns[:4],new_c_names)), inplace=True)

all_vs_others_iqr_melt = all_vs_others_iqr_df.melt()


df = all_vs_others_iqr_melt    [all_vs_others_iqr_melt.variable.isin(all_vs_others_iqr_df.iloc[:,[0,1,2,3]].columns)].copy()
#do a boxplot and swarmplot on the same data
fig, ax = plt.subplots(figsize=(10, 10))
#ax.set_xscale("log")
sns.violinplot(y="value", x="variable", data=df, cut=0,
          whis=np.inf, palette=sns.color_palette('colorblind'))
plt.setp(ax.artists, alpha=.01)
#sns.swarmplot(y="value", x="variable", data=df,
 #             size=2, color=".3", linewidth=0)
#set the labels
plt.ylim(0, 125000)
#plt.tight_layout()

#add the title
font0 = FontProperties()
font_axis = font0.copy()
font_axis.set_size(20)

#set font size for labels and such
fs = 20

#add the title and legends
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')
font.set_size(20)

title = 'Minimum distance to genes in given group'
plt.title(title.replace('to', 'to\n'), fontproperties=font)
plt.xlabel("Comparisons" , fontproperties=font_axis)
plt.ylabel('Distance in bp', fontproperties=font_axis)

#add the stats to it as well with numbers and lines
ax.text(0.15, 119000, 'p~%.2E'% error_e, color='k', fontsize=fs)
ax.plot([-0.1, 1.1], [118000, 118000],color ='k' ,lw=1)
ax.text(2.25, 99000, 'p~%.2E'% error_b, color='k', fontsize=fs)
ax.plot([1.9, 3.1], [98000, 98000],color ='k' ,lw=1)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=-45)
#save the file as well
#fontsize of ticks
ax.tick_params(labelsize=fs)

out_file_name = "_".join(title.split(' '))
fig.savefig(os.path.join(OUT_FOLDER_FIG, out_file_name+'_v3.png'), dpi=600)

_, p_ab = scipy.stats.ranksums(p_allall, p_bself)
_, p_ae = scipy.stats.ranksums(p_allall, p_eself)
_, p_be = scipy.stats.ranksums(p_bself, p_eself)

print(statsmodels.sandbox.stats.multicomp.multipletests([p_ab,p_ae, p_be],    alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False))

#now make a nearest neightbour dataframe
nn_df = pd.concat([p_allall, p_bself, p_eself], names=['All genes', 'BUSCOs', 'Effectors'], axis=1)

new_c_names = ['All genes subset', "BUSCOs", 'Candidate effectors']
nn_df.rename(columns=dict(zip(nn_df.columns, new_c_names)),inplace=True)

iqr_df_low = nn_df.apply(lambda x: x.quantile(0.25) - 1.5*(x.quantile(0.75) - x.quantile(0.25)) )
iqr_df_low.name ='low'
iqr_df_high = nn_df.apply(lambda x: x.quantile(0.75) + 1.5*(x.quantile(0.75) - x.quantile(0.25)) )
iqr_df_high.name = 'high'

iqr_df = pd.concat([iqr_df_low, iqr_df_high], axis=1).T

iqr_nn_df = nn_df.apply(lambda x: x[(x > iqr_df.loc['low', x.name]) & (x  < iqr_df.loc['high', x.name])], axis=0)
plt.title('Violine plot of nearest neighbour in the same category')
sns.violinplot(data=iqr_nn_df  , palette=sns.color_palette('colorblind'), cut=0)

_, p_ab = scipy.stats.ranksums(p_allall, p_bself)
_, p_ae = scipy.stats.ranksums(p_allall, p_eself)
_, p_be = scipy.stats.ranksums(p_bself, p_eself)

print(statsmodels.sandbox.stats.multicomp.multipletests([p_ab,p_ae, p_be],    alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False))

corrected_errors_2 = statsmodels.sandbox.stats.multicomp.multipletests([p_ab,p_ae, p_be],    alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)

from matplotlib.font_manager import FontProperties
#chanage df here
df = iqr_nn_df

#set style
sns.set_style("white")
#do a boxplot and swarmplot on the same data
fig, ax = plt.subplots(figsize=(10, 10))
#ax.set_xscale("log")
sns.violinplot( data=df, cut=0, 
          whis=np.inf, palette=sns.color_palette('colorblind'))
plt.setp(ax.artists, alpha=.01)
#sns.swarmplot( data=df,
            #size=2, color=".3", linewidth=0)
#set the labels
plt.ylim(0, 105000)

#add the title
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')
font.set_size(20)

#set font size for labels and such
fs = 20
#add the title
font0 = FontProperties()
font_axis = font0.copy()
font_axis.set_size(20)

title = 'Minimum distance to genes in the same group'
plt.title(title.replace('to', 'to\n'), fontproperties=font)
plt.xlabel("Gene categories", fontproperties=font_axis)
plt.ylabel('Distance in bp', fontproperties=font_axis)
#add the stats to it as well with numbers and lines
ax.text(0.15, 89000, 'p~%.2E'% corrected_errors_2[1][0], color='k',fontsize=fs)
ax.plot([-0.1, 1.1], [88000, 88000],color ='k' ,lw=1)
ax.text(1.25, 81000, 'p~%.2E'% corrected_errors_2[1][1], color='k',fontsize=fs)
ax.plot([0.1, 2.1], [80000, 80000],color ='k' ,lw=1)
ax.text(1.25, 70000, 'p~%.2E'% corrected_errors_2[1][2], color='k',fontsize=fs)
ax.plot([0.9, 2.1], [69000, 69000],color ='k' ,lw=1)
sns.despine(offset=10, trim=True)
sns.set_style("white")
#save the file as well
out_file_name = "_".join(title.split(' '))
#fontsize of ticks
ax.tick_params(labelsize=fs)
fig.savefig(os.path.join(OUT_FOLDER_FIG, out_file_name+'_v3.png'), dpi=600)

fig, ax = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(22, 8))

##set font size for labels and such
fs = 20

limit = 500000000000
df = _5_to_3_df(p_gene_bed_fn, p_gene_bed_fn)
df2 = _5_to_3_df( p_effector_bed_fn, p_gene_bed_fn)
df3= _5_to_3_df(p_busco_bed_fn,p_gene_bed_fn )
df = df[(df['3_distance'] < limit) & (df['5_distance'] < limit)].copy()
df2 = df2[(df2['3_distance'] < limit) & (df2['5_distance'] < limit)].copy()
df3 = df3[(df2['3_distance'] < limit) & (df3['5_distance'] < limit)].copy()
hb = ax[0].hexbin(df['3_distance'], df['5_distance'], xscale='log', yscale='log',                  gridsize=(50,50)        ,  cmap='magma_r',extent=[0, 6, 0, 6])
ax[0].text(1.5, 1.5, 'n=%i'% len(df), color='k',fontsize=fs)
hb1 = ax[2].hexbin(df2['3_distance'], df2['5_distance'],xscale='log', yscale='log',                   gridsize=(50,50)        ,  cmap='magma_r',extent=[0, 6, 0, 6])
ax[2].text(1.5, 1.5, 'n=%i'% len(df2), color='k',fontsize=fs)
hb2 = ax[1].hexbin(df3['3_distance'], df3['5_distance'], xscale='log', yscale='log',                   gridsize=(50,50)        ,  cmap='magma_r',extent=[0, 6, 0, 6])
ax[1].text(1.5, 1.5, 'n=%i'% len(df3), color='k',fontsize=fs)
#axis font properties
font0 = FontProperties()
font_axis = font0.copy()
font_axis.set_size(20)



cb = fig.colorbar(hb, ax=ax[0])
cb1 = fig.colorbar(hb1, ax=ax[2])
cb2 = fig.colorbar(hb2, ax=ax[1])

#change the label size of colorbars
cbfs=20
cb.ax.tick_params(labelsize=cbfs)
cb1.ax.tick_params(labelsize=cbfs)
cb2.ax.tick_params(labelsize=cbfs)

# Set common labels
ax[1].set_xlabel("3' flanking distance", fontsize=fs+4)
ax[0].set_ylabel("5' flanking distance", fontsize=fs+4)

#set colorbar labels
sns.despine(offset=10, trim=True)
#cb.set_label('All gene')
#cb1.set_label('Effectors')
#cb2.set_label('BUSCOS')

#set subtitels
fs_title = 24
ax[1].set_title('BUSCOs', fontsize = fs_title )
ax[0].set_title('All genes', fontsize = fs_title )
ax[2].set_title('Candidate effectors', fontsize = fs_title )

#set main title
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')
font.set_size(20)
title = 'Connectivity hexplots for indicated gene groups'
#plt.suptitle(title, fontproperties=font)

#fontsize of ticks
ax[0].tick_params(labelsize=fs)
ax[1].tick_params(labelsize=fs)
ax[2].tick_params(labelsize=fs)

out_file_name = "_".join(title.split(' '))
fig.savefig(os.path.join(OUT_FOLDER_FIG, out_file_name+'_v3.png'), dpi=600)

max_distance = 12000

#getting 5' and 3' distance for random subset of effectors
e_to_e_sub_3 = p_effector_bed_rand_sub.closest( p_effector_bed_rand_sub ,  N=True, iu=True, D='a' ).to_dataframe()
e_to_e_sub_3.rename(columns={12:'3_distance', 3:'query', 9:'3_target', 0:'contig'}, inplace=True)
e_to_e_sub_5 = p_effector_bed_rand_sub.closest( p_effector_bed_rand_sub.fn ,  N=True, id=True, D='a' ).to_dataframe()
e_to_e_sub_5.rename(columns={12:'5_distance', 3:'query', 9:'5_target', 0:'contig'}, inplace=True)
distance_effector_df, bin_effector_df = _5_to_3_chains(e_to_e_sub_5,e_to_e_sub_3,max_distance=max_distance, label='Effectors') 

#now for busco
p_busco_bed_rand_sub_3 = p_busco_bed_rand_sub.closest( p_busco_bed_rand_sub.fn ,  N=True, iu=True, D='a' ).to_dataframe()
p_busco_bed_rand_sub_3.rename(columns={12:'3_distance', 3:'query', 9:'3_target', 0:'contig'}, inplace=True)
p_busco_bed_rand_sub_5 = p_busco_bed_rand_sub.closest( p_busco_bed_rand_sub.fn ,  N=True, id=True, D='a' ).to_dataframe()
p_busco_bed_rand_sub_5.rename(columns={12:'5_distance', 3:'query', 9:'5_target', 0:'contig'}, inplace=True)
distance_busco_df, bin_busco_df = _5_to_3_chains(p_busco_bed_rand_sub_5,p_busco_bed_rand_sub_3,max_distance=max_distance, label='BUSCOs')

#now for random subest of all genes
all_all_rand_3 = p_allall_rand_sub.closest( p_allall_rand_sub ,  N=True, iu=True, D='a' ).to_dataframe()
all_all_rand_3.rename(columns={12:'3_distance', 3:'query', 9:'3_target', 0:'contig'}, inplace=True)
all_all_rand_5 = p_allall_rand_sub.closest( p_allall_rand_sub.fn ,  N=True, id=True, D='a' ).to_dataframe()
all_all_rand_5.rename(columns={12:'5_distance', 3:'query', 9:'5_target', 0:'contig'}, inplace=True)
distance_all_gene_df, bin_all_gene_df = _5_to_3_chains(all_all_rand_5,all_all_rand_3,max_distance=max_distance, label = 'All genes')

overall_bining_df = pd.concat([bin_effector_df,bin_busco_df,bin_all_gene_df ])

sns.color_palette('colorblind', 3)

#set yourself up for the plots
overall_bining_df['member_count_log'] = np.log2(overall_bining_df.member_count)

conversion_dict = dict(zip(overall_bining_df.label.unique(), range(1,4)))
conversion_dict_color = dict(zip(overall_bining_df.label.unique(), sns.color_palette('colorblind', 3)))

#get the data labels and the color labels
overall_bining_df['number_labels'] = overall_bining_df.label.apply(lambda x: conversion_dict[x])
overall_bining_df['color_labels'] = overall_bining_df.label.apply(lambda x: conversion_dict_color[x])

#set the overall sns style

sns.set_style("white")

#start the figure
fig, ax = plt.subplots(1,1, figsize=(10,4))
title = "Gene clusters with a maximum inter-gene distance of %i" % max_distance
#fill the figure with a scatter plot
ax.scatter(overall_bining_df.bin_size, overall_bining_df.number_labels, s=overall_bining_df.member_count,            color =overall_bining_df['color_labels'] )
#set font size 
fs=15

#add the labels
for label, x, y in zip(overall_bining_df.member_count,overall_bining_df.bin_size, overall_bining_df.number_labels ):
    plt.annotate(label, xy =(x+0.2,y),fontsize=fs)
plt.xlabel('Cluster size', fontsize=fs)
ax.set_yticks(range(1,4))
ax.set_yticklabels(overall_bining_df.label.unique())
ax.tick_params(labelsize=fs)

#add the title
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')
font.set_size(16)
plt.title(title.replace(str(max_distance), '%ikb'%(int(max_distance)/1000))          , fontproperties=font, )
ax.title.set_position([0.5,1.05])


sns.despine(offset=10)
out_file_name = "_".join(title.split(' '))
fig.savefig(os.path.join(OUT_FOLDER_FIG, out_file_name+'_v2.png'), dpi=600)

exp = [889,382,129,24,20,0,0]
obs_e =[793,378,171,64,24,6,7]
obs_b = [658,410,177,140,40,12,7]

_, p_ae = scipy.stats.chisquare(exp, obs_e)
_, p_ab = scipy.stats.chisquare(exp, obs_b)
_, p_eb = scipy.stats.chisquare(obs_e, obs_b)
statsmodels.sandbox.stats.multicomp.multipletests([p_ae, p_ab, p_eb],alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)

#Define the PATH
BASE_AA_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12'
BASE_A_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/032017_assembly'
BLAST_RESULT_PATH = os.path.join(BASE_AA_PATH,'allele_analysis' )
ALLELE_PATH =os.path.join(BASE_AA_PATH ,'allele_analysis/alleles_proteinortho_graph516')
ALLELE_QC_PATH = os.path.join(BASE_AA_PATH, 'allele_analysis',                               'no_alleles_proteinortho_graph516_QC_Qcov80_PctID70_evalue01')
LIST_PATH = os.path.join(BASE_AA_PATH, 'enrichment_analysis', 'lists')
POST_ALLELE_PATH = os.path.join(BASE_AA_PATH, 'post_allele_analysis')
OUT_PATH = os.path.join(POST_ALLELE_PATH, 'proteinortho_graph516_QC_Qcov80_PctID70_evalue01')
if not os.path.exists(POST_ALLELE_PATH):
    os.mkdir(POST_ALLELE_PATH)
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

#get all the alleles in as they are not filtered by QCov and PctID but simply taken straight from the 
# proteinortho
allele_header = ['p_gene', 'h_gene']
a_overlap_df = pd.read_csv(os.path.join(ALLELE_PATH,                     'Pst_104E_v12_p_ctg.h_contig_overlap.alleles'), sep='\t', header = None,                           names=allele_header)
a_overlap_df['Linkage'] = 'h_contig_overlap'
a_no_roverlap_df = pd.read_csv(os.path.join(ALLELE_PATH,                     'Pst_104E_v12_p_ctg.no_respective_h_contig_overlap.alleles'), sep='\t', header = None,                           names=allele_header)
a_no_roverlap_df['Linkage'] = 'no_r_overlap'
a_no_soverlap_df = pd.read_csv(os.path.join(ALLELE_PATH,                     'Pst_104E_v12_p_ctg.no_specific_h_contig_overlap.alleles'), sep='\t', header = None,                           names=allele_header)
a_no_soverlap_df['Linkage'] = 'no_s_overlap'
allele_df = pd.concat([a_overlap_df, a_no_roverlap_df,a_no_soverlap_df ], axis=0)

#get the blast dataframe and the QC dataframe
allele_blast_df = pd.read_csv(os.path.join(BLAST_RESULT_PATH, 'Pst_104E_v12_p_ctg.Pst_104E_v12_h_ctg.0.001.blastp.outfmt6.allele_analysis'), sep='\t')
allele_QC_df = pd.read_csv(os.path.join(ALLELE_QC_PATH, 'Pst_104E_v12_ph_ctg.no_alleles_QC.Qcov80.PctID70.df'), sep='\t')
allele_QC_df = allele_QC_df[~((allele_QC_df.Query.isin(allele_df.p_gene))|(allele_QC_df.Query.isin(allele_df.h_gene)))]
#quick check if all the numbers of genes add up
len(allele_df.p_gene.unique())+ len(allele_df.h_gene.unique())+len(allele_QC_df.Query.unique()) == 30249

#make a new column linking stuff together
allele_blast_df['match'] = allele_blast_df.Query + allele_blast_df.Target
allele_df['match'] = allele_df.p_gene + allele_df.h_gene

allele_hits_only_df = allele_blast_df[(allele_blast_df.match.isin(allele_df.match))].copy()

#get the PctID fo the allele blast hits primary contigs onto haplotig gene models
effecter_bpctid = allele_hits_only_df[(allele_hits_only_df.Query.isin(p_effector_list))]['PctID'].tolist()

noeffecter_bpctid = allele_hits_only_df[~(allele_hits_only_df.Query.isin(p_effector_list))]['PctID'].tolist()

busco_bpctid = allele_hits_only_df[(allele_hits_only_df.Query.isin(p_busco_list))]['PctID'].tolist()

non_busco_list = pd.read_csv(p_non_busco_list_fn, sep='\t', header =None)[0].tolist()
non_be_list = pd.read_csv(p_noeffectorp_nobusco_bed_fn, sep='\t', header =None)[3].str.replace('TU', 'model').tolist()

non_busco_bpctid = allele_hits_only_df[(allele_hits_only_df.Query.isin(non_busco_list))]['PctID'].tolist()
non_be_list_bpctid = allele_hits_only_df[(allele_hits_only_df.Query.isin(non_be_list))]['PctID'].tolist()

#bind all the blast percentage IDs together
bpctid_df = pd.concat([pd.Series(effecter_bpctid),pd.Series(noeffecter_bpctid)                          , pd.Series(busco_bpctid), pd.Series(non_busco_bpctid)                      ,pd.Series(non_be_list_bpctid)], axis=1)
bpctid_df.rename(columns={0: 'effector', 1:'no_effector', 2: 'buscos', 3: 'non_buscos',                         4:'no_be'}, inplace=True)
bpctid_df.describe()

bpctid_melt_df = bpctid_df.melt()

#a quick plot of the distributions of blast pct id of allele pairs in different categories
f, ax = plt.subplots(figsize=(15, 10))
#ax.set_xscale("log")
sns.violinplot(x="value", y="variable", data=bpctid_melt_df, cut=0,
          whis=np.inf, saturation=0.5)
plt.setp(ax.artists, alpha=.01)
#sns.swarmplot(x="value", y="variable", data=bpctid_melt_df,
  #            size=2, color=".3", linewidth=0)

#quick non-parametric test comparing the different pctid distributions in different categories
stats.mstats.kruskalwallis([float(x) for x in bpctid_df.effector.dropna()]                           ,[float(x) for x in bpctid_df.no_be.dropna()],                          [float(x) for x in bpctid_df.buscos.dropna()])

W_PATH = os.path.join(BASE_AA_PATH, 'window_analysis')
if not os.path.exists(W_PATH):
    os.mkdir(W_PATH)

#make some windows beds
window_fn_dict = {}
window_bed_dict = {}
window_fn_dict['w30kb_s1kb'] = os.path.join(W_PATH, 'Pst_104E_v12_p_ctg.w30.s1.bed')
window_fn_dict['w100kb_s1kb'] = os.path.join(W_PATH, 'Pst_104E_v12_p_ctg.w100.s1.bed')
window_fn_dict['w30kb'] = os.path.join(W_PATH, 'Pst_104E_v12_p_ctg.w30.bed')
window_fn_dict['w100kb'] = os.path.join(W_PATH, 'Pst_104E_v12_p_ctg.w100.bed')
genome_size_f_fn = os.path.join(W_PATH, 'Pst_104E_v12_p_ctg.genome_file')
contig_fn = os.path.join(GFF_FOLDER,'Pst_104E_v12_p_ctg.fa' )
os.chdir(W_PATH)
#now make the window files
get_ipython().system("bedtools makewindows -g {genome_size_f_fn} -w 100000 -s 1000 > {window_fn_dict['w100kb_s1kb']}")
get_ipython().system("bedtools makewindows -g {genome_size_f_fn} -w 30000 -s 1000 > {window_fn_dict['w30kb_s1kb']}")
get_ipython().system("bedtools makewindows -g {genome_size_f_fn} -w 100000 > {window_fn_dict['w100kb']}")
get_ipython().system("bedtools makewindows -g {genome_size_f_fn} -w 30000 > {window_fn_dict['w30kb']}")

#new make a bedtools window dataframe
for key, value in window_fn_dict.items() :
    window_bed_dict[key] = BedTool(value)

#now make an AT bed df dict and save it out
window_AT_dict = {}
for key, value in window_bed_dict.items():
    tmp_df = value.nucleotide_content(fi=contig_fn).to_dataframe().iloc[1:,[0,1,2, 3]]
    tmp_df.rename(columns={'name':'%AT'}, inplace=True)
    tmp_fn = window_fn_dict[key].replace('bed', 'AT.bed')
    tmp_df.to_csv(tmp_fn, header=None, sep='\t', index=None)
    tmp_fn = window_fn_dict[key].replace('bed', 'AT.circabed')
    tmp_df.to_csv(tmp_fn, sep='\t', index=None)
    window_AT_dict[key] = tmp_df
    tmp_df = ''

feature_fn_dict = {}
feature_fn_dict['genes'] = os.path.join(LIST_PATH, 'Pst_104E_v12_p_all.gene.bed' )
feature_fn_dict['effector'] = os.path.join(LIST_PATH, 'Pst_104E_v12_p_effector.gene.bed' )
feature_fn_dict['busco'] = os.path.join(LIST_PATH, 'Pst_104E_v12_p_busco.gene.bed' )
feature_fn_dict['haustoria'] = os.path.join(LIST_PATH, 'Pst_104E_v12_cluster_8.gene.bed' )
feature_fn_dict['no_be'] = os.path.join(LIST_PATH, 'Pst_104E_v12_p_non_busco_non_effector.gene.bed')
feature_fn_dict['no_effector'] =os.path.join(LIST_PATH, 'Pst_104E_v12_p_noeffector.gene.bed')
feature_fn_dict['no_busco'] = os.path.join(LIST_PATH, 'Pst_104E_v12_p_non_busco.gene.bed')
feature_fn_dict['TE_g400'] = os.path.join(LIST_PATH,                                'Pst_104E_v12_p_ctg.REPET.sorted.g400_superfamily.bed' )
feature_fn_dict['TE_g1000'] = os.path.join(LIST_PATH,                                'Pst_104E_v12_p_ctg.REPET.sorted.g1000_superfamily.bed' )
feature_fn_dict['TE_g0'] = os.path.join(LIST_PATH,                                'Pst_104E_v12_p_ctg.REPET.sorted.superfamily.bed' )

feature_bed_dict = {}
for key, value in feature_fn_dict.items():
    feature_bed_dict[key] = BedTool(value)

feature_overlap_df_dict = {}
for wkey, wbed in window_bed_dict.items():
    for fkey, fbed in feature_bed_dict.items():
        tmp_df = wbed.coverage(fbed, F=0.1).to_dataframe().iloc[:,[0,1,2,3,6]]
        tmp_df.rename(columns={'name': 'overlap_count', 'thickStart': 'overlap_fraction'}, inplace=True)
        tmp_fn = feature_fn_dict[fkey].replace('bed', '%s.overlap.bed' % wkey)
        feature_overlap_df_dict[tmp_fn.split('/')[-1]] = tmp_df
        tmp_df.to_csv(tmp_fn, sep='\t', header=None, index=None)
        tmp_fn = feature_fn_dict[fkey].replace('bed', '%s.overlap.circabed' % wkey)
        tmp_df.to_csv(tmp_fn, sep='\t', index=None)
        
#tmp_bed.intersect(feature_bed_dict[key], c=True, F=0.1).to_dataframe() 

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_effector.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,                    feature_overlap_df_dict['Pst_104E_v12_p_ctg.REPET.sorted.superfamily.w30kb_s1kb.overlap.bed'].overlap_fraction)

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_all.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,                    feature_overlap_df_dict['Pst_104E_v12_p_ctg.REPET.sorted.superfamily.w30kb_s1kb.overlap.bed'].overlap_fraction)

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_busco.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,                    feature_overlap_df_dict['Pst_104E_v12_p_ctg.REPET.sorted.superfamily.w30kb_s1kb.overlap.bed'].overlap_fraction)

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_non_busco_non_effector.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,                    feature_overlap_df_dict['Pst_104E_v12_p_ctg.REPET.sorted.superfamily.w30kb_s1kb.overlap.bed'].overlap_fraction)

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_effector.gene.w30kb_s1kb.overlap.bed'].overlap_count,                    feature_overlap_df_dict['Pst_104E_v12_p_ctg.REPET.sorted.superfamily.w30kb_s1kb.overlap.bed'].overlap_count)

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_all.gene.w30kb_s1kb.overlap.bed'].overlap_count,                    feature_overlap_df_dict['Pst_104E_v12_p_ctg.REPET.sorted.superfamily.w30kb_s1kb.overlap.bed'].overlap_count)

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_busco.gene.w30kb_s1kb.overlap.bed'].overlap_count,                    feature_overlap_df_dict['Pst_104E_v12_p_ctg.REPET.sorted.superfamily.w30kb_s1kb.overlap.bed'].overlap_count)

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_non_busco_non_effector.gene.w30kb_s1kb.overlap.bed'].overlap_count,                    feature_overlap_df_dict['Pst_104E_v12_p_ctg.REPET.sorted.superfamily.w30kb_s1kb.overlap.bed'].overlap_count)

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_non_busco_non_effector.gene.w30kb_s1kb.overlap.bed'].overlap_fraction ,[1-np.float(x) for x in window_AT_dict['w30kb_s1kb']['%AT']])

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_busco.gene.w30kb_s1kb.overlap.bed'].overlap_fraction ,[1-np.float(x) for x in window_AT_dict['w30kb_s1kb']['%AT']])

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_effector.gene.w30kb_s1kb.overlap.bed'].overlap_fraction ,[1-np.float(x) for x in window_AT_dict['w30kb_s1kb']['%AT']])

stats.stats.spearmanr(feature_overlap_df_dict['Pst_104E_v12_p_all.gene.w30kb_s1kb.overlap.bed'].overlap_fraction ,[1-np.float(x) for x in window_AT_dict['w30kb_s1kb']['%AT']])

#this idea is based on the Genome Sequence of S. sclerotiorum in GBE to test if effectors
#are in gene spares region in the sense of non-effectors vs. effectors and not
#neccessarily nearest neighours
#this is consistent with the idea that candidate effectors cluster with each other and
#this clustering drives the gene hexbin plot
#seems to be only visible for certain window sizes.
#too big a window might dilute the signal
x_now = [np.log10(x/y) for x,y in   zip(feature_overlap_df_dict['Pst_104E_v12_p_effector.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,       feature_overlap_df_dict['Pst_104E_v12_p_noeffector.gene.w30kb_s1kb.overlap.bed'].overlap_fraction)]
y_now=[np.log10(x) for x in feature_overlap_df_dict['Pst_104E_v12_p_all.gene.w30kb_s1kb.overlap.bed'].overlap_fraction]  
fig, ax = plt.subplots(figsize=(12,12))
plt.scatter(x=x_now,y=y_now, marker ='+', color='k', alpha=0.3)
print(stats.stats.spearmanr(x_now, y_now,nan_policy='omit'))

#this idea is based on the Genome Sequence of S. sclerotiorum in GBE to test if effectors
#are in gene spares region in the sense of non-effectors vs. effectors and not
#neccessarily nearest neighours
x_now = [np.log10(x/y) for x,y in   zip(feature_overlap_df_dict['Pst_104E_v12_p_effector.gene.w100kb_s1kb.overlap.bed'].overlap_fraction,       feature_overlap_df_dict['Pst_104E_v12_p_noeffector.gene.w100kb_s1kb.overlap.bed'].overlap_fraction)]
y_now=[np.log10(x) for x in feature_overlap_df_dict['Pst_104E_v12_p_all.gene.w100kb_s1kb.overlap.bed'].overlap_fraction]  
fig, ax = plt.subplots(figsize=(12,12))
plt.scatter(x=x_now,y=y_now, marker ='+', color='k', alpha=0.3)
print(stats.stats.spearmanr(x_now, y_now,nan_policy='omit'))

#busco follow the same pattern again this might be linked to the clustering of BUSCOs with
#each other
x_now = [np.log10(x/y) for x,y in   zip(feature_overlap_df_dict['Pst_104E_v12_p_busco.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,       feature_overlap_df_dict['Pst_104E_v12_p_non_busco.gene.w30kb_s1kb.overlap.bed'].overlap_fraction)]
y_now=[np.log10(x) for x in feature_overlap_df_dict['Pst_104E_v12_p_all.gene.w30kb_s1kb.overlap.bed'].overlap_fraction]  
fig, ax = plt.subplots(figsize=(12,12))
plt.scatter(x=x_now,y=y_now, marker ='+', color='k', alpha=0.3)
#sns.regplot(x=np.array(x_now),y=np.array(y_now), marker ='+', color='k')
print(stats.stats.spearmanr(x_now, y_now, nan_policy='omit'))

x_now = [np.log10(x/y) for x,y in   zip(feature_overlap_df_dict['Pst_104E_v12_p_busco.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,       feature_overlap_df_dict['Pst_104E_v12_p_non_busco.gene.w30kb_s1kb.overlap.bed'].overlap_fraction)]
y_now=[np.log10(x) for x in feature_overlap_df_dict['Pst_104E_v12_p_all.gene.w30kb_s1kb.overlap.bed'].overlap_fraction]  
fig, ax = plt.subplots(figsize=(12,12))
plt.scatter(x=x_now,y=y_now, marker ='+', color='k')
#sns.regplot(x=np.array(x_now),y=np.array(y_now), marker ='+', color='k')
print(stats.stats.spearmanr(x_now, y_now, nan_policy='omit'))

#unclear for this part of the effect would need to look more into it
x_now=[np.log10((x+y)/z) for x,y,z in   zip(feature_overlap_df_dict['Pst_104E_v12_p_effector.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,       feature_overlap_df_dict['Pst_104E_v12_p_busco.gene.w30kb_s1kb.overlap.bed'].overlap_fraction,       feature_overlap_df_dict['Pst_104E_v12_p_non_busco_non_effector.gene.w30kb_s1kb.overlap.bed'].overlap_fraction)]
y_now=[np.log10(x) for x in feature_overlap_df_dict['Pst_104E_v12_p_all.gene.w30kb_s1kb.overlap.bed'].overlap_fraction]
fig, ax = plt.subplots(figsize=(12,12))
plt.scatter(x=x_now,y=y_now, marker ='+', color='k', alpha =0.1)
#sns.regplot(x=np.array(x_now),y=np.array(y_now), marker ='+', color='k')
print(stats.stats.spearmanr(x_now, y_now, nan_policy='omit'))




get_ipython().system(' export PATH=$PATH:/projects/ps-yeolab/software/bin')

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import random
import os
import pybedtools
import csv
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec
from gscripts.general import region_helpers
from collections import defaultdict
from itertools import izip
from gscripts.general import dataviz
import seaborn as sns
pd.options.display.max_colwidth = 500

sep_score_threshold = 0.5
p_val_threshold = 0.05

def gencode_to_ensembl(gene_list):
    """
    Takes gencode formatted ID, returns ensembl formatted ID (basically chops off .version)
    """
    for gene in gene_list:
        yield gene.split(".")[0]

# returns the sep_score and q-values of events
def open_omniviewer(fn, exclude = 'sk', remove_complex = True):
    """
    Reads in a file of events, returns sep_score table and q_val table
    Renames and indexes on event name:gene, notes:splice_type, and event position:location
    Args:
        fn (string) : file from omniviewer
        exclude (string) : since the table includes both values for 
            "sk(ip)" and "in(clude)", we just need one for each event, 
            as they will be mirrored
        remove_complex (boolean) : removes complex events
    Returns:
        all_sepscores (pd.DataFrame) : df of sepscores
        all_qvalues (pd.DataFrame) : df of qvalues
    """
    #Think about adding in / flitering on q-values
    df = pd.read_table(fn).sort(columns=['event name', 'path'])
    if(remove_complex):
        df = df[(df.path != exclude) &  ~(df.notes.str.contains("complex"))]
    else:
        df = df[(df.path != exclude)]
    # df = df[df.notes != 'complex']
    df.index = pd.MultiIndex.from_tuples(list(izip(df['event name'], df['notes'], df['event position'])), 
                                         names=["gene", "splice_type", "location"])
   
    num_sepscores = np.arange((len(df.columns) - 5) / 7) # get column positions
    sepscores = np.array(num_sepscores) * 7 + 6 # get column positions
    qvalues = sepscores + 1
    return df.iloc[:,sepscores].sort_index(axis=1), df.iloc[:,qvalues].sort_index(axis=1)

date='4-30-17'
all_data = '/home/bay001/projects/tbos_splicing_20160203/permanent_data/all_tbos_data_keep.csv' # downloaded from exon.ucsc.edu
gencode_db = "/projects/ps-yeolab/genomes/hg19/gencode_v19/gencode.v19.annotation.gtf.db" # gffutils-cli create --disable-infer <gfffile> --output <outfile>'
mart_bedfile = "/home/bay001/projects/tbos_splicing_20160203/permanent_data/hg19_genes.bed" # biomart file of all genes as bed
# alt_cassette_exons_bedfile = "/home/bay001/projects/tbos_20160203/permanent_data/htahg19_alt_cass.bed"
alt_cassette_exons_bedfile = "/projects/ps-yeolab/genomes/hg19/hta2_microarray/chtahg19_alt_cass.bed"
img_dir = '/home/bay001/projects/tbos_splicing_20160203/scripts/images/{}/'.format(date)
event_centric_dir = '/projects/ps-yeolab3/bay001/tbos/annotations/event_centric/{}'.format(date)
exon_centric_dir = '/projects/ps-yeolab3/bay001/tbos/annotations/exon_centric/{}'.format(date)

# gleans annotation info from gencode_db (gffutils-cli create --disable-infer <gfffile> --output <outfile>)
gene_id_to_name = region_helpers.gene_id_to_name(gencode_db)
name_to_gene_id = {value: key for key, value in gene_id_to_name.items()}
ensembl_to_gencode = {key.split(".")[0]: key for key in gene_id_to_name.keys()}

# Remove columns we won't use
# Rename columns based on expt conditions
all_events_sepscore, all_events_qvalue = open_omniviewer(all_data)
events = [all_events_sepscore, all_events_qvalue]
cols = ['sepscore','q-value']
index_names = ["rbp","expression"]
tb_names = [
    ['apobec4','kd'],['apobec4','over'],
    ['boll','over'],
    ['ccnl1','kd'],['ccnl1','over'],
    ['cstf1','kd'],['cstf1','over'],
    ['dazap1','kd'], ['dazap1','over'],
    ['fubp1','kd'],['fubp1','over'],
    ['luc7l2','kd'],['luc7l2','over'],
    ['rbfox2','kd'],['rbfox2','over'],
    ['rbm11','kd'],['rbm11','over'],
    ['tial1','kd'],['tial1','over'],
    ['trnau1ap','kd'],['trnau1ap','over']
]
for i in range(0,len(cols)):
    samples_to_remove = ['hta_sepscore_yeo_aug15_one_apobec4_kd_{}'.format(cols[i]),
                         'hta_sepscore_yeo_aug15_one_apobec4_over_{}'.format(cols[i]),
                         'hta_sepscore_yeo_aug15_all_rbfox2_kd6_{}'.format(cols[i]),
                         'hta_sepscore_yeo_aug15_all_trnau1ap_over2_{}'.format(cols[i])]
    events[i].drop(samples_to_remove, inplace=True, axis=1)
    events[i].columns = pd.MultiIndex.from_tuples(tb_names,names=index_names)
all_events_sepscore.head()
all_events_sepscore.reset_index().to_csv('/home/bay001/projects/tbos_splicing_20160203/data/temp.tsv',sep='\t')

# reindex hta file for easier parsing
# fill nans with 0 or 1
all_events_sepscore = all_events_sepscore.fillna(0.)
all_events_qvalue = all_events_qvalue.fillna(1.)

all_events_sepscore = all_events_sepscore.groupby(level=all_events_sepscore.index.names)  
all_events_qvalue = all_events_qvalue.groupby(level=all_events_qvalue.index.names)  

all_events_sepscore = all_events_sepscore.last()
all_events_qvalue = all_events_qvalue.last()

all_events_sepscore.head()

all_events_sepscore.reset_index()[all_events_sepscore.reset_index()['splice_type'].str.contains('alt_cassette')]

# overlap/rename annotation data with ensembl data 
genes = pybedtools.BedTool(mart_bedfile)

# coordinate:event dictionary
splice_annotation_dict_full = dict(zip(all_events_sepscore.index.get_level_values(level="location"), 
                                  all_events_sepscore.index.get_level_values(level="gene")))
# coordinate:splicetype dictionary
splice_type_dict_full = dict(zip(all_events_sepscore.index.get_level_values(level="location"), 
                            map(lambda x: x.strip(), all_events_sepscore.index.get_level_values(level="splice_type"))))

# Should think about how to better assign these, possibly based off the assigned gene name... 
intervals = []
for interval in all_events_sepscore.index.get_level_values(level="location"):
    chrom, loc, strand = interval.split(":")
    start, stop = loc.split("-")
    intervals.append(pybedtools.interval_constructor([chrom, start, stop, interval, "0", strand]))

# Locations of each event
locations = pybedtools.BedTool(intervals)

# Locations of each event and its associated name
locations_and_names = locations.intersect(genes, loj=True, s=True)

# Location:list(gene_names) dictionary
splice_location_to_gene_id_full = defaultdict(list)
for location in locations_and_names:
    splice_location_to_gene_id_full[location.name].append(location[9]) # location[9] is the gene name

# reindex hta file for easier parsing (TIDY DATA)
all_events_sepscore.index = all_events_sepscore.index.get_level_values(level="location")
all_events_sepscore = all_events_sepscore.T
all_events_sepscore = all_events_sepscore.stack()
all_events_sepscore = pd.DataFrame(all_events_sepscore, columns=["sep_score"])

all_events_qvalue.index = all_events_qvalue.index.get_level_values(level="location")
all_events_qvalue = all_events_qvalue.T
all_events_qvalue = all_events_qvalue.stack()
all_events_qvalue = pd.DataFrame(all_events_qvalue, columns=["q_value"])

all_events_sepscore = all_events_sepscore.join(all_events_qvalue)
all_events_sepscore.head()

# overlap/rename annotation data with ensembl data 
# TAKE LOCATION AND ASSIGN GENE ID TO IT
# make sure there is sufficient matches of gene to gencode id
accurate_location_to_gene_id_full = {}
c = 0
e = 0
for splice_location in all_events_sepscore.index.get_level_values(level="location"): # for each location
    annotated_gene_name = splice_annotation_dict_full[splice_location].split("_")[0] # get the gene name (eg. PYROXD2)
    
    """
    For each gene in this position, if it matches the original hta2 annotation, c+=1, else if no gene
    is reported, e+=1. In other words, iterate over all genes associated with this position, return the
    most likely to be the one that was originally annotated.
    """
    for gene_id in splice_location_to_gene_id_full[splice_location]: # for each located at this position
        if gene_id == ".": # no name
            gene_id = "error"
            e = e + 1
        elif gene_id_to_name[gene_id] == annotated_gene_name:
            c = c + 1
            break
    """
    Report most accurate gene ID for each location:
    - if gene_id_to_name['ENSG00000204859.7'] == ZBTB48, and 
    if the splice_annotation_dict_full[chr1:6640130-6640600:+] == ZBTB48_(65), 
    accurate_location_to_gene_id_full[chr1:6640130-6640600:+] == ENSG00000204859.7
    """
    accurate_location_to_gene_id_full[splice_location] = gene_id

print("match:{} error:{} percentage: {}".format(c,e,(e/float(e+c))))

# TAKE GENE NAME AND VARIOUS ANNOTATION TYPES/THINGS AND ADDING IT ONTO THE DATAFRAME

all_events_sepscore['gene_id'] = [accurate_location_to_gene_id_full[splice_location] for splice_location in all_events_sepscore.index.get_level_values(level="location")]
all_events_sepscore['old_splice_annotation'] = [splice_annotation_dict_full[splice_location] for splice_location in all_events_sepscore.index.get_level_values(level="location")]
all_events_sepscore['splice_type'] = [splice_type_dict_full[splice_location] for splice_location in all_events_sepscore.index.get_level_values(level="location")]

"""
Don't really know why we need this, but keeping it in for future reference.
"""
c = 0 # count of events
e = 0 # count of errors

gene_names = []
for splice_location in all_events_sepscore['gene_id']:
    try:
        gene_names.append(gene_id_to_name[splice_location])
        c = c + 1
    except:
        gene_names.append("error")
        e = e + 1
        
all_events_sepscore['gene_name'] = gene_names

print(c, e)

alt_cassette_events_full = all_events_sepscore[all_events_sepscore.splice_type == "alt_cassette"]
alt_cassette_exons = pd.read_table(alt_cassette_exons_bedfile, header=None,
              names=["chrom_internal", "start_internal", "stop_internal", "event_id","score","strand_internal"])
alt_cassette_events_full = pd.merge(left=alt_cassette_events_full.reset_index(), 
                               right=alt_cassette_exons, 
                               left_on="old_splice_annotation", 
                               right_on="event_id").set_index(['rbp', 'expression', 'location']).sort_index()
dfx = alt_cassette_events_full
print("all unique cassette exons", alt_cassette_events_full.shape) # without removing overlapping regions (if two alternatively spliced exons share the same flanking exons).
alt_cassette_events_full = alt_cassette_events_full.groupby(level=['rbp', 'expression', 'location']).first()
dfy = alt_cassette_events_full.groupby(level=['rbp', 'expression', 'location']).first()
print("all unique cassette events", alt_cassette_events_full.shape) # after removing overlapping events.
dfy_s = dfy[(dfy.sep_score.abs() > sep_score_threshold) & (dfy.q_value < p_val_threshold)]
print("all unique significant cassette events", dfy_s.shape)
dfz = all_events_sepscore[(all_events_sepscore.sep_score.abs() > sep_score_threshold) & (all_events_sepscore.q_value < p_val_threshold)]
print("significant events (all types) : ",dfz.shape)

# make sure events are not being duplicated for some reason.
X = dfy.reset_index()
X[X.duplicated(['rbp','expression','location'])]

interesting_rbps = ['apobec4','boll','ccnl1','cstf1','dazap1','fubp1','luc7l2','rbfox2','rbm11','tial1','trnau1ap']

# items = alt_cassette_events_full.groupby(level=["rbp", "expression"]).count().sep_score
with dataviz.Figure(os.path.join(img_dir, "total_number_of_events.svg"), figsize=(12,5), tight_layout=True) as fig:
    ax = fig.add_subplot(1,2,1)
    items = dfz.ix[interesting_rbps].groupby(level=["rbp", "expression"]).count().sep_score
    # names = np.array([" ".join([rbp_name_dict[item[0]], cell_type_dict[item[1]]]) for item in items.index])
    # names = np.array(['apobec4-kd','apobec4-over','rbfox2-1','rbfox2-6','rbfox2-kd','rbfox2-kd1','rbfox2-over'])
    names = np.array([' '.join(item) for item in items.index])
    sns.barplot(names,
                items.values,
                ci=None,
                palette="Paired",
                hline=.1,
                ax=ax,
                x_order=names
                )
    [tick.set_rotation(90) for tick in ax.get_xticklabels()]
    ax.set_title("Number of Alt Splicing Events", fontsize=18)
    ax.set_ylabel("Number of Significantly\nChanging Events", fontsize=14)
    [tick.set_fontsize(14) for tick in ax.get_xticklabels()]

    ax = fig.add_subplot(1,2,2)
    # items = significant_alt_cassette_events.loc[['rbfox2','apobec4'], :].groupby(level=["rbp", "expression"]).count().sep_score
    # names = np.array([" ".join([rbp_name_dict[item[0]], cell_type_dict[item[1]]]) for item in items.index])
    items = dfy_s.ix[interesting_rbps].groupby(level=["rbp", "expression"]).count().sep_score
    names = np.array([' '.join(item) for item in items.index])
    sns.barplot(names,
                items.values,
                ci=None,
                palette="Paired",
                hline=.1, 
                ax=ax,
                x_order=names)
    [tick.set_rotation(90) for tick in ax.get_xticklabels()]
    [tick.set_fontsize(12) for tick in ax.get_xticklabels()]

    ax.set_title("Number of Alt Cassette Events")
    #ax.set_ylabel("Number of Events")

num_events = {}

# for name, df in significant_events.loc[['rbfox2','apobec4'], :].groupby(level=['rbp', 'expression']):
for name, df in dfz.ix[interesting_rbps].groupby(level=['rbp','expression']):
    num_events[name] = df.groupby("splice_type").count().sep_score
    
num_events = pd.DataFrame(num_events)
num_events.columns = pd.MultiIndex.from_tuples(num_events.columns)
num_events = num_events.fillna(0)

fraction_events = num_events / num_events.sum() 
cumsum_events = fraction_events.cumsum()
# um_events.sum()
print(cumsum_events.filter(like="fubp1"))
num_events

percent_events = num_events/num_events.sum()

# percent_events = percent_events.unstack
percent_events.columns = ['%s%s' % (a, ' - %s' % b if b else '') for a, b in percent_events.columns]
# sns.clustermap(percent_events, linewidths=0, xticklabels=percent_events.columns, yticklabels=percent_events.index)
percent_events.to_csv("percentages_sepscore_{}.tsv".format(sep_score_threshold),sep="\t",quote=False,header=True,index=True)
percent_over = percent_events.filter(like="over")
percent_under = percent_events.filter(like="kd")
# percent_change = percent_over - percent_under
# sns.clustermap(percent_change, linewidths=0, xticklabels=percent_events.columns, yticklabels=percent_events.index)
percent_over

percent_under = percent_under.rename(columns={col: col.split(' - ')[0] for col in percent_under.columns})
del percent_over['boll - over'] # BOLL has no KD expression for comparison, so we need to delete it.
percent_over = percent_over.rename(columns={col: col.split(' - ')[0] for col in percent_over.columns})

percent_change = percent_over - percent_under
percent_change.to_csv("percent_change_{}.tsv".format(sep_score_threshold),sep="\t")
percent_change

num_rows = 1
num_cols = 1
with dataviz.Figure(os.path.join(img_dir, "total_fractional_composition_all_samples.svg"), figsize=(10, 4)) as fig:
    ax = fig.add_subplot(num_rows,num_cols,1)
    
    legend_builder = []
    legend_labels = []
    for splice_type, color in izip(reversed(cumsum_events.index), sns.color_palette("Set2", len(cumsum_events.index))):
        names = np.array([" ".join(item) for item in cumsum_events.columns])

        sns.barplot(names, 
                    y=cumsum_events.ix[splice_type], color=color, ax=ax)
        
        legend_builder.append(plt.Rectangle((0,0),.25,.25, fc=color, edgecolor = 'none'))
        legend_labels.append(splice_type)

    sns.despine(ax=ax, left=True)
    
   

    l = ax.legend(legend_builder, 
                  legend_labels, loc=1, ncol = 1, 
                  prop={'size':10}, 
                  bbox_to_anchor=(1.5, 1))
    l.draw_frame(False)
    [tick.set_rotation(90) for tick in ax.get_xticklabels()]
    #Need to change to percent
    ax.set_ylabel("Fraction of Events", fontsize=14)
    [tick.set_fontsize(12) for tick in ax.get_xticklabels()]
    ax.set_title("Fraction of events among {}".format(interesting_rbps))

strong = all_events_sepscore[(all_events_sepscore.sep_score.abs() > 2) & (all_events_sepscore.q_value < p_val_threshold)].reset_index()
medium = all_events_sepscore[(all_events_sepscore.sep_score.abs() <= 2) & (all_events_sepscore.sep_score.abs() > 1) & (all_events_sepscore.q_value < p_val_threshold)].reset_index()
weak = all_events_sepscore[(all_events_sepscore.sep_score.abs() <= 1) & (all_events_sepscore.sep_score.abs() > 0.5) & (all_events_sepscore.q_value < p_val_threshold)].reset_index()
significant_events = all_events_sepscore[(all_events_sepscore.sep_score.abs() > sep_score_threshold) & (all_events_sepscore.q_value < p_val_threshold)]
Z = weak[(weak['rbp']=='ccnl1') & (weak['expression']=='kd')]

interesting_rbps = ['apobec4','boll','ccnl1','cstf1','dazap1','fubp1','luc7l2','rbfox2','rbm11','tial1','trnau1ap']
strong_count = []
med_count = []
weak_count = []
columns = []
for name, df in significant_events.ix[interesting_rbps].groupby(level=['rbp','expression']):
    # strong_count['{}_{}'.format(name[0],name[1])] = len(strong[(strong['rbp']==name[0]) & (strong['expression']==name[1])].index)
    # med_count['{}_{}'.format(name[0],name[1])] = len(medium[(medium['rbp']==name[0]) & (medium['expression']==name[1])].index)
    # weak_count['{}_{}'.format(name[0],name[1])] = len(weak[(weak['rbp']==name[0]) & (weak['expression']==name[1])].index)
    
    strong_count.append(len(strong[(strong['rbp']==name[0]) & (strong['expression']==name[1])].index))
    med_count.append(len(medium[(medium['rbp']==name[0]) & (medium['expression']==name[1])].index))
    weak_count.append(len(weak[(weak['rbp']==name[0]) & (weak['expression']==name[1])].index))
    
    columns.append('{}_{}'.format(name[0],name[1]))
    s_df = pd.DataFrame(strong_count)
    m_df = pd.DataFrame(med_count)
    w_df = pd.DataFrame(weak_count)

dft = pd.concat([w_df,m_df+w_df,s_df+m_df+w_df],axis=1)
dft.columns = ['Weak (0.5 < 1)','Medium (1 < 2)','Strong (2+)']
dft = dft.T
dft.columns = columns
dft

strong

num_rows = 1
num_cols = 1
cumsum_expr_events = dft
with dataviz.Figure(os.path.join(img_dir, "total_number_and_fraction.svg"), figsize=(10, 8)) as fig:
    ax = fig.add_subplot(num_rows,num_cols,1)
    
    legend_builder = []
    legend_labels = []
    for splice_type, color in izip(reversed(cumsum_expr_events.index), sns.color_palette("Set2", len(cumsum_expr_events.index))):
        names = np.array([item for item in cumsum_expr_events.columns])
        sns.barplot(names, 
                    cumsum_expr_events.ix[splice_type], 
                    color=color, 
                    ax=ax)
        
        legend_builder.append(plt.Rectangle((0,0),.25,.25, fc=color, edgecolor = 'none'))
        legend_labels.append(splice_type)

    sns.despine(ax=ax, left=True)
    
   

    l = ax.legend(legend_builder, 
                  legend_labels, loc=1, ncol = 2, 
                  prop={'size':8}, 
                  bbox_to_anchor=(1.5, 1))
    l.draw_frame(False)
    [tick.set_rotation(90) for tick in ax.get_xticklabels()]
    #Need to change to percent
    ax.set_ylabel("Total Number of Events", fontsize=10)
    [tick.set_fontsize(12) for tick in ax.get_xticklabels()]
    ax.set_title("Total Number and sep_score Distribution",y=(1.08))

# Sanity check - CCNL1 KD should have way less events than OVER for some reason
X = all_events_sepscore.reset_index()
Y = X[(X['rbp']=='ccnl1') & (X['expression']=='over') & (abs(X['sep_score'])>0.5) & (X['q_value'] < 0.05) & (abs(X['sep_score'])<1)]
Y.shape

X = all_events_sepscore.reset_index()
Y = X[(X['rbp']=='ccnl1') & (X['expression']=='kd') & (abs(X['sep_score'])>0.5) & (X['q_value'] < 0.05) & (abs(X['sep_score'])<1)]
Y.shape

# sanity check, these should be the same number of rows:
print(dfy.shape)
print(all_events_sepscore.reset_index()[all_events_sepscore.reset_index()['splice_type']=='alt_cassette'].shape)

# we actually don't need dfy, since we're no longer finding skipped exons within these events in this notebook.
to_export = all_events_sepscore.reset_index()[all_events_sepscore.reset_index()['splice_type']=='alt_cassette']
to_export.head()

annotation_dir = '/projects/ps-yeolab3/bay001/tbos/annotations/event_centric/{}'.format(date)
rbps = set(to_export['rbp'])
for rbp in rbps:
    kd = to_export[(to_export['rbp']==rbp) & (to_export['expression']=='kd')]
    kd.to_csv(os.path.join(annotation_dir,'{}-{}.sepscore'.format(rbp,'kd')),sep='\t',index=None)
    over = to_export[(to_export['rbp']==rbp) & (to_export['expression']=='over')]
    over.to_csv(os.path.join(annotation_dir,'{}-{}.sepscore'.format(rbp,'over')),sep='\t',index=None)

set(to_export['rbp'])

all_events_sepscore, all_events_qvalue = open_omniviewer(all_data)
all_events_sepscore

all_events_sepscore.columns




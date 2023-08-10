get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
from Bio import SeqIO
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from itertools import product
import colormap as cmaps

def return_contig_id(x):
    import re
    pattern = re.compile('_(pcontig_[0-9]*)')
    return pattern.search(x).groups()[0]

#define the input folder
ASSEMBLETIC_FODLER = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/Assembletics'
FIGURE_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/figures'

os.chdir(ASSEMBLETIC_FODLER)

#get the folders and the pcontigs fasta files
_8kbp_folders = [x for x in os.listdir(ASSEMBLETIC_FODLER) if x.endswith('_8kbp')]
pcontig_fa = [x for x in os.listdir(ASSEMBLETIC_FODLER) if x.endswith('.fa') and '_pcontig' in x and 'h_ctgs' not in x]

len(_8kbp_folders) - len(pcontig_fa) #makes sense as this is the number of pwoh contigs plus whole assembly 

#make dataframe for 10kb windows
_8kbp_folders.sort()
Assemblytics_summary_df_list_10kb = []
#Assemblytics_summary_df_list_50kb = []
variation_types = ['Insertion', 'Deletion', 'Tandem_expansion', 'Tandem_contraction', 'Repeat_expansion', 'Repeat_contraction']
for folder in _8kbp_folders:
    #get the summary file
    summary_file_10kb = [x for x in os.listdir(os.path.abspath(folder))                     if x.endswith('10kp.Assemblytics_structural_variants.summary.csv')][0]
    #summary_file_50kb = [x for x in os.listdir(os.path.abspath(folder)) \
              #      if x.endswith('_50kp.Assemblytics_structural_variants.summary.csv')][0]
    file_name_10kb = ''
    file_name_10kb = os.path.join(ASSEMBLETIC_FODLER, folder, summary_file_10kb)
    #file_name_50kb = ''
    #file_name_50kb = os.path.join(base_folder, folder, summary_file_50kb)
    #now convert summary file to a df usable file
    with open(file_name_10kb) as s_handle:
        out_handle = open(file_name_10kb.replace('.csv', '.df'), 'w')
        variation_type = ''
        for line in s_handle:
            if line.strip() == '':
                continue
            if 'Total' in line:
                continue
            line = line.strip('\n')
            if line in variation_types:
                variation_type = line
                next
            else:
                line = variation_type + ','+line
                print(line, file=out_handle)
        out_handle.close()  
    #now read in the dataframe and reformat the indexes so it is useful later on
    #Multiindex column contig_id, numbers; Multiindex rows Type and size range
    contig = ''
    contig = return_contig_id(folder)
    #print('Reading in %s' % (contig))
    var_df = ''
    var_df = pd.read_csv(file_name_10kb.replace('.csv', '.df'), header=None, names=['Type', 'Size range', 'Count', 'Total bp'])
    if len(var_df) == 0: #some contigs might not have any variations called.
        continue
    index = pd.MultiIndex.from_tuples(list(zip(var_df['Type'],var_df['Size range'])), names=['Type', 'Size range'])
    var_df = var_df.set_index(index)
    var_df = var_df.iloc[:,2:] #subset the two remaining useful columns
    old_columns = list(var_df.columns)
    col_index = pd.MultiIndex.from_tuples(list(zip([contig, contig], old_columns)),names = ['contig_id','numbers'])
    var_df.columns = col_index
    Assemblytics_summary_df_list_10kb.append(var_df)
Assemblytics_summary_df_10kb = pd.concat(Assemblytics_summary_df_list_10kb, axis=1)

#make dataframe for 50kb windows
_8kbp_folders.sort()
Assemblytics_summary_df_list_50kb = []
variation_types = ['Insertion', 'Deletion', 'Tandem_expansion', 'Tandem_contraction', 'Repeat_expansion', 'Repeat_contraction']
for folder in _8kbp_folders:
    #get the summary file
    
    summary_file_50kb = [os.path.join(os.path.abspath(folder), x) for x in os.listdir(os.path.abspath(folder))               if x.endswith('_50kp.Assemblytics_structural_variants.summary.csv')][0]
    file_name_50kb = ''
    file_name_50kb = os.path.join(ASSEMBLETIC_FODLER, folder, summary_file_50kb)
    #now convert summary file to a df usable file
    with open(summary_file_50kb) as s_handle:
        out_handle = open(file_name_50kb.replace('.csv', '.df'), 'w')
        variation_type = ''
        for line in s_handle:
            if line.strip() == '':
                continue
            if 'Total' in line:
                continue
            line = line.strip('\n')
            if line in variation_types:
                variation_type = line
                next
            else:
                line = variation_type + ','+line
                print(line, file=out_handle)
        out_handle.close()  
    #now read in the dataframe and reformat the indexes so it is useful later on
    #Multiindex column contig_id, numbers; Multiindex rows Type and size range
    contig = ''
    contig = return_contig_id(folder)
    #print('Reading in %s' % (contig))
    var_df = ''
    var_df = pd.read_csv(file_name_50kb.replace('.csv', '.df'), header=None, names=['Type', 'Size range', 'Count', 'Total bp'])
    if len(var_df) == 0: #some contigs might not have any variations called.
        continue
    index = pd.MultiIndex.from_tuples(list(zip(var_df['Type'],var_df['Size range'])), names=['Type', 'Size range'])
    var_df = var_df.set_index(index)
    var_df = var_df.iloc[:,2:] #subset the two remaining useful columns
    old_columns = list(var_df.columns)
    col_index = pd.MultiIndex.from_tuples(list(zip([contig, contig], old_columns)),names = ['contig_id','numbers'])
    var_df.columns = col_index
    Assemblytics_summary_df_list_50kb.append(var_df)
Assemblytics_summary_df_50kb = pd.concat(Assemblytics_summary_df_list_50kb, axis=1)

#save out the summary dataframes
Assemblytics_summary_df_50kb.to_csv(os.path.join(ASSEMBLETIC_FODLER, 'Assemblytics_summary_df_50kb.df'))
Assemblytics_summary_df_10kb.to_csv(os.path.join(ASSEMBLETIC_FODLER, 'Assemblytics_summary_df_10kb.df'))

Assemblytics_summary_df_10kb.unstack()

Assemblytics_summary_df_10kb.sum(level='numbers', axis=1)['Total bp']/1000

#here get the summary of all Types of variations by size intervals given by Assembletics
Size_summary_10kb = Assemblytics_summary_df_10kb.sum(level='numbers', axis=1)['Total bp']/1000
index_a = [x.replace(' bp', '') for x in Size_summary_10kb["Insertion"].index]
var_types = Size_summary_10kb.index.levels[0]


#here get the summary of all Types of variations by size intervals given by Assembletics
Size_summary_10kb = Assemblytics_summary_df_10kb.sum(level='numbers', axis=1)['Total bp']/1000
index_a = [x.replace(' bp', '') for x in Size_summary_10kb["Insertion"].index]
var_types = Size_summary_10kb.index.levels[0]
#plot
sns.set_style("white")

pwh_size = 79770604
#here generate a faced plot for the size variation types
no_subplots = len(var_types)
no_of_subplots_pair = [int(no_subplots/2), 2]
subplot_coordinates = list(product(range(no_of_subplots_pair[0]), range(no_of_subplots_pair[1])))
subplot_coordinates_list = [list(l) for l in subplot_coordinates]
subplot_coordinates_list_rows = [i[0] for i in subplot_coordinates_list]
subplot_coordinates_list_columns = [i[1] for i in subplot_coordinates_list]
fig, ax = plt.subplots(no_of_subplots_pair[0], no_of_subplots_pair[1], figsize=(16,20),                   sharey='row', sharex='all')
#
#up to here generate what is needed to specificy the plots
ymax = Size_summary_10kb.max()
ind = np.arange(len(index_a))
fs = 24 #fontsize
#function to generate the subplots
def subplots(ax_ind1, ax_ind2, ind, series, name):
    sns.set_style("ticks")
    sns.despine()
    width = 0.35
    ax[ax_ind1, ax_ind2].bar(ind,series, color=sns.color_palette("husl", 6), alpha=0.8)
    for x in range(0, len(series)):
        if series[x]*1000 > 1000000:
            ax[ax_ind1, ax_ind2].text            (x-0.45, series[x]*1.02, int(series[x]*1000), fontsize=fs)
        elif series[x]*1000 > 100000:
            ax[ax_ind1, ax_ind2].text            (x-0.4, series[x]*1.02, int(series[x]*1000), fontsize=fs)
        elif series[x]*1000 < 10000:
            ax[ax_ind1, ax_ind2].text            (x-0.2, series[x]*2, int(series[x]*1000), fontsize=fs)
        else:
            ax[ax_ind1, ax_ind2].text            (x-0.3, series[x]*1.2, int(series[x]*1000), fontsize=fs)
    ax[ax_ind1, ax_ind2].set_title(name, fontsize = fs)
    if ax_ind2 == 0:
        ax[ax_ind1, ax_ind2].set_ylabel('Total kbp', fontsize = fs)
    ax[ax_ind1, ax_ind2].grid(False, which='Major')
    ax[ax_ind1, ax_ind2].tick_params(axis='both', which='major', labelsize=18, pad=3)
    ax[ax_ind1, ax_ind2].set_xticks([0, 1, 2 ,3] )
    ax[ax_ind1, ax_ind2].set_xticklabels(index_a, {'fontsize' : 18, 'horizontalalignment': 'center'}  )
    #ax[ax_ind1, ax_ind2].set_ylim(0, series.max()*1000*1.2)
    for tick in ax[ax_ind1, ax_ind2].yaxis.get_major_ticks():
              tick.label.set_fontsize(21)
        
    
for ax_ind1, ax_ind2, _type in  zip(subplot_coordinates_list_rows,subplot_coordinates_list_columns, var_types):
    subplots(ax_ind1, ax_ind2, ind, Size_summary_10kb[_type], _type)
sns.despine(offset=10, trim=True)
plt.tight_layout()
fig_name = 'Assemblytics_summary_df_10kb_v1.type_summary.png'
#plt.savefig(os.path.join(FIGURE_PATH, fig_name), bbox_inches='tight')

#here get the summary of all Types of variations by size intervals given by Assembletics
Size_summary_10kb = Assemblytics_summary_df_10kb.sum(level='numbers', axis=1)['Total bp']/1000
index_a = [x.replace(' bp', '') for x in Size_summary_10kb["Insertion"].index]
var_types = Size_summary_10kb.index.levels[0]
#plot
sns.despine(offset=10, trim=True)
pwh_size = 79770604
#here generate a faced plot for the size variation types
no_subplots = len(var_types)
no_of_subplots_pair = [int(no_subplots/2), 2]
subplot_coordinates = list(product(range(no_of_subplots_pair[0]), range(no_of_subplots_pair[1])))
subplot_coordinates_list = [list(l) for l in subplot_coordinates]
subplot_coordinates_list_rows = [i[0] for i in subplot_coordinates_list]
subplot_coordinates_list_columns = [i[1] for i in subplot_coordinates_list]
fig, ax = plt.subplots(no_of_subplots_pair[0], no_of_subplots_pair[1], figsize=(16,20),                   sharey='row', sharex='all')
#
#up to here generate what is needed to specificy the plots
ymax = Size_summary_10kb.max()
ind = np.arange(len(index_a))
fs = 25 #fontsize
#function to generate the subplots
def subplots(ax_ind1, ax_ind2, ind, series, name):
    sns.set_style("ticks")
    sns.despine()
    width = 0.35
    ax[ax_ind1, ax_ind2].bar(ind,series, color=sns.color_palette("colorblind", 6), alpha=0.8)
    for x in range(0, len(series)):
            _percentage = (series[x]*1000)/pwh_size*100
            ax[ax_ind1, ax_ind2].text            (x-0.2, series[x]+10, round(_percentage,2), fontsize=fs)
    ax[ax_ind1, ax_ind2].set_title(name, fontsize = fs)
    if ax_ind2 == 0:
        ax[ax_ind1, ax_ind2].set_ylabel('Total kbp', fontsize = fs)
    ax[ax_ind1, ax_ind2].grid(False, which='Major')
    ax[ax_ind1, ax_ind2].tick_params(axis='both', which='major', labelsize=22, pad=3)
    ax[ax_ind1, ax_ind2].set_xticks([0, 1, 2 ,3] )
    ax[ax_ind1, ax_ind2].set_xticklabels(index_a, {'fontsize' : 24, 'horizontalalignment': 'center'}  )
    ax[ax_ind1, ax_ind2].tick_params(labelsize=24)
    
    for tick in ax[ax_ind1, ax_ind2].yaxis.get_major_ticks():
              tick.label.set_fontsize(25)

    
for ax_ind1, ax_ind2, _type in  zip(subplot_coordinates_list_rows,subplot_coordinates_list_columns, var_types):
    subplots(ax_ind1, ax_ind2, ind, Size_summary_10kb[_type], _type)
#sns.despine(offset=10, trim=True)
plt.tight_layout()
fig_name = 'Assemblytics_summary_df_10kb_v3.type_summary.png'
plt.savefig(os.path.join(FIGURE_PATH, fig_name), bbox_inches='tight')

#here get the summary of all Types of variations by size intervals given by Assembletics
Size_summary_50kb = Assemblytics_summary_df_50kb.sum(level='numbers', axis=1)['Total bp']/1000
index_a = [x.replace(' bp', '') for x in Size_summary_50kb["Insertion"].index]
var_types = Size_summary_50kb.index.levels[0]
#plot
sns.despine(offset=10, trim=True)
pwh_size = 79770604
#here generate a faced plot for the size variation types
no_subplots = len(var_types)
no_of_subplots_pair = [int(no_subplots/2), 2]
subplot_coordinates = list(product(range(no_of_subplots_pair[0]), range(no_of_subplots_pair[1])))
subplot_coordinates_list = [list(l) for l in subplot_coordinates]
subplot_coordinates_list_rows = [i[0] for i in subplot_coordinates_list]
subplot_coordinates_list_columns = [i[1] for i in subplot_coordinates_list]
fig, ax = plt.subplots(no_of_subplots_pair[0], no_of_subplots_pair[1], figsize=(20,20),                   sharey='row', sharex='all')
#
#up to here generate what is needed to specificy the plots
ymax = Size_summary_10kb.max()
ind = np.arange(len(index_a))
fs = 24 #fontsize
#function to generate the subplots
def subplots(ax_ind1, ax_ind2, ind, series, name):
    sns.set_style("ticks")
    sns.despine()
    width = 0.35
    ax[ax_ind1, ax_ind2].bar(ind,series, color=sns.color_palette("husl", 6), alpha=0.8)
    for x in range(0, len(series)):
            _percentage = (series[x]*1000)/pwh_size*100
            ax[ax_ind1, ax_ind2].text            (x-0.2, series[x]+10, round(_percentage,2), fontsize=fs)
    ax[ax_ind1, ax_ind2].set_title(name, fontsize = fs)
    if ax_ind2 == 0:
        ax[ax_ind1, ax_ind2].set_ylabel('Total kbp', fontsize = fs)
    ax[ax_ind1, ax_ind2].grid(False, which='Major')
    ax[ax_ind1, ax_ind2].tick_params(axis='both', which='major', labelsize=18, pad=3)
    ax[ax_ind1, ax_ind2].set_xticks([0, 1, 2 ,3, 4] )
    ax[ax_ind1, ax_ind2].set_xticklabels(index_a, {'fontsize' : 18, 'horizontalalignment': 'center'}  )
    #ax[ax_ind1, ax_ind2].set_ylim(0, series.max()*1000*1.2)
    for tick in ax[ax_ind1, ax_ind2].yaxis.get_major_ticks():
              tick.label.set_fontsize(21)
        
    
for ax_ind1, ax_ind2, _type in  zip(subplot_coordinates_list_rows,subplot_coordinates_list_columns, var_types):
    subplots(ax_ind1, ax_ind2, ind, Size_summary_50kb[_type], _type)
plt.tight_layout()
fig_name = 'Assemblytics_summary_df_50kb_v2.type_summary.png'
plt.savefig(os.path.join(FIGURE_PATH, fig_name), bbox_inches='tight')

#summarizes here for 10kb
pwh_size = 79770604
Total_summary_df_10kb = Assemblytics_summary_df_10kb.unstack().sum(level='numbers', axis=1)
Total_summary_df_10kb['variation [1/kbp]'] = Total_summary_df_10kb['Total bp']/pwh_size * 1000
total_relative_var_10kb = Total_summary_df_10kb['Total bp'].sum()/pwh_size*100
print(round(total_relative_var_10kb, 2))

Total_summary_df_10kb['Total bp'].sum()

#summarizes here for 10kb
pwh_size = 79770604
Total_summary_df_50kb = Assemblytics_summary_df_50kb.unstack().sum(level='numbers', axis=1)
Total_summary_df_50kb['variation [1/kbp]'] = Total_summary_df_50kb['Total bp']/pwh_size * 1000
total_relative_var_50kb = Total_summary_df_50kb['Total bp'].sum()/pwh_size*100
print(round(total_relative_var_50kb, 2))

Total_summary_df_50kb['Total bp'].sum()

#find the 

Assemblytics_summary_df_10kb.head()


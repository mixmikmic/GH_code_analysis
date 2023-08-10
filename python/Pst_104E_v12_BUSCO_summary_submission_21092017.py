get_ipython().run_line_magic('matplotlib', 'inline')

import os
from Bio import SeqIO
import pandas as pd
import re
from pybedtools import BedTool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import rcParams and make it outfit also when saving files
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#define base folders
BUSCO_BASE_FOLDER = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/busco/Pst_104E_v12'
FIGURE_FOLDER = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/figures'

#first sort out the combined BUSCO of p and h
#read in the data form the primary contig analysis
fn_p = os.path.join(BUSCO_BASE_FOLDER, 'run_Pst_104E_v12_p_ctg.anno.protein.fa.protein')
busco_header = ['busco', 'status', 'protein_ID', 'Score', 'Length']
Busco_p = pd.read_csv(os.path.join(fn_p, 'full_table_Pst_104E_v12_p_ctg.anno.protein.fa.protein.tsv'),                     header=None, names=busco_header, sep='\t', comment='#')

Complete_busco_p = Busco_p

#first sort out the combined BUSCO of p and h
#read in the data form the haplotig analysis
fn_h = os.path.join(BUSCO_BASE_FOLDER, 'run_Pst_104E_v12_h_ctg.anno.protein.fa.protein')
Busco_h = pd.read_csv(os.path.join(fn_h, 'full_table_Pst_104E_v12_h_ctg.anno.protein.fa.protein.tsv'),                     header=None, names=busco_header, sep='\t', comment='#')
Complete_busco_h = Busco_h

#check for Buscos that are missing or fragmented in p
m_or_f_in_p = Complete_busco_p[(Complete_busco_p.status == 'Missing')                                | (Complete_busco_p.status == 'Fragmented')]

#get all h buscos that are missing in p
p_missing_replaced = Complete_busco_h[(Complete_busco_h.busco.isin(Complete_busco_p[(Complete_busco_p.status == 'Missing')]['busco']))                &(Complete_busco_h.status != 'Missing')]

#get all h buscos that are fragmented in p and not Missing or Fragmented in h
p_fragmented_replaced = Complete_busco_h[(Complete_busco_h.busco.isin(                        Complete_busco_p[(Complete_busco_p.status == 'Fragmented')]['busco']))                &(Complete_busco_h.status != 'Missing')&(Complete_busco_h.status != 'Fragmented')]

#now remove all the buscos that will be replaced
Fixed_total_busco = Complete_busco_p[(~Complete_busco_p.busco.isin(p_missing_replaced.busco.append(p_fragmented_replaced.busco)))].copy()

Fixed_total_busco = pd.concat([Fixed_total_busco, p_missing_replaced, p_fragmented_replaced])

#drop the duplicates as this interfers with the counting and representation at the end
Pst_104E_ph  = Fixed_total_busco.drop_duplicates(subset='busco').groupby('status')['busco'].count()

#the sum should be the total amount of searched buscos for odb9 thats 1335
Pst_104E_ph.sum()

total_buscos = Pst_104E_ph['Complete'] + Pst_104E_ph['Duplicated']
total_buscos = pd.Series([total_buscos], name='total_buscos')

Pst_104E_ph = Pst_104E_ph.append(total_buscos).reset_index(drop=True)

total_buscos 

busco_index = ['Complete and single-copy BUSCOs', 'Complete and duplicated BUSCOs', 'Fragmented BUSCOs', 'Missing BUSCOs', 'Complete BUSCOs']

Pst_104E_ph.index = busco_index

Pst_104E_ph

#these were done on the command line and transfered manually. In know I know I could have
#written a parser
Pst_104E_p = pd.Series([1121, 133, 48, 33, 1254], index= busco_index)
Pst_104E_h = pd.Series([1052,93,42,148,1145], index= busco_index)
Pst_78 = pd.Series([1135, 141,40,19,1276], index= busco_index)
Pst_130 = pd.Series([1005, 27, 181,122,1032], index= busco_index)
Pst_0821 = pd.Series([599,5,439,292,604], index= busco_index)
Pst_43 = pd.Series([933,25,245,132,958], index= busco_index)
Pst_887 = pd.Series([471,6,425,433,477], index= busco_index)
Pst_21 = pd.Series([942,35,224,134,977], index= busco_index)

SBusco_df = pd.concat([Pst_104E_p, Pst_104E_h, Pst_104E_ph, Pst_78, Pst_130, Pst_0821, Pst_43, Pst_887,Pst_21 ], axis=1)
names=['Pst-104E p', 'Pst-104E h', 'Pst-104E ph', 'Pst-78', 'Pst-130', 'Pst-0821', 'Pst-43', 'Pst-887','Pst-21' ]
SBusco_df.rename(columns=dict(zip(SBusco_df.columns, names)), inplace=True)

SBusco_df_sorted = SBusco_df.T.sort_values(by='Complete BUSCOs' ,ascending=False)

#now add a fake fragmented column to plot it as the first column in the plot
SBusco_df_sorted['Fake Fragmented'] = SBusco_df_sorted['Fragmented BUSCOs'] + SBusco_df_sorted['Complete BUSCOs']

SBusco_df_sorted

sns.palplot(sns.color_palette('colorblind')[0:4])

#sns.set_color_codes(palette='deep')

sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(14, 16))
#set the font size 
fs = 24

sns.barplot(x='Fake Fragmented',y=SBusco_df_sorted.index, data=SBusco_df_sorted,             label='Fragmented BUSCOs', color=sns.color_palette('colorblind')[0],alpha=1  )

#sns.set_color_codes("pastel")
sns.barplot(x='Complete BUSCOs',y=SBusco_df_sorted.index, data=SBusco_df_sorted,             label='Complete BUSCOs', color=sns.color_palette('colorblind')[2],alpha=1 )

#sns.set_color_codes("muted")
sns.barplot(x='Complete and single-copy BUSCOs',y=SBusco_df_sorted.index,            data=SBusco_df_sorted, label='Complete and single-copy BUSCOs',             color=sns.color_palette('colorblind')[1], alpha=1 )

#sns.set_color_codes("pastel")
sns.barplot(x='Complete and duplicated BUSCOs',y=SBusco_df_sorted.index,             data=SBusco_df_sorted, label='Complete and duplicated BUSCOs',             color=sns.color_palette('colorblind')[4],alpha=1  )



# Add a legend and informative axis label
ax.legend(ncol=2, loc=(0.1, 1.01), frameon=True,fancybox=True, fontsize = fs)
count = 0
for y, complete, missing in zip(np.arange(0, 9, 1),SBusco_df_sorted['Complete BUSCOs'], SBusco_df_sorted['Missing BUSCOs'] ):
    if count == 0:
        ax.text(1500, y, 'Complete BUSCOs: %i/(1302)*\nMissing BUSCOs: %i/(10)*' % (complete, missing)           ,{'fontsize' : fs})

    else:
        ax.text(1500, y, 'Complete BUSCOs: %i\nMissing BUSCOs: %i' % (complete, missing)           ,{'fontsize' : fs})
    count += 1
plt.xlabel('Number of BUSCOs', {'fontsize' : fs})
plt.axvline(x=1335,  color='k')
ax.text(1500,8.8, '#BUSCO searched: 1335',{'fontsize' : fs})
#adjust the last ticks
ax.text(1300,8.72,'1335', fontsize=fs)
sns.despine(left=True, bottom=True)
#add xticks as wanted
plt.xticks([0, 200,400,600,800,1000,1200] )
ax.tick_params(axis = 'y', labelsize=fs)
ax.tick_params(axis = 'x', labelsize=fs)
plt.savefig(os.path.join(FIGURE_FOLDER, 'Busco_summary_figure_v3.png'), dpi=600, bbox_inches="tight")




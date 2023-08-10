get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import matplotlib.pyplot as plt
#import seaborn
import matplotlib
from Bio import SeqIO, SeqUtils
import os

#Define the PATH
BASE_AA_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12'
BASE_A_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/032017_assembly'
#for now use the previous mapping that still included high coverage regions
BASE_ORIGINAL = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_E104_v1'
COV_IN_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_E104_v1/COV'
BAM_IN_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_E104_v1/SRM'
#apply analysis restricted to final assembly Pst_104E_v12
COV_OUT_PATH = os.path.join(BASE_AA_PATH, 'COV')
if not os.path.isdir(COV_OUT_PATH):
    os.mkdir(COV_OUT_PATH)

input_genome = 'Pst_E104_v1'
coverage_file_suffix = 'bwamem.Pst79_folder5.sam.sorted.bam.aall.cov'
output_genome = 'Pst_104E_v12'

#get all the cov files with this coverage_file_suffix. Expacts to get all three p, h and ph mapping
cov_files = [os.path.join(COV_IN_PATH, x) for x in os.listdir(COV_IN_PATH) if x.endswith(coverage_file_suffix)]

cov_header = ["contig", "position", 'coverage']

ph_cov = pd.read_csv([y for y in cov_files if 'ph_ctg' in y][0], sep='\t', header=None, names=cov_header)
print('Read in following file as ph_coverage produced with samtools depth -aa feature: %s' %[y for y in cov_files if 'ph_ctg' in y][0])

p_cov = pd.read_csv([y for y in cov_files if 'p_ctg' in y][0], sep='\t', header=None, names=cov_header)
print('Read in following file as p_coverage produced with samtools depth -aa feature: %s' %[y for y in cov_files if 'p_ctg' in y][0])

h_cov = pd.read_csv([y for y in cov_files if 'h_ctg' in y][0], sep='\t', header=None, names=cov_header)
print('Read in following file as h_coverage produced with samtools depth -aa feature: %s' %[y for y in cov_files if 'h_ctg' in y][0])

#summarize the mean coverage by contigs for p_contigs when mapping against p_contigs only
mean_cov_per_contig = p_cov.groupby('contig').mean()

mean_cov_per_contig['coverage'].plot.box()

overall_mean = p_cov['coverage'].mean()

overall_std = p_cov['coverage'].std()

print("The mean overall coverage is %.2f and the std is %.2f for p" % (overall_mean, overall_std))

#summarize the mean coverage by contigs for all contigs when mapping against p and h contigs. Plot only coverage plot for p_contigs
mean_cov_per_contig_ph = ph_cov.groupby('contig').mean()
mean_cov_per_contig_ph['contig'] = mean_cov_per_contig_ph.index
mean_cov_per_contig_ph[mean_cov_per_contig_ph['contig'].str.contains('pcontig')]['coverage'].plot.box(sym ='r+')
overall_mean_ph = ph_cov['coverage'].mean()

overall_std_ph = ph_cov['coverage'].std()

print("The mean overall coverage is %.2f and the std is %.2f for ph mapping" % (overall_mean_ph, overall_std_ph))

#summarize the mean coverage by contigs for h contigs when mapping against h contigs. Plot only coverage plot for h_contigs
mean_cov_per_contig_h = h_cov.groupby('contig').mean()
mean_cov_per_contig_h['contig'] = mean_cov_per_contig_h.index
mean_cov_per_contig_h['coverage'].plot.box(sym ='r+')

overall_mean_h = h_cov['coverage'].mean()

overall_std_h = h_cov['coverage'].std()

print("The mean overall coverage is %.2f and the std is %.2f for ph mapping" % (overall_mean_h, overall_std_h))

#drop all contigs that have mean coverage above 2000 as calcuated on p mapping.
pcontig_greater_2000 = mean_cov_per_contig[mean_cov_per_contig['coverage'] > 2000].index
p_cov_smaller_2000 = p_cov[~p_cov['contig'].isin(pcontig_greater_2000)]
pcontig_smaller_2000 = p_cov_smaller_2000['contig'].unique()
ph_cov_smaller_2000_p = ph_cov[ph_cov['contig'].isin(pcontig_smaller_2000)]

#drop all contigs that have mean coverage above 2000 as calcuated on p mapping. 
#In this case h selected on the ph contig
ph_cov['pcontig'] = ph_cov['contig'].str.replace('h','p').str[:-4] #this is a bit of a hack as the pcontigs are also
#proccessed but shorten so the next line selects only h contigs
ph_cov_smaller_2000_h = ph_cov[ph_cov['pcontig'].isin(pcontig_smaller_2000)]

#drop all contigs that have mean coverage above 2000 as calcuated on p mapping. Apply this to the h contigs as well
mean_cov_per_contig_h['pcontig'] = mean_cov_per_contig_h['contig'].str.replace('h','p').str[:-4]
h_cov['pcontig'] = h_cov['contig'].str.replace('h','p').str[:-4]
h_cov_smaller_2000 = h_cov[h_cov['pcontig'].isin(pcontig_smaller_2000)]

mean_cov_contig_s2000 = p_cov_smaller_2000.groupby(by='contig')['coverage'].mean()

mean_cov_contig_s2000.plot.box()

mean_s2000 = mean_cov_contig_s2000.mean()
std_s2000 = mean_cov_contig_s2000.std()

print("The mean overall coverage for s2000 contigs is %.2f and the std is %.2f" % (mean_s2000, std_s2000))

mean_cov_contig_s2000_ph_p = ph_cov_smaller_2000_p.groupby(by='contig')['coverage'].mean()
mean_cov_contig_s2000_ph_p.plot.box()
plt.title('mean_cov_contig_s2000_ph_p')

mean_cov_contig_s2000_h = h_cov_smaller_2000.groupby(by='contig')['coverage'].mean()
mean_cov_contig_s2000_h.plot.box()
plt.title('mean_cov_contig_s2000_h')

mean_cov_contig_s2000_ph_h = ph_cov_smaller_2000_h.groupby(by='contig')['coverage'].mean()
mean_cov_contig_s2000_ph_h.plot.box()
plt.title('mean_cov_contig_s2000_ph_h')

#these might be collpased repeats
mean_cov_contig_s2000_h[mean_cov_contig_s2000_h > 200]

mean_cov_contig_s2000_ph_h[mean_cov_contig_s2000_ph_h >200]

len(ph_cov_smaller_2000_p['contig'].unique())

#get the list of p contigs with haplotig [pwh] and without haplotig [pwoh]
pwh_list = pd.read_csv('/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_E104_v1/Pst_E104_v1_pwh_ctg.txt',sep='\t', header=None)[0].tolist()
pwoh_list = pd.read_csv('/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_E104_v1/Pst_E104_v1_pwoh_ctg.txt',sep='\t', header=None)[0].tolist()

mean_s2000_ph_p = mean_cov_contig_s2000_ph_p.mean()
std_s2000_ph_p = mean_cov_contig_s2000_ph_p.std()

print("The mean overall coverage for primary contigs with s2000 contigs is %.2f and the std is %.2f" % (mean_s2000_ph_p, std_s2000_ph_p))

mean_s2000_ph_h = mean_cov_contig_s2000_ph_h.mean()
std_s2000_ph_h = mean_cov_contig_s2000_ph_h.std()

print("The mean overall coverage for haplotigs with primary contigs cov  < 2000 is %.2f and the std is %.2f" % (mean_s2000_ph_h, std_s2000_ph_h))

#mean_cov_contig_s2000_ph_pwh represents all contigs that are below 2000x coverage and are pwh contigs
mean_cov_contig_s2000_ph_pwh = mean_cov_contig_s2000_ph_p[mean_cov_contig_s2000_ph_p.index.isin(pwh_list)]
mean_cov_contig_s2000_ph_pwh.plot.box()
plt.title('mean_cov_contig_s2000_ph_pwh')
mean_s2000_ph_pwh = mean_cov_contig_s2000_ph_pwh.mean()
std_s2000_ph_pwh = mean_cov_contig_s2000_ph_pwh.std()
print("The mean overall coverage for s2000 contigs of pwh contigs while ph mapping is %.2f and the std is %.2f" % (mean_s2000_ph_pwh, std_s2000_ph_pwh))

mean_cov_contig_s2000_ph_pwoh = mean_cov_contig_s2000_ph_p[mean_cov_contig_s2000_ph_p.index.isin(pwoh_list)]
mean_cov_contig_s2000_ph_pwoh.plot.box()
plt.title('mean_cov_contig_s2000_ph_pwoh')
mean_s2000_ph_pwoh = mean_cov_contig_s2000_ph_pwoh.mean()
std_s2000_ph_pwoh = mean_cov_contig_s2000_ph_pwoh.std()
print("The mean overall coverage for s2000 contigs of pwoh contigs while ph mapping is %.2f and the std is %.2f" % (mean_s2000_ph_pwoh, std_s2000_ph_pwoh))

#add h contigs as well
print("We have %i sensible pwh contigs with %i h contigs and %i sensible pwoh contigs" % (len(mean_cov_contig_s2000_ph_pwh),len(mean_cov_contig_s2000_ph_h.index), len(mean_cov_contig_s2000_ph_pwoh) ))

mean_cov_contig_s2000_ph_h.index

mean_cov_contig_s2000_ph_pwh.index

mean_cov_contig_s2000_ph_p.index

#think about what thresholds to pick in the long run.
threshold_up_ph_p = mean_s2000_ph_p + 2*std_s2000_ph_p
threshold_down_ph_p = mean_s2000_ph_p - 2*std_s2000_ph_p

threshold_up = mean_s2000 + 2*std_s2000
threshold_down = mean_s2000 - 2*std_s2000

#potnetial fully homozygous contigs
mean_cov_contig_s2000_ph_pwoh[mean_cov_contig_s2000_ph_pwoh > threshold_up_ph_p  ]

mean_cov_contig_s2000_ph_pwoh

2*mean_s2000_ph_p - std_s2000_ph_p

mean_s2000 - std_s2000

#think about what thresholds to pick in the long run.
threshold_up_ph_p = mean_s2000_ph_p + 2*std_s2000_ph_p
threshold_down_ph_p = mean_s2000_ph_p - 2*std_s2000_ph_p

threshold_up = mean_s2000 + 2*std_s2000
threshold_down = mean_s2000 - 2*std_s2000



import warnings
warnings.filterwarnings('ignore')

#now write a loop that does it all for your over the whole two dataframes
bed_p_uniqe_list = []
bed_p_homo_list = []
process_p_df_dict = {}
process_ph_df_dict = {}
for contig in pcontig_smaller_2000:
    tmp_p_df = ''
    tmp_ph_df = ''
    #now subset the two dataframes
    tmp_p_df = p_cov[p_cov['contig'] == contig]
    tmp_p_df_ph = ph_cov[ph_cov['contig'] ==  contig]
    #generarte the rolling windows
    tmp_p_df['Rolling_w1000_p'] = tmp_p_df.rolling(window=1000, min_periods=1, center=True, win_type='blackmanharris')['coverage'].mean()
    tmp_p_df_ph['Rolling_w1000_ph_p'] = tmp_p_df_ph.rolling(window=1000, min_periods=1, center=True, win_type='blackmanharris')['coverage'].mean()
    tmp_p_df['Rolling_w10000_p'] = tmp_p_df.rolling(window=10000, min_periods=1, center=True, win_type='blackmanharris')['coverage'].mean()
    tmp_p_df_ph['Rolling_w10000_ph_p'] = tmp_p_df_ph.rolling(window=10000, min_periods=1,center=True, win_type='blackmanharris')['coverage'].mean()
    tmp_p_df['Rolling_w1000_ph_p'] = tmp_p_df_ph['Rolling_w1000_ph_p']
    process_p_df_dict[contig] = tmp_p_df
    process_ph_df_dict[contig] = tmp_ph_df
    #potentially p_unique DNA streatches are defined as p contig cov streatches, while doing p mapping, that are heterozygous coverage
    # coverage -> mean_s2000_ph_p
    # [Rolling_w1000_p < mean_s2000_ph_p + 2*std_s2000_ph_p]
    tmp_p_df_p_unique = tmp_p_df[tmp_p_df['Rolling_w1000_p'] < (mean_s2000_ph_p + 2*std_s2000_ph_p)]
    if len(tmp_p_df_p_unique) > 0:
        tmp_p_df_p_unique.reset_index(drop=True, inplace=True)
        #add a position +1 column by copying the position datafram 1: and adding making position+1 for the last element
        # in the dataframe equal to its own value
        tmp_p_df_p_unique['position+1']= tmp_p_df_p_unique.loc[1:, 'position'].        append(pd.Series(tmp_p_df_p_unique.loc[len(tmp_p_df_p_unique)-1, 'position'], index=[tmp_p_df_p_unique.index[-1]])).reset_index(drop=True)

        tmp_p_df_p_unique['position_diff+1'] = tmp_p_df_p_unique['position+1'] - tmp_p_df_p_unique['position']

        #add a position -1 column by copying the position datafram :len-2 and adding/making position-1 for the first element
        # in the dataframe equal to its own value
        position_1 = list(tmp_p_df_p_unique.loc[:len(tmp_p_df_p_unique)-2, 'position'])
        position_1.insert(0, tmp_p_df_p_unique.loc[0, 'position'])

        tmp_p_df_p_unique['position-1']= position_1

        tmp_p_df_p_unique['position_diff-1'] =  tmp_p_df_p_unique['position'] - tmp_p_df_p_unique['position-1']
        #start points of feature streatch => where previous position is unequal 1 away
        #tmp_p_df_p_unique[tmp_p_df_p_unique['position_diff-1'] != 1 ].head()

        start_pos_index = ''
        stop_pos_index = ''
        contig_name_list = ''
        p_unique_bed = ''
        #this should be good  now as it flows double check and loop around to finish this off
        start_pos_index = tmp_p_df_p_unique[tmp_p_df_p_unique['position_diff-1'] != 1 ].index
        stop_pos_index = tmp_p_df_p_unique[tmp_p_df_p_unique['position_diff+1'] != 1 ].index

        contig_name_list = [contig]*len(start_pos_index)

        start_pos = [tmp_p_df_p_unique.loc[pos, 'position'] -1 for pos in start_pos_index]
        stop_pos = [tmp_p_df_p_unique.loc[pos, 'position']  for pos in stop_pos_index]

        p_unique_bed = pd.DataFrame([contig_name_list, start_pos, stop_pos]).T
        bed_p_uniqe_list.append(p_unique_bed)
    
    #potentially p_homo DNA streatches are defined as p contig cov streatches, while doing ph mapping, that are homozygous coverage
    # coverage -> 2*mean_s2000_ph_p
    # [Rolling_w1000_p > 2*mean_s2000_ph_p - 2*std_s2000_ph_p]
    #here might be a consideration to ask for a difference in profile (covariance != 1)
    tmp_p_df_p_homo = tmp_p_df[(tmp_p_df['Rolling_w1000_ph_p'] > (2*mean_s2000_ph_p - 2*std_s2000_ph_p))]
    if len(tmp_p_df_p_homo) > 0:
        tmp_p_df_p_homo.reset_index(drop=True, inplace=True)
        #add a position +1 column by copying the position datafram 1: and adding making position+1 for the last element
        # in the dataframe equal to its own value
        tmp_p_df_p_homo['position+1']= tmp_p_df_p_homo.loc[1:, 'position'].        append(pd.Series(tmp_p_df_p_homo.loc[len(tmp_p_df_p_homo)-1, 'position'], index=[tmp_p_df_p_homo.index[-1]])).reset_index(drop=True)

        tmp_p_df_p_homo['position_diff+1'] = tmp_p_df_p_homo['position+1'] - tmp_p_df_p_homo['position']

        #add a position -1 column by copying the position datafram :len-2 and adding/making position-1 for the first element
        # in the dataframe equal to its own value
        position_1 = list(tmp_p_df_p_homo.loc[:len(tmp_p_df_p_homo)-2, 'position'])
        position_1.insert(0, tmp_p_df_p_homo.loc[0, 'position'])

        tmp_p_df_p_homo['position-1']= position_1

        tmp_p_df_p_homo['position_diff-1'] =  tmp_p_df_p_homo['position'] - tmp_p_df_p_homo['position-1']
        #start points of feature streatch => where previous position is unequal 1 away
        #tmp_p_df_p_homo[tmp_p_df_p_homo['position_diff-1'] != 1 ].head()

        start_pos_index = ''
        stop_pos_index = ''
        contig_name_list = ''
        p_homo_bed = ''
        #this should be good  now as it flows double check and loop around to finish this off
        start_pos_index = tmp_p_df_p_homo[tmp_p_df_p_homo['position_diff-1'] != 1 ].index
        stop_pos_index = tmp_p_df_p_homo[tmp_p_df_p_homo['position_diff+1'] != 1 ].index

        contig_name_list = [contig]*len(start_pos_index)

        start_pos = [tmp_p_df_p_homo.loc[pos, 'position'] -1 for pos in start_pos_index]
        stop_pos = [tmp_p_df_p_homo.loc[pos, 'position']  for pos in stop_pos_index]

        p_homo_bed = pd.DataFrame([contig_name_list, start_pos, stop_pos]).T
        bed_p_homo_list.append(p_homo_bed)

print('Contig %s done.' % contig)

len(bed_p_uniqe_list)

p_homo_bed_df = pd.concat(bed_p_homo_list).sort_values(by=[0,1])
p_unique_bed_df =  pd.concat(bed_p_uniqe_list).sort_values(by=[0,1])

#p_homo_bed_df.to_csv(cov_folder+'Pst_E104_v1_ph_ctg.ph_p_homo_cov.bed', header=None, index=None, sep='\t')
#p_unique_bed_df.to_csv(cov_folder+'Pst_E104_v1_ph_ctg.p_p_het_cov.bed', header=None, index=None, sep='\t')

p_homo_bed_df.to_csv(os.path.join(COV_OUT_PATH, output_genome + '_ph_ctg.ph_p_homo_cov.bed'), header=None, index = None, sep ='\t')
p_unique_bed_df.to_csv(os.path.join(COV_OUT_PATH, output_genome + '-ph_ctg.p_p_het_cov.bed'), header=None, index = None, sep ='\t')

#also write out the proceessed dataframes with rolling and such
process_p_df = pd.concat(process_p_df_dict.values()).sort_values(by=['contig','position'])
process_ph_df = pd.concat(process_p_df_dict.values()).sort_values(by=['contig','position'])
process_p_df.to_csv(os.path.join(COV_OUT_PATH, input_genome + 'p_ctg.' +coverage_file_suffix.replace('.cov', '.processed.cov')), index = None, sep ='\t')
process_ph_df.to_csv(os.path.join(COV_OUT_PATH, input_genome + 'ph_ctg.'+coverage_file_suffix.replace('.cov', '.processed.cov')), index = None, sep ='\t')

process_p_df[process_p_df['Rolling_w1000_p'] <  20]




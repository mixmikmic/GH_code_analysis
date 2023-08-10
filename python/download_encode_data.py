import pandas as pd
import numpy as np
import os
import glob
pd.set_option("display.max_columns",500)

def get_replicate_num(df):
    """ By Xintao's definition, the lower replicate id sorted alphanumerically is the first replicate. """
    df['Replicate_number'] = df.sort_values(
        ['Experiment_ID', 'RBP', 'Cell_line', 'Replicate']
    ).groupby(['Experiment_ID','RBP','Cell_line'])['Replicate'].rank()
    df['Replicate_number'] = df['Replicate_number'].astype(int)
    return df

wd = '/projects/ps-yeolab3/encode/rnaseq'
hepg2_filelist = pd.read_table(os.path.join(wd, 'HepG2_fileID_table.20171211_files.txt'), comment='#')
hepg2_exptlist = pd.read_table(os.path.join(wd, 'HepG2_expID_table.20171211_datasets.txt'), comment='#')

k562_filelist = pd.read_table(os.path.join(wd, 'K562_fileID_table.20171211_files.txt'), comment='#')
k562_exptlist = pd.read_table(os.path.join(wd, 'K562_expID_table.20171211_datasets.txt'), comment='#')

k562_filelist = get_replicate_num(k562_filelist)
hepg2_filelist = get_replicate_num(hepg2_filelist)

hepg2 = {'filelist':hepg2_filelist, 'exptlist':hepg2_exptlist}
k562 = {'filelist':k562_filelist, 'exptlist':k562_exptlist}





def concat_expt_filelists(cell):
    # merge the knockdown experiments
    merged = pd.merge(
        cell['exptlist'], 
        cell['filelist'], 
        how='left', 
        left_on=['Knockdow_exp','RBP','Cell_line'], 
        right_on=['Experiment_ID','RBP','Cell_line']
    )
    kd_cols = {colname:'KD {}'.format(colname) for colname in [
        'Experiment_ID', 'Replicate', 'FASTQ_R1', 'FASTQ_R2', 'BAM', 'TSV', 'RBP', 'Replicate_number',
    ]}
    merged.rename(columns=kd_cols, inplace=True)

    # merge the control experiments
    # RBP is omitted because the control expt RBPs are all 'non-target'
    
    merged = pd.merge(
        merged, 
        cell['filelist'], 
        how='left', 
        left_on=['Control_exp','Cell_line', 'KD Replicate_number'], 
        right_on=['Experiment_ID','Cell_line', 'Replicate_number']
    )

    ctrl_cols = {colname:'Control {}'.format(colname) for colname in [
        'Experiment_ID', 'Replicate', 'FASTQ_R1', 'FASTQ_R2', 'BAM', 'TSV', 'RBP', 'Replicate_number',
    ]}
    merged.rename(columns=ctrl_cols, inplace=True)
    
    # let's just add the BAM suffix here
    merged['KD BAM'] = merged['KD BAM'] + '.bam'
    merged['Control BAM'] = merged['Control BAM'] + '.bam'
    
    return merged

hepg2_combined_filelist = concat_expt_filelists(hepg2)
hepg2_combined_filelist.iloc[pd.isnull(hepg2_combined_filelist).any(1).nonzero()[0]]

k562_combined_filelist = concat_expt_filelists(k562)
k562_combined_filelist.iloc[pd.isnull(k562_combined_filelist).any(1).nonzero()[0]]

def get_rmats_formatted_SE_name(row):
    return "{}-{}-{}-SE.MATS.JunctionCountOnly.txt".format(
        row['KD RBP'], row['SET'], row['Cell_line']
    )

k562_combined_filelist['RMATS_FILE'] = k562_combined_filelist.apply(get_rmats_formatted_SE_name, axis=1)
hepg2_combined_filelist['RMATS_FILE'] = hepg2_combined_filelist.apply(get_rmats_formatted_SE_name, axis=1)

bam_master_directory = '/projects/ps-yeolab3/encode/rnaseq/shrna_knockdown_graveley_tophat/'
rmats_master_directory = '/projects/ps-yeolab3/encode/rnaseq/alt_splicing/graveley_rmats_current/'
missing_filelist = '/home/bay001/projects/maps_20160420/permanent_data/missing_rmats_files_20171218.txt'

missing_files = []
for cell_file in [hepg2_combined_filelist, k562_combined_filelist]:
    for rmats_file in cell_file['RMATS_FILE']:
        found_file = glob.glob(os.path.join(rmats_master_directory, rmats_file))
        if len(found_file) == 0:
            missing_files.append(rmats_file)
            
with open(missing_filelist, 'w') as o:
    for missing_file in set(missing_files):
        o.write(missing_file + '\n')

missing_files = []
for cell_file in [hepg2_combined_filelist, k562_combined_filelist]:
    for bam_file in cell_file['KD BAM']:
        found_file = glob.glob(os.path.join(bam_master_directory, bam_file))
        if len(found_file) == 0:
            missing_files.append(bam_file)
missing_files

# ENCFF257RZL.bam is an old filename from xintao that has to be manually updated
hepg2_combined_filelist[hepg2_combined_filelist['KD BAM']=='ENCFF257RZL.bam']

pd.concat([hepg2_combined_filelist, k562_combined_filelist]).to_csv(
    '/projects/ps-yeolab3/encode/rnaseq/encode_master_filelist.txt', sep='\t', index=False
)

reformatted_hepg2 = hepg2_combined_filelist[['KD RBP', 'RMATS_FILE']]
reformatted_hepg2['EXP'] = reformatted_hepg2['RMATS_FILE'].str.replace('-SE.MATS.JunctionCountOnly.txt','')
del reformatted_hepg2['RMATS_FILE']
reformatted_hepg2.columns = ['Official_RBP', 'EXP']
reformatted_hepg2.drop_duplicates(['Official_RBP', 'EXP'], inplace=True)
reformatted_hepg2.to_csv(
    '/home/bay001/projects/maps_20160420/permanent_data/RNASeq_final_exp_list_HepG2.csv_12-18-2017.csv',
    sep='\t', index=False
)

reformatted_k562 = k562_combined_filelist[['KD RBP', 'RMATS_FILE']]
reformatted_k562['EXP'] = reformatted_k562['RMATS_FILE'].str.replace('-SE.MATS.JunctionCountOnly.txt','')
del reformatted_k562['RMATS_FILE']
reformatted_k562.columns = ['Official_RBP', 'EXP']
reformatted_k562.drop_duplicates(['Official_RBP', 'EXP'], inplace=True)
reformatted_k562.to_csv(
    '/home/bay001/projects/maps_20160420/permanent_data/RNASeq_final_exp_list_K562.csv_12-18-2017.csv',
    sep='\t'
)




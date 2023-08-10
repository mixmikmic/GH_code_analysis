from clustergrammer_widget import *
import pandas as pd
import numpy as np
net = Network(clustergrammer_widget)

# load quantile normalized PTM data
filename = '../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/ptmCCLE_col-iqn.txt'
net.load_file(filename)

# filter out ptms that have more than seven measurements
net.filter_threshold('row', threshold=0, num_occur=30)

# normalize PTMs across all cell lines 
net.normalize(axis='row', norm_type='zscore', keep_orig=False)

net.swap_nan_for_zero()
net.cluster()
net.dat['mat'].shape

# net.cluster(views=[])
# net.widget()

ptm_proc = net.export_df()
rows = ptm_proc.index.tolist()

# add PTM type as a category
row_cats = []
for inst_row in rows:
    inst_cat = 'Data-Type: ' + inst_row.split('_')[1].split('_')[0]
    row_cats.append( (inst_row, inst_cat) )

ptm_proc.index = row_cats    

net.load_df(ptm_proc)
net.cluster(views=[])
net.widget()

filename = '../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CCLE_CST_lung.txt'
net.load_file(filename)

# keep the top 1000 variable genes across the cell lines
net.filter_N_top('row', 1000, 'var')
net.normalize(axis='row', norm_type='zscore', keep_orig=False)

# set max abs-val of any expression Z-score to 10
# we do not care about extreme outliers
# net.dat['mat'] = np.clip(net.dat['mat'], -10, 10)
net.dat['mat'].shape

net.cluster()
net.widget()

ccle = net.export_df()
exp_rows = ccle.index.tolist()

exp_row_cats = []
for inst_row in exp_rows:
    exp_row_cats.append( (inst_row, 'Data-Type: Exp') )
    
ccle.index = exp_row_cats

merge_df = ptm_proc.append(ccle)
merge_df.shape

net.load_df(merge_df)
net.cluster(views=[])
net.widget()

cols = merge_df.columns.tolist()

# load cell line information from json 
cl_info = net.load_json_to_dict('../cell_line_info/cell_line_info_dict.json')

col_cats = []
for inst_col in cols:
    inst_info = cl_info[inst_col]
    inst_tuple = ('Cell Line: '+inst_col,)
    
    inst_tuple = inst_tuple + ( 'Histology: '+inst_info['Histology'] ,)
    inst_tuple = inst_tuple + ( 'Sub-Histology: '+inst_info['Sub-Histology'] ,)
    
    for inst_mut in ['mut-TP53', 'mut-EGFR', 'mut-RB1', 'mut-KRAS']:
    
        inst_string = inst_mut+ ': ' + str(inst_info[inst_mut])
        inst_tuple = inst_tuple + (inst_string,)

    col_cats.append(inst_tuple)
    
merge_df.columns = col_cats    
ptm_proc.columns = col_cats
ccle.columns = col_cats

merge_df.to_csv('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_merge.txt', sep='\t')
ptm_proc.to_csv('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_ptm.txt', sep='\t')
ccle.to_csv('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_exp.txt', sep='\t')

net.load_file('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_merge.txt')
net.cluster(views=[])
net.widget()

# net.load_file('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_ptm.txt')
# net.cluster(views=[])
# net.widget()

for inst_type in ['phospho', 'Rme1', 'Kme1', 'Exp', 'AcK']:
    df = net.export_df()
    df = df.transpose()

    genes = df.columns.tolist()
    
    genes = [i for i in genes if i[1] == 'Data-Type: ' + inst_type]
    
    df = df[genes]
    
    df = df.transpose()
    print(df.shape)
    
    df.to_csv('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_'+ inst_type + '.txt', sep='\t')
    




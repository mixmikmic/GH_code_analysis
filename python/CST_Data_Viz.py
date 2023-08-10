# make instance of Clustergrammer's Network object and pass in the clustergrammer_widget class
from clustergrammer_widget import *
net = Network(clustergrammer_widget)

# load our data
net.load_file('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_ptm.txt')

# check shape of data
print('PTM data shape: '+ '' + str(net.dat['mat'].shape))

# manually set category colors for rows and columns
net.set_cat_color('row', 1, 'Data-Type: phospho', 'red')
net.set_cat_color('row', 1, 'Data-Type: Rme1', 'purple')
net.set_cat_color('row', 1, 'Data-Type: AcK', 'blue')
net.set_cat_color('row', 1, 'Data-Type: Kme1', 'grey')
net.set_cat_color('col', 1, 'Histology: SCLC', 'red')
net.set_cat_color('col', 1, 'Histology: NSCLC', 'blue')
net.set_cat_color('col', 2, 'Sub-Histology: SCLC', 'red')
net.set_cat_color('col', 2, 'Sub-Histology: NSCLC', 'blue')
net.set_cat_color('col', 2, 'Sub-Histology: squamous_cell_carcinoma', 'yellow')
net.set_cat_color('col', 2, 'Sub-Histology: bronchioloalveolar_adenocarcinoma', 'orange')
net.set_cat_color('col', 2, 'Sub-Histology: adenocarcinoma', 'grey')

net.cluster(views=[])
net.widget()

# # here we are using the interactive dendrogram (not shown) to select clusters and export them to TSVs using
# # the widget_df method. 
# ptm_sclc = net.widget_df()
# ptm_nsclc = net.widget_df()
# ptm_sclc.to_csv('histology_clusters/ptm_sclc.txt', sep='\t')
# ptm_nsclc.to_csv('histology_clusters/ptm_nsclc.txt', sep='\t')

net.load_file('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_exp.txt')
print('Expression data shape: ' + str(net.dat['mat'].shape))

net.set_cat_color('row', 1, 'Data-Type: Exp', 'yellow')
net.cluster(views=[])
net.widget()

# exp_sclc = net.widget_df()
# exp_nsclc = net.widget_df()
# exp_sclc.to_csv('histology_clusters/exp_sclc.txt', sep='\t')
# exp_nsclc.to_csv('histology_clusters/exp_nsclc.txt', sep='\t')

# load merged PTM and gene expression data
net.load_file('../lung_cellline_3_1_16/lung_cl_all_ptm/precalc_processed/CST_CCLE_merge.txt')
net.cluster(views=[])
net.widget()

# merge_sclc = net.widget_df()
# merge_nsclc = net.widget_df()
# merge_sclc.to_csv('histology_clusters/merge_sclc.txt', sep='\t')
# merge_nsclc.to_csv('histology_clusters/merge_nsclc.txt', sep='\t')

net.load_file('histology_clusters/merge_sclc.txt')
net.enrichrgram('GO_Biological_Process_2015')
net.cluster(views=[])
net.widget()

net.load_file('histology_clusters/merge_nsclc.txt')
net.enrichrgram('GO_Biological_Process_2015')
net.cluster(views=[])
net.widget()


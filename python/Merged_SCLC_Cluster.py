from clustergrammer_widget import *
net = Network(clustergrammer_widget)

# load data
net.load_file('histology_clusters/merge_sclc.txt')
merge_sclc = net.export_df()

# manually set category colors for rows and columns
net.set_cat_color('row', 1, 'Data-Type: phospho', 'red')
net.set_cat_color('row', 1, 'Data-Type: Rme1', 'purple')
net.set_cat_color('row', 1, 'Data-Type: AcK', 'blue')
net.set_cat_color('row', 1, 'Data-Type: Kme1', 'grey')
net.set_cat_color('row', 1, 'Data-Type: Exp', 'yellow')
net.set_cat_color('col', 1, 'Histology: SCLC', 'red')
net.set_cat_color('col', 1, 'Histology: NSCLC', 'blue')
net.set_cat_color('col', 2, 'Sub-Histology: SCLC', 'red')
net.set_cat_color('col', 2, 'Sub-Histology: NSCLC', 'blue')
net.set_cat_color('col', 2, 'Sub-Histology: squamous_cell_carcinoma', 'yellow')
net.set_cat_color('col', 2, 'Sub-Histology: bronchioloalveolar_adenocarcinoma', 'orange')
net.set_cat_color('col', 2, 'Sub-Histology: adenocarcinoma', 'grey')

net.cluster(views=[])
net.widget()

net.enrichrgram('GO_Biological_Process_2015')
net.cluster(views=[])
net.widget()

net.enrichrgram('Disease_Perturbations_from_GEO_up')
net.cluster(views=[])
net.widget()

net.enrichrgram('MGI_Mammalian_Phenotype_Level_4')
net.cluster(views=[])
net.widget()




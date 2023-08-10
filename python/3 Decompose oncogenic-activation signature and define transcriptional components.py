from notebook_environment import *


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

# Load CCLE object
with gzip.open('../data/ccle.pickle.gz') as f:
    CCLE = pickle.load(f)

# Load kras signature genes file
kras_relevant_genes = pd.read_table(
    '../output/kras_relevant_genes.txt', squeeze=True)

# Select kras signature gene rows from CCLE gene expression
rpkm__kras_relevant_gene_x_ccle_cellline = CCLE['Gene Expression']['df'].loc[
    kras_relevant_genes, :]

# Print CCLE gene expression of kras signature genes
rpkm__kras_relevant_gene_x_ccle_cellline

# Drop columns with only 1 unique object
df = ccal.drop_df_slices(
    rpkm__kras_relevant_gene_x_ccle_cellline, 0, max_n_unique_objects=1)

# Drop rows with only 1 unique object
df = ccal.drop_df_slices(df, 1, max_n_unique_objects=1)

# Rank normalize
array_2d = ccal.normalize_2d_array(df.values, 'rank', axis=0)

# Scale
array_2d *= 10000

# Convert to DataFrame
rpkm__kras_relevant_gene_x_ccle_cellline = pd.DataFrame(
    array_2d, index=df.index, columns=df.columns)

# nmfs, nmfccs, cccs = ccal.define_components(
#     rpkm__kras_relevant_gene_x_ccle_cellline,
#     range(2, 11),
#     '../output/nmfccs',
#     algorithm='ls',
#     n_clusterings=30,
#     n_jobs=9,
#     random_seed=6137)

w_matrix = pd.read_table('../output/nmfccs/nmf_k9_w.txt', index_col=0)
h_matrix = pd.read_table('../output/nmfccs/nmf_k9_h.txt', index_col=0)

# Re-label components to have same names as in the paper
indices = [
    'C1',
    'C3',
    'C9',
    'C8',
    'C6',
    'C7',
    'C5',
    'C2',
    'C4',
]

w_matrix.columns = indices
h_matrix.index = indices

w_matrix.to_csv('../output/nmfccs/nmf_k9_w.txt', sep='\t')
h_matrix.to_csv('../output/nmfccs/nmf_k9_h.txt', sep='\t')

ccal.plot_nmf(w_matrix, h_matrix)

ccal.make_comparison_panel(
    h_matrix,
    h_matrix,
    axis=1,
    array_2d_0_name='Component',
    array_2d_1_name='Component')


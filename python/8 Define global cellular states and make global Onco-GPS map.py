from notebook_environment import *


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

h_matrix = pd.read_table('../output/nmfccs/nmf_k9_h.txt', index_col=0)

ccal.plot_heatmap(
    h_matrix,
    normalization_method='-0-',
    normalization_axis=1,
    cluster=True,
    xlabel='Sample',
    ylabel='KRAS Component',
    xticklabels=False)

# '-0-' normalize each row
a = ccal.normalize_2d_array(h_matrix.values, '-0-', axis=1)

# Clip values 3 standard deviation away from each row
a = a.clip(min=-3, max=3)

# '0-1' normalize each row
a = ccal.normalize_2d_array(a, '0-1', axis=1)

h_matrix = pd.DataFrame(a, index=h_matrix.index, columns=h_matrix.columns)

# ds, hcs, cs, cccs = ccal.define_states(
#     h_matrix,
#     range(5, 17),
#     n_clusterings=30,
#     random_seed=830574,
#     directory_path='../output/global_hccs')

cs = pd.read_table('../output/global_hccs/hccs.txt', index_col=0)

sample_states = cs.loc['K15']

component_names = [
    'C1 ERBB3/PI3K',
    'C3 RAS/WNT/PI3K',
    'C9 KRAS/AP1',
    'C8 MYC',
    'C6 BRAF/MAPK',
    'C7 TNF/NFkB',
    'C5 HNF1/PAX8',
    'C2 MYC/E2F',
    'C4 EMT',
]

colors = [
    '#E74C3C',
    '#FFD700',
    '#4B0082',
    '#993300',
    '#4169E1',
    '#90EE90',
    '#F4BD60',
    '#8B008B',
    '#FA8072',
    '#B0E0E6',
    '#20D9BA',
    '#DA70D6',
    '#D2691E',
    '#DC143C',
    '#2E8B57',
]

ccal.plot_heatmap(
    h_matrix,
    normalization_method='-0-',
    normalization_axis=1,
    data_type='continuous',
    annotation_colors=colors,
    column_annotation=sample_states,
    title='H-Matrix Clustering by State',
    xlabel='Sample',
    ylabel='Component',
    xticklabels=False)

gps_map = ccal.GPSMap(h_matrix, pull_power=2, mds_random_seed=310990)

gps_map.set_sample_phenotypes(
    sample_states,
    phenotype_type='categorical',
    bandwidth_factor=5,
    phenotype_color_map=mpl.colors.ListedColormap(colors),
    phenotype_to_str={i: 'State {}'.format(i + 1)
                      for i in range(15)}, )

gps_map.plot_samples_with_phenotype()

gps_map._ax = None
gps_map.save('../output/global_gps_map.pickle.gz')


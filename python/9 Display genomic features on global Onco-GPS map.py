from notebook_environment import *


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

gps_map = ccal.GPSMap.load('../output/global_gps_map.pickle.gz')

gps_map.plot_samples_with_phenotype()

with gzip.open('../data/ccle.pickle.gz') as f:
    CCLE = pickle.load(f)

sorted(CCLE.keys())

annotation = 'AIGNER_ZEB1_TARGETS'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Set']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Set']['data_type']],
    annotation_max_stds=[3],
    title='ZEB1 Targets',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'HINATA_NFKB_TARGETS_FIBROBLAST_UP'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Set']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Set']['data_type']],
    annotation_max_stds=[3],
    title='NFKB Activation',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'lung'

gps_map.plot_samples_with_annotations(
    CCLE['Primary Site']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Primary Site']['data_type']],
    title='Lung Cancers',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'ovary'

gps_map.plot_samples_with_annotations(
    CCLE['Primary Site']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Primary Site']['data_type']],
    title='Ovarian Cancers',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'ERBB3'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Expression']['data_type']],
    annotation_max_stds=[3],
    title='ERBB3 Expression',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'AXL'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Expression']['data_type']],
    annotation_max_stds=[3],
    title='AXL Expression',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'E-Cadherin-R-V'

gps_map.plot_samples_with_annotations(
    CCLE['Protein Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Protein Expression']['data_type']],
    annotation_max_stds=[3],
    title='E-Cadherin',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'HER3-R-V'

gps_map.plot_samples_with_annotations(
    CCLE['Protein Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Protein Expression']['data_type']],
    annotation_max_stds=[3],
    title='HER3',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'KRAS_MUT'

gps_map.plot_samples_with_annotations(
    CCLE['Mutation']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Mutation']['data_type']],
    annotation_max_stds=[3],
    title='KRAS Mutation',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'EGFR_MUT'

gps_map.plot_samples_with_annotations(
    CCLE['Mutation']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Mutation']['data_type']],
    annotation_max_stds=[3],
    title='EGFR Mutation',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'SOX10'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Dependency (Achilles)']['df'].loc[[annotation],
                                                 gps_map._samples].T,
    [CCLE['Gene Dependency (Achilles)']['data_type']],
    annotation_max_stds=[3],
    title='SOX10 Dependency',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'CTNNB1'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Dependency (Achilles)']['df'].loc[[annotation],
                                                 gps_map._samples].T,
    [CCLE['Gene Dependency (Achilles)']['data_type']],
    annotation_max_stds=[3],
    title='CTNNB1 Dependency',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))

annotation = 'PLX-4720'

gps_map.plot_samples_with_annotations(
    CCLE['Drug Sensitivity (CTD^2)']['df'].loc[[annotation], gps_map._samples]
    .T, [CCLE['Drug Sensitivity (CTD^2)']['data_type']],
    annotation_max_stds=[3],
    title='PLX-4720 BRAF Inhibitor',
    violin_or_box='box',
    file_path='../output/annotated_global_onco_gps_maps/{}'.format(annotation))


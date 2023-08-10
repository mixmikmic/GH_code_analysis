from notebook_environment import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

gps_map = ccal.GPSMap.load('../output/kras_mutant_gps_map.pickle.gz')

gps_map.plot_samples_with_phenotype()

# Read CCLE object
with gzip.open('../data/ccle.pickle.gz') as f:
    CCLE = pickle.load(f)

annotation = 'AIGNER_ZEB1_TARGETS'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Set']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Set']['data_type']],
    annotation_max_stds=[3],
    title='ZEB1 Targets',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'HINATA_NFKB_TARGETS_FIBROBLAST_UP'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Set']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Set']['data_type']],
    annotation_max_stds=[3],
    title='NFKB Activation',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'lung'

gps_map.plot_samples_with_annotations(
    CCLE['Primary Site']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Primary Site']['data_type']],
    title='Onco-GPS for Lung Cancers',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'large_intestine'

gps_map.plot_samples_with_annotations(
    CCLE['Primary Site']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Primary Site']['data_type']],
    title='Colon Cancers',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'ERBB3'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Expression']['data_type']],
    annotation_max_stds=[3],
    title='ERBB3 Expression',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'MET'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Expression']['data_type']],
    annotation_max_stds=[3],
    title='MET Expression',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'EGFR'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Expression']['data_type']],
    annotation_max_stds=[3],
    title='EGFR Expression',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'CD274'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Gene Expression']['data_type']],
    annotation_max_stds=[3],
    title='CD274 (PD-L1) Expression',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'E-Cadherin-R-V'

gps_map.plot_samples_with_annotations(
    CCLE['Protein Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Protein Expression']['data_type']],
    annotation_max_stds=[3],
    title='E-Cadherin',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'HER3-R-V'

gps_map.plot_samples_with_annotations(
    CCLE['Protein Expression']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Protein Expression']['data_type']],
    annotation_max_stds=[3],
    title='HER3',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'EGFR_MUT'

gps_map.plot_samples_with_annotations(
    CCLE['Mutation']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Mutation']['data_type']],
    annotation_max_stds=[3],
    title='EGFR Mutation',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'GNAS_MUT'

gps_map.plot_samples_with_annotations(
    CCLE['Mutation']['df'].loc[[annotation], gps_map._samples].T,
    [CCLE['Mutation']['data_type']],
    annotation_max_stds=[3],
    title='GNAS Mutation',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'CTNNB1'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Dependency (Achilles)']['df'].loc[[annotation],
                                                 gps_map._samples].T,
    [CCLE['Gene Dependency (Achilles)']['data_type']],
    annotation_max_stds=[3],
    title='Beta-Catenin Dependency',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'GNAS'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Dependency (Achilles)']['df'].loc[[annotation],
                                                 gps_map._samples].T,
    [CCLE['Gene Dependency (Achilles)']['data_type']],
    annotation_max_stds=[3],
    title='GNAS Dependency',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'KRAS'

gps_map.plot_samples_with_annotations(
    CCLE['Gene Dependency (Achilles)']['df'].loc[[annotation],
                                                 gps_map._samples].T,
    [CCLE['Gene Dependency (Achilles)']['data_type']],
    annotation_max_stds=[3],
    title='KRAS Dependency',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'avicin D'

gps_map.plot_samples_with_annotations(
    CCLE['Drug Sensitivity (CTD^2)']['df'].loc[[annotation], gps_map._samples]
    .T, [CCLE['Drug Sensitivity (CTD^2)']['data_type']],
    annotation_max_stds=[3],
    title='Avicin D',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))

annotation = 'niclosamide'

gps_map.plot_samples_with_annotations(
    CCLE['Drug Sensitivity (CTD^2)']['df'].loc[[annotation], gps_map._samples]
    .T, [CCLE['Drug Sensitivity (CTD^2)']['data_type']],
    annotation_max_stds=[3],
    title='niclosamide',
    violin_or_box='box',
    file_path='../output/annotated_kras_onco_gps_maps/{}_onco_gps'.format(
        annotation))


from notebook_environment import *


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

h_matrix = pd.read_table('../output/nmfccs/nmf_k9_h.txt', index_col=0)

with gzip.open('../data/ccle.pickle.gz') as f:
    CCLE = pickle.load(f)

for i, component in h_matrix.iterrows():

    for features_name, d in CCLE.items():

        features_ = d['df']
        emphasis = d['emphasis']
        data_type = d['data_type']

        print('Annotating with {} (emphasis={} & data_type={})'.format(
            features_name, emphasis, data_type))

#         ccal.make_match_panel(
#             component,
#             features_,
#             n_jobs=28,
#             n_features=20,
#             n_samplings=0,
#             n_permutations=0,
#             scores_ascending=[True, False][emphasis == 'high'],
#             features_type=data_type,
#             title=features_name,
#             file_path_prefix='../output/match_components/match_{}_and_{}'.
#             format(i, features_name))

#         mpl.pyplot.show()

# Select component
component = 'C3'
target = h_matrix.loc[component, :]
target.name = 'KRAS Component {}'.format(component)

# Set up multiple features
CCLE['Mutation']['indices'] = ['KRAS_MUT', 'APC_MUT', 'CTNNB1_MUT']
CCLE['Mutation']['index_aliases'] = ['KRAS', 'APC', 'CTNNB1']

CCLE['Gene Set']['indices'] = ['SINGH_KRAS_DEPENDENCY_SIGNATURE_']
CCLE['Gene Set']['index_aliases'] = ['KRAS Dependency']

CCLE['Gene Dependency (Achilles)']['indices'] = ['KRAS', 'CTNNB1']
CCLE['Gene Dependency (Achilles)']['index_aliases'] = ['KRAS', 'CTNNB1']

multiple_features = OrderedDict({
    k: CCLE[k]
    for k in ['Mutation', 'Gene Set', 'Gene Dependency (Achilles)']
})

# Make summary match panel
ccal.make_summary_match_panel(
    target,
    multiple_features,
    title='Selected Features for {}'.format(target.name),
    file_path='../output/match_components/{}.summary_match_panel.png'.format(
        target.name))

# Select component
component = 'C6'
target = h_matrix.loc[component, :]
target.name = 'KRAS Component {}'.format(component)

# Set up multiple features
CCLE['Mutation']['indices'] = ['BRAF.V600E_MUT']
CCLE['Mutation']['index_aliases'] = ['BRAF V600E']

CCLE['Gene Set']['indices'] = ['BRAF_UP', 'ETV1_UP']
CCLE['Gene Set']['index_aliases'] = [
    'BRAF Oncogenic Signature', 'ETV1 Oncogenic Signature'
]

CCLE['Drug Sensitivity (CTD^2)']['indices'] = [
    'PLX-4720', 'selumetinib', 'PD318088'
]
CCLE['Drug Sensitivity (CTD^2)']['index_aliases'] = [
    'PLX4720 (BRAF Inhibitor)', 'Selumetinib (MEK1 and MEK2 Inhibitor)',
    'PD318088 (MEK1 and MEK2 Inhibitor)'
]

multiple_features = OrderedDict(
    {k: CCLE[k]
     for k in ['Mutation', 'Gene Set', 'Drug Sensitivity (CTD^2)']})

# Make summary match panel
ccal.make_summary_match_panel(
    target,
    multiple_features,
    title='Selected Features for {}'.format(target.name),
    file_path='../output/match_components/{}.summary_match_panel.png'.format(
        target.name))

# Select component
component = 'C7'
target = h_matrix.loc[component, :]
target.name = 'KRAS Component {}'.format(component)

# Set up multiple features
CCLE['Gene Expression']['indices'] = ['FOSL1']
CCLE['Gene Expression']['index_aliases'] = ['FOSL1']

CCLE['Gene Set']['indices'] = ['HINATA_NFKB_TARGETS_FIBROBLAST_UP']
CCLE['Gene Set']['index_aliases'] = ['Genes Up-Regulated by p50 and p65']

CCLE['Regulator Gene Set']['indices'] = ['GGGNNTTTCC_V$NFKB_Q6_01', 'V$AP1_Q4']
CCLE['Regulator Gene Set']['index_aliases'] = [
    'NFKB Transcription Factor Targets', 'AP1 Transcription Factor Targets'
]

CCLE['Protein Expression']['indices'] = [
    'NF-kB-p65_pS536-R-C', 'FRA1_pS265-R-E'
]
CCLE['Protein Expression']['index_aliases'] = ['NFKB p65 pS536', 'FRA1 pS265']

multiple_features = OrderedDict({
    k: CCLE[k]
    for k in [
        'Gene Expression', 'Gene Set', 'Regulator Gene Set',
        'Protein Expression'
    ]
})

# Make summary match panel
ccal.make_summary_match_panel(
    target,
    multiple_features,
    title='Selected Features for {}'.format(target.name),
    file_path='../output/match_components/{}.summary_match_panel.png'.format(
        target.name))

# Select component
component = 'C4'
target = h_matrix.loc[component, :]
target.name = 'KRAS Component {}'.format(component)

# Set up multiple features
CCLE['Gene Set']['indices'] = ['TAUBE_EMT_UP', 'GROGER_EMT_UP']
CCLE['Gene Set']['index_aliases'] = [
    'EMT Inducing Transcription Factors', 'EMT Gene Set'
]

CCLE['Regulator Gene Set']['indices'] = ['V$AREB6_03', 'IPA_ZEB1']
CCLE['Regulator Gene Set']['index_aliases'] = [
    'Targets of TCF8', 'Targets of ZEB1'
]

CCLE['Protein Expression']['indices'] = ['N-Cadherin-R-V', 'E-Cadherin-R-V']
CCLE['Protein Expression']['index_aliases'] = ['N-Cadherin', 'E-Cadherin']

multiple_features = OrderedDict({
    k: CCLE[k]
    for k in ['Gene Set', 'Regulator Gene Set', 'Protein Expression']
})

# Make summary match panel
ccal.make_summary_match_panel(
    target,
    multiple_features,
    title='Selected Features for {}'.format(target.name),
    file_path='../output/match_components/{}.summary_match_panel.png'.format(
        target.name))

# Select component
component = 'C2'
target = h_matrix.loc[component, :]
target.name = 'KRAS Component {}'.format(component)

# Set up multiple features
CCLE['Regulator Gene Set']['indices'] = [
    'V$E2F_02', 'V$MAX_01', 'V$MYCMAX_01', 'IPA_MYC'
]

CCLE['Regulator Gene Set']['index_aliases'] = [
    'Targets of E2F', 'Targets of MAX', 'Targets of MYC and MAX',
    'Targets of MYC'
]

multiple_features = OrderedDict({k: CCLE[k] for k in ['Regulator Gene Set']})

# Make summary match panel
ccal.make_summary_match_panel(
    target,
    multiple_features,
    title='Selected Features for {}'.format(target.name),
    file_path='../output/match_components/{}.summary_match_panel.png'.format(
        target.name))

# Select component
component = 'C5'
target = h_matrix.loc[component, :]
target.name = 'KRAS Component {}'.format(component)

# Set up multiple features
CCLE['Gene Expression']['indices'] = ['PAX8', 'HNF1B']
CCLE['Gene Expression']['index_aliases'] = ['PAX8', 'HNF1B']

CCLE['Gene Dependency (Achilles)']['indices'] = ['PAX8', 'HNF1B']
CCLE['Gene Dependency (Achilles)']['index_aliases'] = ['PAX8', 'HNF1B']

multiple_features = OrderedDict(
    {k: CCLE[k]
     for k in ['Gene Expression', 'Gene Dependency (Achilles)']})

# Make summary match panel
ccal.make_summary_match_panel(
    target,
    multiple_features,
    title='Selected Features for {}'.format(target.name),
    file_path='../output/match_components/{}.summary_match_panel.png'.format(
        target.name))


from notebook_environment import *


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

# Unzip data
ccal.unzip('../data/CTRPv2.2_2015_pub_CancerDisc_5_1210.zip')
ccal.unzip('../data/gene_set__gene_set_x_ccle_cellline.gct.zip')
ccal.unzip('../data/CCLE_MUT_EXPR_RPPA_OncoGPS.zip')

# Rename Achilles RNAi dataset
df = pd.read_csv('../data/ExpandedGeneZSolsCleaned.csv', index_col=0)
ccal.write_gct(df, '../data/achilles__gene_x_ccle_cellline.gct')

# Read compound data
auc = pd.read_table('../data/v22.data.auc_sensitivities.txt')
print(auc.shape)

cpd = pd.read_table('../data/v22.meta.per_compound.txt', index_col=0)
print(cpd.shape)

ccl = pd.read_table('../data/v22.meta.per_cell_line.txt', index_col=0)
print(ccl.shape)

# Make dict for faster ID-to-name look up
cpd_d = cpd['cpd_name'].to_dict()
ccl_d = ccl['ccl_name'].to_dict()

# Make empty compound-x-cellline matrix
compound_x_cellline = pd.DataFrame(
    index=sorted(set(cpd['cpd_name'])), columns=sorted(set(ccl['ccl_name'])))
print(compound_x_cellline.shape)

# Populate compound-x-cellline matrix
for i, (i_cpd, i_ccl, a) in auc.iterrows():

    # Get compound name
    cpd_n = cpd_d[i_cpd]

    # Get cellline name
    ccl_n = ccl_d[i_ccl]

    # Get current AUC
    a_ = compound_x_cellline.loc[cpd_n, ccl_n]

    # If the current AUC is not set, set with this AUC
    if pd.isnull(a_):
        compound_x_cellline.loc[cpd_n, ccl_n] = a

    # If this AUC is smaller than the current AUC, set with this AUC
    elif a < a_:

        print('Updating AUC of compound {} on cellline {}: {:.3f} ==> {:.3f}'.
              format(cpd_n, ccl_n, a_, a))

        compound_x_cellline.loc[cpd_n, ccl_n] = a

# Update cellline names to match CCLE cellline names
columns = list(compound_x_cellline.columns)

# Read CCLE cellline annotations
a = pd.read_table('../data/CCLE_sample_info_file_2012-10-18.txt', index_col=0)

# Get CCLE cellline names
for i, ccl_n in enumerate(compound_x_cellline.columns):

    matches = []

    for ccle_n in a.index:
        if ccl_n.lower() == ccle_n.lower().split('_')[0]:
            matches.append(ccle_n)

    if 0 == len(matches):
        print('0 match: {}; matching substring ...'.format(ccl_n))

        for ccle_n in a.index:

            if ccl_n.lower() in ccle_n.lower():

                print('\t{} ==> {}.'.format(ccl_n, ccle_n))
                matches.append(ccle_n)

    if 1 == len(matches):

        print('{} ==> {}.'.format(ccl_n, matches[0]))
        columns[i] = matches[0]

    else:
        print('1 < matches: {} ==> {}'.format(ccl_n, matches))

# Update with CCLE cellline names
compound_x_cellline.columns = columns

# Write .gct file
ccal.write_gct(compound_x_cellline,
               '../data/ctd2__compound_x_ccle_cellline.gct')

compound_x_cellline

for fn in [
        'gene_x_kras_isogenic_and_imortalized_celllines.gct',
        'mutation__gene_x_ccle_cellline.gct',
        'rpkm__gene_x_ccle_cellline.gct',
        'gene_set__gene_set_x_ccle_cellline.gct',
        'regulator__gene_set_x_ccle_cellline.gct',
        'rppa__protein_x_ccle_cellline.gct',
        'achilles__gene_x_ccle_cellline.gct',
        'ctd2__compound_x_ccle_cellline.gct',
        'annotation__feature_x_ccle_cellline.gct',
]:
    assert fn in os.listdir('../data/'), 'Missing {}!'.format(fn)

# Make the CCLE data object used in coming chapters.

ccle = {
    'Mutation': {
        'df': ccal.read_gct('../data/mutation__gene_x_ccle_cellline.gct'),
        'emphasis': 'high',
        'data_type': 'binary'
    },
    'Gene Expression': {
        'df': ccal.read_gct('../data/rpkm__gene_x_ccle_cellline.gct'),
        'emphasis': 'high',
        'data_type': 'continuous'
    },
    'Gene Set': {
        'df': ccal.read_gct('../data/gene_set__gene_set_x_ccle_cellline.gct'),
        'emphasis': 'high',
        'data_type': 'continuous'
    },
    'Regulator Gene Set': {
        'df': ccal.read_gct('../data/regulator__gene_set_x_ccle_cellline.gct'),
        'emphasis': 'high',
        'data_type': 'continuous'
    },
    'Protein Expression': {
        'df': ccal.read_gct('../data/rppa__protein_x_ccle_cellline.gct'),
        'emphasis': 'high',
        'data_type': 'continuous'
    },
    'Gene Dependency (Achilles)': {
        'df': ccal.read_gct('../data/achilles__gene_x_ccle_cellline.gct'),
        'emphasis': 'low',
        'data_type': 'continuous'
    },
    'Drug Sensitivity (CTD^2)': {
        'df': ccal.read_gct('../data/ctd2__compound_x_ccle_cellline.gct'),
        'emphasis': 'low',
        'data_type': 'continuous'
    },
    'Primary Site': {
        'df':
        ccal.make_membership_df_from_categorical_series(
            ccal.read_gct('../data/annotation__feature_x_ccle_cellline.gct')
            .loc['Site Primary']),
        'emphasis':
        'high',
        'data_type':
        'binary'
    }
}

with gzip.open('../data/ccle.pickle.gz', 'wb') as f:

    pickle.dump(ccle, f)


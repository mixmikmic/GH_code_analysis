from notebook_environment import *


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

with gzip.open('../data/ccle.pickle.gz') as f:
    CCLE = pickle.load(f)

cs = pd.read_table('../output/hccs/hccs.txt', index_col=0)

kras_sample_labels = cs.loc['K4']

kras_sample_labels.name = 'State'

state_x_sample = ccal.make_membership_df_from_categorical_series(
    kras_sample_labels)
state_x_sample.index = ['State {}'.format(i + 1) for i in state_x_sample.index]

state_x_sample

for i, state in state_x_sample.iterrows():

    for features_name, d in CCLE.items():

        features_ = d['df']
        emphasis = d['emphasis']
        data_type = d['data_type']

        print('Annotating with {} (emphasis={} & data_type={})'.format(
            features_name, emphasis, data_type))

#         ccal.make_match_panel(
#             state,
#             features_,
#             n_jobs=16,
#             n_features=20,
#             n_samplings=3,
#             n_permutations=3,
#             scores_ascending=[True, False][emphasis == 'high'],
#             features_type=data_type,
#             title=features_name,
#             file_path_prefix='../output/match_states/match_{}_and_{}'.format(
#                 i, features_name))

#         mpl.pyplot.show()


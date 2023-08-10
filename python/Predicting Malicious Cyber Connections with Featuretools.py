import featuretools as ft
from featuretools.primitives import CumMean, Percentile
from featuretools.selection import remove_low_information_features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import utils

cyber_df = pd.read_csv("CyberFLTenDays.csv")
cyber_df.index.name = "log_id"
cyber_df.reset_index(inplace=True, drop=False)
cyber_df['label'] = cyber_df['label'].map({'N': False, 'A': True}, na_action='ignore')

# Sample down negative examples because very few positives
# Can also do this after the feature engineering step (but doing it here reduces computation time)
cyber_df_pos = cyber_df[cyber_df['label']]
cyber_df_neg = cyber_df[~cyber_df['label']].sample(100000)
cyber_df = pd.concat([cyber_df_pos, cyber_df_neg]).sort_values(['secs'])

cyber_df.head()

es = ft.EntitySet("CyberLL")
# create an index column
cyber_df["name_host_pair"] = cyber_df["src_name"].str.cat(
                                [cyber_df["dest_name"],
                                 cyber_df["src_host"],
                                 cyber_df["dest_host"]],
                                sep=' / ')
cyber_df["session_id"] = cyber_df["src_name"].str.cat(
                                 cyber_df["dest_name"],
                                 sep=' / ')

es.entity_from_dataframe("log",
                         cyber_df,
                         index="log_id",
                         time_index="secs")
es.normalize_entity(base_entity_id="log",
                    new_entity_id="name_host_pairs",
                    index="name_host_pair",
                    additional_variables=["src_name", "dest_name",
                                          "src_host", "dest_host",
                                          #"src_pair",
                                          #"dest_pair",
                                          "session_id",
                                          "label"])
es.normalize_entity(base_entity_id="name_host_pairs",
                    new_entity_id="sessions",
                    index="session_id",
                    additional_variables=["dest_name", "src_name"])

cyber_df.head()

def generate_cutoffs(cyber_df, index_col, after_n_obs, lead, prediction_window):
    window_start = after_n_obs + lead
    window_end = window_start + prediction_window
    grouped = cyber_df.groupby(index_col)[index_col].count()
    grouped.name = "count"
    min_obs = after_n_obs + lead + 1
    enough_examples = grouped[grouped > min_obs].to_frame().reset_index()
    enough_examples = cyber_df[cyber_df[index_col].isin(enough_examples[index_col])]
    def get_label_and_cutoff(df):
        cutoff = df.iloc[after_n_obs]
        cutoff['label'] = df.iloc[window_start: window_end]["label"].any()
        return cutoff
    cutoffs = enough_examples.groupby(index_col)[[index_col, "secs", "label"]].apply(get_label_and_cutoff)
    return cutoffs

cutoffs = generate_cutoffs(cyber_df, "session_id", 2, 2, 10)

cutoffs['label'].value_counts()

fm, fl = ft.dfs(entityset=es, target_entity="name_host_pairs", cutoff_time=cutoffs,
                cutoff_time_in_index=True,
                verbose=True, max_depth=3)

fm.head()

fm = fm.reorder_levels(['time', 'session_id']).sort_index()
cutoffs = cutoffs.set_index('secs', append=True).reorder_levels(['secs', 'session_id']).sort_index()
fm['label'] = cutoffs['label'].values

fm_encoded, fl_encoded = ft.encode_features(fm, fl)
fm_encoded, fl_encoded = remove_low_information_features(fm_encoded, fl_encoded)

train, test = train_test_split(fm_encoded, test_size=0.2, shuffle=True)

X_train = train
y_train = X_train.pop('label')
X_test = test
y_test = X_test.pop('label')

imputer = Imputer(missing_values='NaN', strategy="mean", axis=0)
scaler = StandardScaler()
clf = RandomForestClassifier(n_jobs=-1)
model = Pipeline([("imputer", imputer),
                  ("scaler", scaler),
                  ("rf", clf)])

model.fit(X_train, y_train)
    
preds = model.predict(X_test)
score = roc_auc_score(preds, y_test)
print('ROC AUC Score: {:.2f}'.format(score))

high_imp_feats = utils.feature_importances(X_train, clf, feats=10)


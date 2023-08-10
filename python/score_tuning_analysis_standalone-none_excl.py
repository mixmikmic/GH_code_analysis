from sklearn.externals import joblib
# from search_service import RelevanceSearchService
import numpy as np
import pandas as pd

scores_df = joblib.load("scores_pickled_wout_none.dump")
scores_df.describe()

len(scores_df)

# check of the tuned thresholds for categories

last_split_thresholds = joblib.load("last_split_thrds_wout_none.dump")
last_split_thresholds

# ScoreTuner instance used for evaluation of the results
from dependencies.scores_tuner import ScoreTuner
eval_score_tuner = ScoreTuner()
eval_score_tuner.cats_original_thresholds = last_split_thresholds

eval_score_tuner.precision_recall_for_category(scores_df["y"], scores_df["eap"], "eap", 0.5)

# true scores distribution for categories: 1. before tuning, 2. after tuning
true_docs_scores_tuned = pd.DataFrame(columns=["cat", "score"])

for cat in set(scores_df.columns)-{"y"}:
    new_scores = scores_df.ix[scores_df["y"]==cat, cat]
    new_cat = [cat]*len(new_scores)
    new_df = pd.DataFrame()
    new_df["score"] = new_scores
    new_df["cat"] = new_cat
    true_docs_scores_tuned = true_docs_scores_tuned.append(new_df)

get_ipython().magic('matplotlib inline')
true_docs_scores_tuned.boxplot("score", by="cat", figsize=(10,5), rot=60)

# TODO; true scores distribution for categories: 1. before tuning, 2. after tuning
false_docs_scores_tuned = pd.DataFrame(columns=["cat", "score"])

for cat in set(scores_df.columns)-{"y"}:
    new_scores = scores_df.ix[scores_df["y"]!=cat, cat]
    new_cat = [cat]*len(new_scores)
    new_df = pd.DataFrame()
    new_df["score"] = new_scores
    new_df["cat"] = new_cat
    false_docs_scores_tuned = false_docs_scores_tuned.append(new_df)

get_ipython().magic('matplotlib inline')
false_docs_scores_tuned.boxplot("score", by="cat", figsize=(10,5), rot=60)

false_docs_scores_tuned["score"] = false_docs_scores_tuned["score"].apply(lambda x: float(x))

# false_docs_scores_tuned[false_docs_scores_tuned["cat"] == "none"]["cat"] = "None"

# drops off categories with on content in evaluation set if there are any
false_docs_scores_tuned = false_docs_scores_tuned.dropna(subset=["score"])

# negative mean scores for categories
cat_meta = pd.DataFrame(index=false_docs_scores_tuned["cat"].unique())
cat_meta["mean_score"] = false_docs_scores_tuned.groupby(by="cat").apply(np.mean)
cat_meta["cat_size"] = true_docs_scores_tuned.groupby(by="cat").count()
cat_meta.sort_values(by="cat_size")

taken_categories = set(scores_df.columns) - {"none", "mobileplatform"}
scores_df_no_nans = scores_df.loc[scores_df["y"].isin(taken_categories), list(taken_categories)]
scores_df_filtered = scores_df_no_nans[list(taken_categories-{"y"})].applymap(float)
y_filtered = scores_df_no_nans["y"]

cats_betas = eval_score_tuner.beta_for_categories_provider(y_filtered)
categories = pd.Series(scores_df_filtered.columns)

cats_perf = categories.apply(lambda cat: eval_score_tuner.f_score_for_category(y_filtered, 
                                                                                   scores_df_filtered[cat], 
                                                                                   cat,
                                                                                   0.5,
                                                                                   cats_betas[cat]))
cats_perf.index = categories
# particular categories performance
cats_perf

eval_score_tuner.weighted_combine_cats_predictions(y_filtered, cats_perf)

from search_service import RelevanceSearchService

eval_service = RelevanceSearchService()
eval_service.minimized_persistence = True
eval_service.load_trained_model("service_persisted_cv_split_wout_none")

neg_samples_csv = pd.read_csv("../../data/experimental/eval_set/Political-media-DFE.csv")
neg_samples_csv["text"].head()

def take_first_n_words(text, n=5):
    n_words = text.split()[:n]
    return reduce(lambda w_one, w_two: "%s %s" % (w_one, w_two), n_words)

def replace_non_ascii(str):
    return ''.join([i if ord(i) < 128 else ' ' for i in str])

neg_samples_df = pd.DataFrame(columns=["id", "doc_header", "doc_content"])
# TODO: set limit here if needed
neg_samples_df[["id", "doc_content"]] = neg_samples_csv[["_unit_id", "text"]]
neg_samples_df["doc_header"] = neg_samples_df["doc_content"].apply(lambda content: take_first_n_words(content, n=5))
neg_samples_df["doc_content"] = neg_samples_df["doc_content"].apply(replace_non_ascii)
neg_samples_df["doc_header"] = neg_samples_df["doc_header"].apply(replace_non_ascii)
neg_samples_df.head()

neg_samples_scores = eval_service.score_docs_bulk(neg_samples_df["id"], 
                                                  neg_samples_df["doc_header"], 
                                                  neg_samples_df["doc_content"])

get_ipython().magic('matplotlib inline')

scores_norm_flattened = pd.DataFrame(columns=["cat", "score"])

for cat in eval_service.vector_classifier.classes_:
    new_scores = neg_samples_scores[cat]
    new_cat = [cat]*len(new_scores)
    new_df = pd.DataFrame()
    new_df["score"] = new_scores
    new_df["cat"] = new_cat
    scores_norm_flattened = scores_norm_flattened.append(new_df)

scores_norm_flattened.boxplot("score", by="cat", figsize=(10,5), rot=60)

# see that the devstudio scoring is incorrectly normalized upwards, 
# whereas the fuse and eap are learnt to correctly normalize down

eval_service.score_tuner.cats_original_thresholds


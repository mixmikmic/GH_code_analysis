import pandas
import logging
import variant_calling_benchmarks
from variant_calling_benchmarks import analysis
import seaborn
reload(analysis)
reload(variant_calling_benchmarks)

import sklearn.linear_model
import sklearn.cross_validation
import sklearn.ensemble


get_ipython().magic('pylab inline')

pandas.set_option("display.max_columns", 500)
logger = logging.getLogger().setLevel(logging.DEBUG)

'''
# Takes 4 mins to load from cloud:
from variant_calling_benchmarks import analysis
benchmark = "gs://variant-calling-benchmarks-results/manifest.aocs.281c84ca79aca6fd.2b655b4f3eb7bb55.json"
%time df = analysis.load_benchmark_result(benchmark)
print(df.shape)
'''

# About 1 min faster to load locally:
get_ipython().magic('time df = analysis.load_benchmark_result("/Users/tim/sinai/data/gcs/variant-calling-benchmarks-results/manifest.aocs.281c84ca79aca6fd.2b655b4f3eb7bb55.json")')
df

df["site"] = ["%s-%d" % (row.contig, row.interbase_start) for (i,row) in df.iterrows()]
df

site_counts = df.site.value_counts()
duplicate_sites = site_counts.index[site_counts > 1]
duplicate_sites
df["alt_disagreement"] = df.site.isin(set(duplicate_sites))
df

df_snv = df.ix[df.snv].copy()
df_snv.shape

def guacamole_status(row):
    pieces = []
    if row.trigger_GERMLINE_POOLED:
        pieces.append("GERMLINE")
    if row.trigger_SOMATIC_POOLED:
        pieces.append("SOMATIC_POOLED")
    if row.trigger_SOMATIC_INDIVIDUAL:
        pieces.append("SOMATIC_INDIVIDUAL")
    if not pieces:
        pieces.append("NO_TRIGGER")
    if row.alt_disagreement:
        pieces.insert(0, "ALT_DISAGREEMENT")
    if row["filter"]:
        pieces.extend(row["filter"])
    return "-".join(pieces)

get_ipython().magic('time df_snv["guacamole_status"] = [guacamole_status(row) for (i, row) in df_snv.iterrows()]')
df_snv

counts = df_snv.ix[df_snv.called_published].guacamole_status.value_counts()
counts

def move_to_front(series, values):
    desired_order = values + [v for v in list(series.index) if v not in values]
    return series[desired_order]

def add_values_to_series(series):
    result = series.copy()
    for col in series.index:
        result["%s [%s = %0.1f%%]" % (col, series[col], series[col] * 100.0 / series.sum())] = series[col]
        del result[col]
    return result

good_calls = ["SOMATIC_POOLED-SOMATIC_INDIVIDUAL", "SOMATIC_POOLED", "SOMATIC_INDIVIDUAL"]
ordered_counts = move_to_front(counts, good_calls)[::-1]
colors = ["green" if c in good_calls else "red" for c in ordered_counts.index]
plot_counts = add_values_to_series(ordered_counts.fillna(0).astype(int))
seaborn.set_context('talk')
matplotlib.rcParams['ytick.labelsize'] = 18 
matplotlib.rcParams['xtick.labelsize'] = 18 


plot_counts.plot('barh', colors=colors, figsize=(10,10))
#pyplot.axhline(len(ordered_counts) - 3.5, color="red")

counts = df_snv.ix[
    (df_snv.trigger_SOMATIC_POOLED | df_snv.trigger_SOMATIC_INDIVIDUAL) & (~df_snv.trigger_GERMLINE_POOLED)
].groupby(["guacamole_status", "called_published"])["genome"].count().sort(inplace=False).iloc[::-1].to_frame().reset_index()


def make_series(sub_counts):
    sub_counts = sub_counts.copy()
    sub_counts.index = sub_counts.guacamole_status
    sub_counts = move_to_front(sub_counts.genome, good_calls).iloc[::-1]
    sub_counts = add_values_to_series(sub_counts.fillna(0).astype(int))
    return sub_counts
    
counts_good = make_series(counts.ix[counts.called_published])
counts_bad = make_series(counts.ix[~counts.called_published])

counts_good.plot('barh',
                 title="In published",
                 colors=["green" if c.split()[0] in good_calls else "red" for c in counts_good.index],
                 figsize=(10,10))
pyplot.ylabel("")

pyplot.figure()
counts_bad.plot('barh',
                title="NOT in published",
                colors=["green" if c.split()[0] in good_calls else "red" for c in counts_bad.index],
                figsize=(10,10))
pyplot.ylabel("")

pyplot.figure()
counts_bad_zoom = counts_bad.iloc[-3:]
counts_bad_zoom.index = [x.split()[0] for x in counts_bad_zoom.index]
counts_bad_zoom = add_values_to_series(counts_bad_zoom)
counts_bad_zoom.plot('barh',
                title="NOT in published",
                colors=["green" if c.split()[0] in good_calls else "red" for c in counts_bad_zoom.index],
                figsize=(10,5))
pyplot.ylabel("")


pyplot.figure()
counts_good_zoom = counts_good.iloc[-3:]
counts_good_zoom.index = [x.split()[0] for x in counts_good_zoom.index]
counts_good_zoom = add_values_to_series(counts_good_zoom)
counts_good_zoom.plot('barh',
                title="In published",
                colors=["green" if c.split()[0] in good_calls else "red" for c in counts_good_zoom.index],
                figsize=(10,5))
pyplot.ylabel("")

#counts_good.reset_index("guacamole_status").plot('barh')



features = ["AD_alt", "AD_ref", "VAF", "FS", "RL_alt", "RL_ref"]
columns = ["called_published"]
for sample in ["normal_blood", "pooled_tumor"]:
    columns.extend(["%s_%s" % (sample, f) for f in features])
ml_df = df_snv.ix[~df_snv.alt_disagreement][columns].dropna()
ml_df["normal_blood_RL_diff"] = ml_df["normal_blood_RL_alt"] - ml_df["normal_blood_RL_ref"]
ml_df["pooled_tumor_RL_diff"] = ml_df["pooled_tumor_RL_alt"] - ml_df["pooled_tumor_RL_ref"]
ml_df

ml_df.iloc[0]

ml_df.to_csv("aocs_ml_df.csv", index=False)

random_state = 0

X = ml_df[ml_df.columns[1:]].values
y = ml_df.called_published.values

X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=.5, random_state=random_state)

model = sklearn.linear_model.LogisticRegression()
y_score = model.fit(X_train, y_train).predict_proba(X_test)[:,1]

pandas.Series(model.coef_[0], index=list(ml_df.columns)[1:])



y_score

(precision, recall, thresholds) = sklearn.metrics.precision_recall_curve(y_test, y_score)
pyplot.plot(recall, precision)
pyplot.xlabel("recall")
pyplot.ylabel('precision')
pyplot.title("AOCS-034 SJC vs. published logistic regression")

model = sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
y_score = model.fit(X_train, y_train).predict_proba(X_test)[:,1]

(precision, recall, thresholds) = sklearn.metrics.precision_recall_curve(y_test, y_score)
pyplot.plot(recall, precision)
pyplot.xlabel("recall")
pyplot.ylabel('precision')
pyplot.title("AOCS-034 SJC vs. published random forest")

pandas.Series(model.feature_importances_, index=list(ml_df.columns)[1:])[::-1].plot('barh')
pyplot.xlabel("Feature importance")
pyplot.title("AOCS-034 random forest classifier SJC vs. published calls")

random_state = 0

X = ml_df[ml_df.columns[1:]].values
y = ml_df.called_published.values

X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=.5, random_state=random_state)

model = sklearn.tree.DecisionTreeClassifier(max_depth=4)
y_score = model.fit(X_train, y_train).predict_proba(X_test)[:,1]

(precision, recall, thresholds) = sklearn.metrics.precision_recall_curve(y_test, y_score)
pyplot.plot(recall, precision)
pyplot.xlabel("recall")
pyplot.ylabel('precision')
pyplot.title("AOCS-034 SJC vs. published decision tree")

sklearn.tree.export_graphviz(
    model.tree_,
    "aocs034-decision-tree.dot", rotate=True,
    feature_names=list(ml_df.columns)[1:])

get_ipython().system(' dot -Tpng aocs034-decision-tree.dot -o aocs034-decision-tree.png')

get_ipython().magic('pinfo sklearn.tree.export_graphviz')


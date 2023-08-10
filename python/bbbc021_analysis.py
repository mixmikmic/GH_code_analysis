import pandas as pd
from microscopium.screens import image_xpress
import os
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import pylab
import matplotlib.pyplot as plt
import matplotlib
import brewer2mpl

get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# first load in the screen metadata
bbbc021_metadata = pd.read_csv("./BBBC021_v1_image.csv")

# now load in the mechanism of action metadata, these map
# compouds to a class of compounds
bbbc021_moa = pd.read_csv("./BBBC021_v1_moa.csv")

# wrangle the metadata into a form that maps screen-plate-well format IDs
# to the compound it was treated with. this dataset contains no controls or
# empty wells, so we don't need to worry about those!!

# first only keep the colums we want -- 
# Image_FileName_DAPI, Image_PathName_DAPI, Image_Metadata_Compound, Image_Metadata_Concentration
bbbc021_metadata = bbbc021_metadata[["Image_FileName_DAPI",
                                     "Image_PathName_DAPI",
                                     "Image_Metadata_Compound", 
                                     "Image_Metadata_Concentration"]]

def fn_to_id(fn):
    sem = image_xpress.ix_semantic_filename(fn)
    return "{0}-{1}-{2}".format("BBBC021", sem["plate"], sem["well"])

# merge the Image_PathName_DAPI and Image_FileName_DAPI column with os.path.join
fn_cols = zip(bbbc021_metadata["Image_PathName_DAPI"], bbbc021_metadata["Image_FileName_DAPI"])
bbbc021_metadata.index = list(map(fn_to_id, [os.path.join(i, j) for (i, j) in fn_cols]))
bbbc021_metadata = bbbc021_metadata[["Image_Metadata_Compound", "Image_Metadata_Concentration"]]
bbbc021_metadata.head()

# good idea to check that different concentrations don't 
# change the expected mechanism of action in the annotations
bbbc021_moa.groupby(['compound', 'moa']).count()

# now merge the dataframes!
right_cols = ["compound", "concentration"]
bbbc021_merged = bbbc021_metadata.reset_index().merge(
    bbbc021_moa, how="outer", 
    left_on=["Image_Metadata_Compound", "Image_Metadata_Concentration"], 
    right_on=right_cols).set_index("index").dropna().drop_duplicates()

# only a subset of the data was annotated -- 103
# how are the classes distributed?
bbbc021_merged.head()
bbbc021_merged.groupby("moa").count()

# only one example for the "DMSO" class. remove this.
bbbc021_merged = bbbc021_merged[bbbc021_merged["compound"] != "DMSO"]

# now load the feature data frame
bbbc021_complete = pd.read_csv("./BBBC021_feature.csv", index_col=0)
# we only want the feature vectors for samples that were annotated
bbbc021_feature = bbbc021_complete.ix[bbbc021_merged.index]
bbbc021_feature.head()

# Now scale the dataframe and we're good to go!
std = StandardScaler().fit_transform(bbbc021_feature.values)
bbbc021_feature = pd.DataFrame(std, columns=bbbc021_feature.columns, index=bbbc021_feature.index)

classifier = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(classifier, 
                                          bbbc021_feature.values, 
                                          bbbc021_merged["moa"].values,
                                          cv=5)
sum(scores / 5)

object_cols = [col for col in bbbc021_feature.columns if "pftas" not in col and "haralick" not in col]
haralick_cols = [col for col in bbbc021_feature.columns if "haralick" in col]
pftas_cols = [col for col in bbbc021_feature.columns if "pftas" in col]

scores = cross_validation.cross_val_score(classifier, 
                                          bbbc021_feature[object_cols].values, 
                                          bbbc021_merged["moa"].values,
                                          cv=5)
sum(scores / 5)

scores = cross_validation.cross_val_score(classifier, 
                                          bbbc021_feature[haralick_cols].values, 
                                          bbbc021_merged["moa"].values,
                                          cv=5)
sum(scores / 5)

scores = cross_validation.cross_val_score(classifier, 
                                          bbbc021_feature[pftas_cols].values, 
                                          bbbc021_merged["moa"].values,
                                          cv=5)
sum(scores / 5)

object_pftas_cols = [col for col in bbbc021_feature.columns if "haralick" not in col]
scores = cross_validation.cross_val_score(classifier, 
                                          bbbc021_feature[object_pftas_cols].values, 
                                          bbbc021_merged["moa"].values,
                                          cv=5)
sum(scores / 5)

et_classifier = ExtraTreesClassifier()
et_classifier.fit(bbbc021_feature[object_pftas_cols].values, bbbc021_merged["moa"].values)

feature_scores = pd.DataFrame(data={"feature": bbbc021_feature[object_pftas_cols].columns, 
                                    "gini": et_classifier.feature_importances_})
feature_scores = feature_scores.sort_values(by="gini", ascending=False)

top_k = 30

plt.barh(np.arange(top_k), feature_scores.head(top_k)["gini"], align="center", alpha=0.4)
plt.ylim([top_k, -1])
plt.yticks(np.arange(top_k), feature_scores.head(30)["feature"])
plt.tight_layout()
plt.title("Feature Importance for Annotated BBBC021 Data Subset")
plt.xlabel("Gini Coefficient")
plt.show()

n_features = bbbc021_feature[object_pftas_cols].shape[1]
sample_size = int(np.round(n_features * 0.65))

all_scores = []
for i in range(10000):
    random_index = np.random.choice(np.arange(n_features), sample_size)
    scores = cross_validation.cross_val_score(classifier, 
                                          bbbc021_feature[object_pftas_cols][random_index].values, 
                                          bbbc021_merged["moa"].values,
                                          cv=5)
    cv_score = sum(scores / 5)
    all_scores.append(cv_score)

pd.DataFrame(all_scores).describe()

ag_clustering = AgglomerativeClustering(n_clusters=12, affinity="cosine", linkage="complete")
ag_predict = ag_clustering.fit_predict(X=bbbc021_feature[object_pftas_cols].values)

metrics.adjusted_rand_score(bbbc021_merged["moa"].values, ag_predict)

rand_seed = 42

bbbc021_pca = PCA(n_components=2).fit_transform(bbbc021_feature[object_pftas_cols].values)
bbbc021_pca_50 = PCA(n_components=50).fit_transform(bbbc021_feature[object_pftas_cols].values)
bbbc021_tsne = TSNE(n_components=2, learning_rate=100, random_state=42).fit_transform(bbbc021_pca_50)

labels = list(set(bbbc021_merged["moa"]))
bmap = brewer2mpl.get_map("Paired", "Qualitative", 12)
color_scale = dict(zip(labels, bmap.mpl_colors))

bbbc021_pca_df = pd.DataFrame(dict(x=bbbc021_pca[:, 0], 
                              y=bbbc021_pca[:, 1], 
                              label=bbbc021_merged["moa"].values), 
                              index=bbbc021_feature.index)
groups = bbbc021_pca_df.groupby('label')

fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    ax.scatter(group.x, group.y, s=45, label=name, c=color_scale[name])
ax.legend(scatterpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
plt.title("PCA")

fig = plt.gcf()
fig.subplots_adjust(bottom=0.2)

plt.show()

bbbc021_tsne_df = pd.DataFrame(dict(x=bbbc021_tsne[:, 0], 
                                    y=bbbc021_tsne[:, 1], 
                                    label=bbbc021_merged["moa"].values), 
                               index=bbbc021_feature.index)
groups = bbbc021_tsne_df.groupby('label')

fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    ax.scatter(group.x, group.y, s=45, label=name, c=color_scale[name])
ax.legend(scatterpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
plt.title("TSNE")

fig = plt.gcf()
fig.subplots_adjust(bottom=0.2)

plt.show()

np.random.seed(13)

# get the set difference of indices in the whole dataset, and indices 
unannot_index = np.setdiff1d(bbbc021_complete.index, bbbc021_feature.index)

# get 20 random examples from the data frame
unannot_sample = np.random.choice(unannot_index, 10)

# combine these samples with the annotated ones, rescale the data-frame
bbbc021_new = bbbc021_complete.ix[bbbc021_feature.index | unannot_sample]


bbbc021_new_std = StandardScaler().fit_transform(bbbc021_new.values)
bbbc021_new = pd.DataFrame(bbbc021_new_std, columns=bbbc021_new.columns, index=bbbc021_new.index)

# embed to tsne
bbbc021_new_pca_50 = PCA(n_components=50).fit_transform(bbbc021_new.values)
bbbc021_new_tsne = TSNE(n_components=2, learning_rate=45, random_state=rand_seed).fit_transform(bbbc021_new_pca_50)

bbbc021_new_tsne_df = pd.DataFrame(dict(x=bbbc021_new_tsne[:, 0], 
                                        y=bbbc021_new_tsne[:, 1]), 
                                   index=bbbc021_new.index)

# add moa labels
bbbc021_new_tsne_df = bbbc021_new_tsne_df.merge(bbbc021_merged, 
                                                how="outer", 
                                                left_index=True, 
                                                right_index=True)

groups = bbbc021_new_tsne_df.fillna("No Annotation").groupby('moa')

fig, ax = plt.subplots()
ax.margins(0.05)

for name, group in groups:
    color = color_scale.get(name)
    if color is None:
        color = (1, 1, 1)
    ax.scatter(group.x, group.y, s=45, label=name, c=color)

ax.legend(scatterpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.05),
      fancybox=True, shadow=True, ncol=4)
plt.title("TSNE")

for idx in unannot_sample:
    row = bbbc021_new_tsne_df.ix[idx]
    
    plt.annotate(
        idx, 
        xy = (row[0], row[1]), xytext = (0, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
fig = plt.gcf()
fig.subplots_adjust(bottom=0.2)

plt.show()

selected_indices = ["BBBC021-22141-D08", 
                    "BBBC021-25701-C07", 
                    "BBBC021-22161-F07", 
                    "BBBC021-27821-C05",
                    "BBBC021-25681-C09", 
                    "BBBC021-34641-C10"]

bbbc021_metadata.ix[selected_indices].drop_duplicates()


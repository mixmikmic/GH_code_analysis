import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')

food = pd.read_csv('./ready_for_cvec.csv', dtype=object, index_col=0)

food = food.dropna().reset_index(drop=True)

food.shape

#prepping for cosine similarities 
food['without_hashes'] = food['without_hashes'] + ' '
grouped = food.groupby('name', as_index=False)['without_hashes'].sum()
comp_list = [x for x in grouped['without_hashes']]

#vectorizing to get cosine similarities across profiles by company
tfidf = TfidfVectorizer(min_df=1)
vect = tfidf.fit_transform(comp_list)
cosine_similarities = pd.DataFrame((vect * vect.T).A)
cosine_similarities.columns = list(grouped['name'].values)
cosine_similarities.index = list(grouped['name'].values)

#plotting
fig, ax = plt.subplots(figsize=(12,8))
cmap = sns.diverging_palette(10, 220, sep=80, n=7)
ax = sns.heatmap(cosine_similarities, annot=True, linewidths=0.5, vmin=.6, vmax=.9)
plt.title('Customer Similarity', fontsize=20)
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=90)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)

cosine_similarities.mean()

X = vect.toarray()
Z = linkage(X, 'ward')

#values closer to one are good- indicating the groups sit close to one another
c, coph_dists = cophenet(Z, pdist(X))
print(c)

fig, ax = plt.subplots(figsize=(9,5))
plt.style.use('fivethirtyeight')
plt.title('Dendrogram of Company Clusters')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90.,leaf_font_size=14., labels= list(grouped['name'].values));




import pandas as pd

model_df = pd.read_json('data/mag_papers_0/mag_subset20K.txt', lines=True)

model_df.shape

model_df.columns

# filter out non-English articles
# keep abstract, authors, fos, keywords, year, title
model_df = model_df[model_df.lang == 'en'].drop_duplicates(subset = 'title', 
                                                           keep = 'first').drop(['doc_type', 
                                                                                                   'doi', 'id', 
                                                                                                   'issue', 'lang', 
                                                                                                   'n_citation', 
                                                                                                   'page_end', 
                                                                                                   'page_start', 
                                                                                                   'publisher', 
                                                                                                   'references',
                                                                                                   'url', 'venue', 
                                                                                                   'volume'], axis=1)

model_df.shape

model_df.head(2)

unique_fos = sorted(list({ feature
                          for paper_row in model_df.fos.fillna('0')
                          for feature in paper_row }))

unique_year = sorted(model_df['year'].astype('str').unique())

len(unique_fos + unique_year)

model_df.shape[0] - pd.isnull(model_df['fos']).sum()

len(unique_fos)

import random
[unique_fos[i] for i in sorted(random.sample(range(len(unique_fos)), 15)) ]

def feature_array(x, unique_array):
    row_dict = {}
    for i in x.index:
        var_dict = {}
        
        for j in range(len(unique_array)):
            if type(x[i]) is list:
                if unique_array[j] in x[i]:
                    var_dict.update({unique_array[j]: 1})
                else:
                    var_dict.update({unique_array[j]: 0})
            else:    
                if unique_array[j] == str(x[i]):
                    var_dict.update({unique_array[j]: 1})
                else:
                    var_dict.update({unique_array[j]: 0})
        
        row_dict.update({i : var_dict})
    
    feature_df = pd.DataFrame.from_dict(row_dict, dtype='str').T
    
    return feature_df

get_ipython().magic("time year_features = feature_array(model_df['year'], unique_year)")

get_ipython().magic("time fos_features = feature_array(model_df['fos'], unique_fos)")

from sys import getsizeof
print('Size of fos feature array: ', getsizeof(fos_features))

year_features.shape[1] + fos_features.shape[1]

# now looking at 10399 x  7760 array for our feature space

get_ipython().magic('time first_features = fos_features.join(year_features).T')

first_size = getsizeof(first_features)

print('Size of first feature array: ', first_size)

first_features.shape

first_features.head()

from scipy.spatial.distance import cosine

def item_collab_filter(features_df):
    item_similarities = pd.DataFrame(index = features_df.columns, columns = features_df.columns)
    
    for i in features_df.columns:
        for j in features_df.columns:
            item_similarities.loc[i][j] = 1 - cosine(features_df[i], features_df[j])
    
    return item_similarities

get_ipython().magic('time first_items = item_collab_filter(first_features.loc[:, 0:1000])')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().magic('matplotlib inline')

sns.set()
ax = sns.heatmap(first_items.fillna(0), 
                 vmin=0, vmax=1, 
                 cmap="YlGnBu", 
                 xticklabels=250, yticklabels=250)
ax.tick_params(labelsize=12)

def paper_recommender(paper_index, items_df):
    print('Based on the paper: \nindex = ', paper_index)
    print(model_df.iloc[paper_index])
    top_results = items_df.loc[paper_index].sort_values(ascending=False).head(4)
    print('\nTop three results: ') 
    order = 1
    for i in top_results.index.tolist()[-3:]:
        print(order,'. Paper index = ', i)
        print('Similarity score: ', top_results[i])
        print(model_df.iloc[i], '\n')
        if order < 5: order += 1

paper_recommender(2, first_items)


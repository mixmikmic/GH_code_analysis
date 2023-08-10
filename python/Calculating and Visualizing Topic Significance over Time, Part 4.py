# Load the necessary libraries
import gensim # Note: I am running 1.0.1
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd # Note: I am running 0.19.2
import pyLDAvis.gensim
import seaborn as sns
import warnings

# Enable in-notebook visualizations
get_ipython().magic('matplotlib inline')

# Temporary fix for persistent warnings of an api change between pandas and seaborn.
warnings.filterwarnings('ignore')

pd.options.display.max_rows = 10

base_dir = "data"
period = "1859-to-1875"
directory = "historical_periods"

metadata_filename = os.path.join(base_dir,'2017-05-corpus-stats/2017-05-Composite-OCR-statistics.csv')
index_filename = os.path.join(base_dir, 'corpora', directory, '{}.txt'.format(period))
labels_filename = os.path.join(base_dir, 'dataframes', directory, '{}_topicLabels.csv'.format(period))
doc_topic_filename = os.path.join(base_dir, 'dataframes', directory, '{}_dtm.csv'.format(period))

def doc_list(index_filename):
    """
    Read in from a json document with index position and filename. 
    File was created during the creation of the corpus (.mm) file to document
    the filename for each file as it was processed.
    
    Returns the index information as a dataframe.
    """
    with open(index_filename) as data_file:    
        data = json.load(data_file)
    docs = pd.DataFrame.from_dict(data, orient='index').reset_index()
    docs.columns = ['index_pos', 'doc_id']
    docs['index_pos'] = docs['index_pos'].astype(int)
  
    return docs


def compile_dataframe( index, dtm, labels, metadata):
    """
    Combines a series of dataframes to create a large composit dataframe.
    """
    doc2metadata = index.merge(metadata, on='doc_id', how="left")
    topics_expanded = dtm.merge(labels, on='topic_id')
    
    df = topics_expanded.merge(doc2metadata, on="index_pos", how="left")
    
    return df

order = ['conference, committee, report, president, secretary, resolved',
         'quarterly, district, society, send, sept, business',
         'association, publishing, chart, dollar, xxii, sign',
         'mother, wife, told, went, young, school',
         'disease, physician, tobacco, patient, poison, medicine',
         'wicked, immortality, righteous, adam, flesh, hell',
        ]

def create_pointplot(df, y_value, hue=None, order=order, col=None, wrap=None, size=6, aspect=1.5, title=""):
    p = sns.factorplot(x="year", y=y_value, kind='point', hue_order=order, hue=hue, 
                       col=col, col_wrap=wrap, col_order=order, size=size, aspect=aspect, data=df)
    p.fig.subplots_adjust(top=0.9)
    p.fig.suptitle(title, fontsize=16)
    return p

metadata = pd.read_csv(metadata_filename, usecols=['doc_id', 'year','title'])
docs_index = doc_list(index_filename)
dt = pd.read_csv(doc_topic_filename)
labels = pd.read_csv(labels_filename)

# Reorient from long to wide
dtm = dt.pivot(index='index_pos', columns='topic_id', values='topic_weight').fillna(0)

# Divide each value in a row by the sum of the row to normalize the values
# https://stackoverflow.com/questions/18594469/normalizing-a-pandas-dataframe-by-row
dtm = dtm.div(dtm.sum(axis=1), axis=0)

# Shift back to a long dataframe
dt_norm = dtm.stack().reset_index()
dt_norm.columns = ['index_pos', 'topic_id', 'norm_topic_weight']

df = compile_dataframe(docs_index, dt_norm, labels, metadata)

df

y_df = df.groupby(['year', 'topic_id']).agg({'norm_topic_weight': 'sum'})

# This achieves a similar computation as the normalizing function above, but without converting into a matrix first.
# For each group (in this case year), we are dividing the values by the sum of the values. 
ny_df = y_df.groupby(level=0).apply(lambda x: x / x.sum()).reset_index()

ny_df.columns = ['year', 'topic_id', 'normalized_weight']
ny_df = ny_df.merge(labels, on="topic_id")

ny_df

# Limit the data to our 6 topic sample
ny_df_filtered = ny_df[(ny_df['topic_id'] >= 15) & (ny_df['topic_id'] <= 20)]

create_pointplot(ny_df_filtered, "normalized_weight", hue="topic_words", 
                 title="Proportion of total topic weight by topic per year.")

create_pointplot(df, 'norm_topic_weight', hue='topic_words',
                 title='Central range of topic weights by topic per year.'
                )

titles = df.groupby(['title', 'topic_id']).agg({'norm_topic_weight': 'sum'})
titles = titles.groupby(level=0).apply(lambda x: x / x.sum()).reset_index()
titles = titles.merge(labels, on='topic_id')

titles.pivot('title','topic_words', 'norm_topic_weight').plot(kind='bar', stacked=True, colormap='Paired', 
      figsize=(8,7), title='Normalized proportions of topic weights per title.')\
.legend(bbox_to_anchor=(1.75, 1))

# Remove the zero values we added in when normalizing each document
df2 = df[df['norm_topic_weight'] > 0]

p_df2 = df2.groupby(['year', 'topic_id']).agg({'norm_topic_weight': 'sum'})

# Normalize each group by dividing by the sum of the group
np_df2 = p_df2.groupby(level=0).apply(lambda x: x / x.sum()).reset_index()
np_df2.columns = ['year', 'topic_id', 'normalized_weight']
np_df2 = np_df2.merge(labels, on="topic_id")

np_df2_filtered = np_df2[(np_df2['topic_id'] >= 15) & (np_df2['topic_id'] <= 20)]

create_pointplot(np_df2_filtered, "normalized_weight", hue="topic_words", 
                 title="Normalized topic weights by topic per year, zero values excluded.")

create_pointplot(df2, 'norm_topic_weight', hue='topic_words',
                 title='Central range of topic weights by topic per year, zero values excluded.'
                )




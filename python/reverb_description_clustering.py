import re
import pandas as pd
import collections
import numpy as np
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
init_notebook_mode(connected=True)

rdf = pd.read_csv('reverb_effects_pedals_v4_08032016.csv')
rdf.shape

rdf = rdf.drop_duplicates('web_url', keep='first')

rdf.title = rdf.title.str.lower().str.strip()
rdf = rdf.drop_duplicates('title', keep='first')

rdf.description = rdf.description.str.lower().str.strip()
rdf = rdf.drop_duplicates('description', keep='first')

rdf = rdf.drop_duplicates('id', keep='first')

rdf = rdf.drop_duplicates('model', keep='first')

rdf.shape

descriptions = rdf.description.str.lower().str.strip()

def filter_brands(descriptions, brand_names):
    clean_descriptions = []
    for description in descriptions:
        for brand in brand_names:
            if brand in description:
                description = description.replace(brand, '').strip()
            else:
                pass
        clean_descriptions.append(description)
    return clean_descriptions

filtered_descriptions = filter_brands(descriptions, set(rdf.make.str.lower()))

def stem_and_tokenize(description):
    stemmer = SnowballStemmer('english')
    tokens = [word.lower().strip() for sentence in sent_tokenize(description) for word in word_tokenize(sentence)]
    filtered_tokens = [token for token in tokens if re.search('^[A-Za-z]+$', token)]
    return [stemmer.stem(token) for token in filtered_tokens]

def tokenize(description):
    tokens = [word.lower().strip() for sentence in sent_tokenize(description) for word in word_tokenize(sentence)]
    filtered_tokens = [token for token in tokens if re.search('^[A-Za-z]+$', token)]
    return filtered_tokens

def generate_stems(descriptions):
    return [stem_and_tokenize(description) for description in descriptions]

def generate_tokens(descriptions):
    return [tokenize(description) for description in descriptions]

token_lists = generate_tokens(filtered_descriptions)
stem_lists = generate_stems(filtered_descriptions)

print('First 10 stems: ', stem_lists[0][:10])
print('First 10 tokens:', token_lists[0][:10])

word_df = pd.DataFrame([x for lst in token_lists for x in lst], 
                       [y for lst in stem_lists for y in lst])
word_df.columns = ['word']
word_df.head(10)

final_tokens = [' '.join([word for word in lst]) for lst in token_lists]
final_stems = [' '.join([word for word in lst]) for lst in stem_lists]
print('Final Stem Example: ', final_stems[0])
print()
print('Final Token Example: ' , final_tokens[0])

get_ipython().run_cell_magic('html', '', '<style>\ntable {float:left}\n</style>')

tfidf = TfidfVectorizer(stop_words='english', tokenizer=stem_and_tokenize, max_df=0.8)
tfidf_matrix = tfidf.fit_transform(final_stems)
tfidf_matrix.shape

tfidf_df = pd.DataFrame(tfidf_matrix.toarray())
tfidf_df.head(10)

num_clusters = 15
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

cluster_labels = km.labels_.tolist()
print('# of labels: ', len(cluster_labels))

sorted_km_centroids = km.cluster_centers_.argsort()[:, ::-1]
len(sorted_km_centroids)

terms = tfidf.get_feature_names()
terms[:10]

def create_centroid_df(sorted_km_centroids, num_clusters, feature_names):
    all_centroid_dicts = []
    for i in range(num_clusters):
        centroid_dict = dict(label = i)
        word_list = []
        for skc in sorted_km_centroids[i, :10]:
            token = list(word_df.ix[feature_names[skc]].word)[0]
            word_list.append(str(token))
            centroid_dict['words'] = ','.join(word_list)
        all_centroid_dicts.append(centroid_dict)
    return all_centroid_dicts

all_centroid_dicts = create_centroid_df(sorted_km_centroids, num_clusters, terms)
all_centroid_dicts_df = pd.DataFrame(all_centroid_dicts)
all_centroid_dicts_df.head()

lsa = TruncatedSVD(n_components=3, random_state=42)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

x_values = lsa_matrix[:,0]
y_values = lsa_matrix[:,1]
z_values = lsa_matrix[:,2]

plot_df = pd.DataFrame(dict(x=x_values, y=y_values, z=z_values, title=rdf.title, label=cluster_labels))

plot_df['words'] = plot_df.label.map(all_centroid_dicts_df.set_index('label').words)
plot_df.head()

data = [Scatter3d(
        x = plot_df.x,
        y = plot_df.y,
        z = plot_df.z,
        mode = 'markers',
        name = 'plot clusters',
        text = plot_df.title,
        showlegend = True,
        marker = dict(
            size = 5,
            color = plot_df.label,
            colorscale = 'Rainbow',
            showscale = True
        )
    )]
layout = Layout(
    title = 'Pedal Clusters based on Product Descriptions',
    margin = dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    xaxis = dict(
        showgrid = False,
        showticklabels = False,
        showline = False,
        zeroline = False
    ),
    yaxis = dict(
        showgrid = False,
        showticklabels = False,
        showline = False,
        zeroline = False
    ),
)
fig = Figure(layout=layout, data=data)
iplot(fig)


import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.spatial.distance import cdist
import numpy as np

corpus =[
    "This is an awesome day!",
    "Weather is bright and sunny",
    "The boat is ready for sailing",
    "Ready the arms",
    "Life is beautiful",
    "babies are beautiful and full of life",
    "Sunny-side omlets",
    "A beer a day keeps beautiful girls at bay!"
]

def print_similar_docs(query, documents, top_n):
    
    vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        min_df=0,
        stop_words='english'
    )
    
    doclist = np.array(documents)
    #Calculate the TF-IDF
    doclist_tfidf = vec.fit_transform(doclist).toarray()    
    query_tfidf = vec.transform([query]).toarray()
    spatial_distances = cdist(query_tfidf, doclist_tfidf, 'cosine')
    rec_idx = spatial_distances.argsort()
    
    print('Query String: %s\n' % query)
    print('Spacial Distances:')
    print(repr(spatial_distances))
    print('Recommendations:')
    print(doclist[rec_idx[0][1:top_n+1]])
    return

print_similar_docs("Today is a rainy day", corpus, 3)



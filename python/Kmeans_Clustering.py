import pandas as pd 

import sys
sys.version 

import tempfile
import zipfile
import os.path

zipFile = "./openSubtitles-5000.json.zip"

print( "Unarchiving ...")
temp_dir = tempfile.mkdtemp()
zip_ref = zipfile.ZipFile(zipFile, 'r')
zip_ref.extractall(temp_dir)
zip_ref.close()

openSubtitlesFile = os.path.join(temp_dir, "openSubtitles-5000.json")
print ("file unarchived to:" + openSubtitlesFile)


import json
from sklearn.feature_extraction.text import CountVectorizer
#from log_progress import log_progress

maxDocsToload = 5000

titles = []
def make_corpus(file):
    with open(file) as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            titles.append(doc.get('Title',''))
            #if 'Sci-Fi' not in doc.get('Genre',''):
            #    continue
            if i % 100 == 0:
                print ("%d " % i, end='') 
            yield doc.get('Text','')
            if i == maxDocsToload:
                break
                
print ("Starting load ...")                
textGenerator = make_corpus(openSubtitlesFile)              
count_vectorizer = CountVectorizer(min_df=2, max_df=0.75, ngram_range=(1,2), max_features=50000,
                                   stop_words='english', analyzer="word", token_pattern="[a-zA-Z]{3,}")
term_freq_matrix = count_vectorizer.fit_transform(textGenerator)
print ("Done.")
print ( "term_freq_matrix shape = %s" % (term_freq_matrix.shape,) )
print ("term_freq_matrix = \n%s" % term_freq_matrix)

print( "Vocabulary length = ", len(count_vectorizer.vocabulary_))
word = "data";
rainingIndex = count_vectorizer.vocabulary_[word];
print( "token index for \"%s\" = %d" % (word,rainingIndex))
feature_names = count_vectorizer.get_feature_names()
print( "feature_names[%d] = %s" % (rainingIndex, feature_names[rainingIndex]))

for i in range(0,1000):
    print( "feature_names[%d] = %s" % (i, feature_names[i]))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(term_freq_matrix)

tf_idf_matrix = tfidf.transform(term_freq_matrix)
print( tf_idf_matrix)

get_ipython().run_cell_magic('time', '', 'from sklearn.cluster import KMeans\nimport numpy\n\nnum_clusters = 5\nkm = KMeans(n_clusters=num_clusters, verbose=True, init=\'k-means++\', n_init=3, n_jobs=-1)\nkm.fit(tf_idf_matrix)\n\nclusters = km.labels_.tolist()\nprint ("cluster id for each document = %s" % clusters)\n\nprint()\n# sort cluster centers by proximity to centroid\norder_centroids = km.cluster_centers_.argsort()[:, ::-1]\n\n        ')

labels = pd.DataFrame(clusters, columns=['Cluster Labels'])
counts = pd.DataFrame(labels['Cluster Labels'].value_counts().sort_index())
counts.columns=['Document Count']
display(counts)

topNWords = 50

df = pd.DataFrame()

for i in range(num_clusters):
    clusterWords = []
    for topWordIndex,ind in enumerate(order_centroids[i, :topNWords]):   
        clusterWords.append( feature_names[ind] )
    df['Cluster %d' % i] = pd.Series(clusterWords)
        #dtype='object', data= [''] * topNWords)
        #print(topWordIndex)        
        #print(ind)
        #print(feature_names[ind])

df.style.set_properties(**{'text-align': 'right'})
df


titlesFrame = pd.DataFrame()
titlesFrame['Labels']=km.labels_
titlesFrame['Titles']=titles

sort = titlesFrame.sort_values(by=['Labels','Titles'])
for i in range(num_clusters):
    display( sort.query('Labels == %d' % i) )




import pandas as pd
import numpy as np 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
import matplotlib.patches as mpatches
import seaborn as sns

food = pd.read_csv('./fastfood.csv', index_col=0)



food = food.drop_duplicates()

food.shape

comps = list(food.Company.unique())[:-3]
del comps[-2]
comps = [x[1:] for x in comps]

stopword = stopwords.words('english')
stopword.extend(comps)
stopword.extend(list(food.Company.unique())[:-3])

companies = list(food.Company.unique())[:-3]
texts = []
for x in companies:
    try:
        something = ' '.join(val for val in food.loc[food['Company'] == x]['text'])
        texts.append(something)
    except:
        pass
    

del texts[-2]

def Text_Cleaner(text):
    """Takes text, eliminates URLS, tokenizes, removes company names, lower cases, removes calls to twitter handles, 
    returns a string"""
    text = re.sub(r'(https)[^\s]+', '', text)
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    lower = [x.lower() for x in words]
    eliminator = [re.sub(r'(mcdon|dunki|denn|redro|sonic|starb|shakesh|domino|crackerb|chipot|wend)[a-z]+','',x)
                  for x in lower]
    eliminator2 = [re.sub(r'@[a-zA-Z0-9]+', '', x) for x in eliminator]
    return ' '.join(eliminator2)

cleaned = [Text_Cleaner(x) for x in texts]

from gensim import corpora, models

np.random.seed(42)

dictionary = corpora.Dictionary(cleaned)

corpus = [dictionary.doc2bow(clean) for clean in cleaned]

ldamodel = models.ldamodel.LdaModel(corpus, 
                                    id2word = dictionary,
                                    num_topics = 7,
                                    passes = 50,
                                    minimum_probability = 0
                                   )

import pyLDAvis.gensim

x = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(x)

for i in ldamodel.print_topics(num_topics=14, num_words=5):
    print(i)

#K-Means Clustering- remember- biased heavily by outliers
#possibly use k-means ++ method in scikit learn
#k-means is not great with non-spherical data, this may not be the best case...

#7155 multiple tweeters out of 27528 tweeters 
food.user_name.value_counts().sort_values(ascending=False).shape

from sklearn.cluster import KMeans, k_means
from sklearn.metrics import silhouette_score

food.columns

food['user_is_verified'] = food['user_is_verified'].astype(bool).astype(int)

#remember to drop these always, weird indexing...
food = food.drop(list(food.loc[food['retweet_count'] == 'False'].index))

cluster_quant = food.drop(['Company', 'text', 'time_tweeted', 
       'unique_code', 'user_coordinates','user_is_verified', 'user_location', 'user_name', 'user_profile_text'], axis=1)
cluster_quant.shape

cluster = cluster_quant.dropna()
cluster.shape

cluster.retweet_count = cluster.retweet_count.astype(int)

cluster.head()





model = KMeans(n_clusters=3, random_state=0)
model.fit(cluster)

cluster['predicted'] = model.predict(cluster)

cluster.retweet_count = np.log(cluster['retweet_count'].values)
cluster.favorite_count = np.log(cluster['favorite_count'].values)

plt.figure(figsize=(15,25))
fg = sns.FacetGrid(data=cluster, hue='predicted', aspect=1.61)
fg.map(plt.scatter, 'retweet_count', 'favorite_count')

#it is useful to scale before clustering
#to optimize k, use the elbow method (inertia vs. silhouette score)


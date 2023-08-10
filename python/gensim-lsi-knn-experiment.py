import pandas as pd
import warnings

warnings.filterwarnings('ignore')
rows = pd.read_csv('../data/hourly_prices.csv', index_col=False, thousands=',')

from gensim.models import LsiModel, TfidfModel
from gensim.corpora import Dictionary

vocab = Dictionary(rows['Labor Category'].str.split())

tfidf = TfidfModel(id2word=vocab, dictionary=vocab)

bows = rows['Labor Category'].apply(lambda x: vocab.doc2bow(x.split()))

vocab.token2id['engineer']

vocab[0]

dict([(vocab[i], round(freq, 2)) for i, freq in tfidf[bows[0]]])

lsi = LsiModel(tfidf[bows], num_topics=5, id2word=vocab, extra_samples=100, power_iters=2)

len(vocab)

topics = lsi[bows]
df_topics = pd.DataFrame([dict(d) for d in topics], index=bows.index, columns=range(5))

lsi.print_topic(1, topn=5)

PRICE_COEFF = 1 / 500.0
XP_COEFF = 1 / 10.0

df_topics['Price'] = (rows['Year 1/base'] * PRICE_COEFF).fillna(0)
df_topics['Experience'] = (rows['MinExpAct'] * XP_COEFF).fillna(0)

from sklearn.neighbors import NearestNeighbors

df_topics = df_topics.fillna(0)
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(df_topics)

neigh.kneighbors(df_topics.ix[0].values.reshape(1, -1), return_distance=False)

def get_neighbors(labor_category, price, experience):
    vector = []
    topic_values = lsi[tfidf[vocab.doc2bow(labor_category.split())]]
    vector.extend([v[1] for v in topic_values])
    vector.extend([price * PRICE_COEFF, experience * XP_COEFF])
    
    neighbors = list(neigh.kneighbors([vector], return_distance=False)[0])
    return pd.DataFrame([rows.loc[i] for i in neighbors], index=neighbors)

get_neighbors('Awesome Engineer', 80, 5)




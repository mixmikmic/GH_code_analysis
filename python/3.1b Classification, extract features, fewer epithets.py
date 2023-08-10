from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithet_index
import pandas

epithet_frequencies = []
for epithet, _ids in get_epithet_index().items():
    epithet_frequencies.append((epithet, len(_ids)))
df = pandas.DataFrame(epithet_frequencies)
df.sort_values(1, ascending=False)

from scipy import stats

distribution = sorted(list(df[1]), reverse=True)
zscores = stats.zscore(distribution)
list(zip(distribution, zscores))

# Make list of epithets to drop
to_drop = df[0].where(df[1] < 26)
to_drop = [epi for epi in to_drop if not type(epi) is float]
to_drop = set(to_drop)
to_drop

import datetime as dt
import os
import time

from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithet_of_author
from cltk.corpus.greek.tlg.parse_tlg_indices import get_id_author
import pandas
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

def stream_lemmatized_files(corpus_dir):
    # return all docs in a dir
    user_dir = os.path.expanduser('~/cltk_data/user_data/' + corpus_dir)
    files = os.listdir(user_dir)

    for file in files:
        filepath = os.path.join(user_dir, file)
        with open(filepath) as fo:
            #TODO rm words less the 3 chars long
            yield file[3:-4], fo.read()

t0 = dt.datetime.utcnow()

map_id_author = get_id_author()

df = pandas.DataFrame(columns=['id', 'author' 'text', 'epithet'])

for _id, text in stream_lemmatized_files('tlg_lemmatized_no_accents_no_stops'):
    author = map_id_author[_id]
    epithet = get_epithet_of_author(_id)
    if epithet in to_drop:
        continue
    df = df.append({'id': _id, 'author': author, 'text': text, 'epithet': epithet}, ignore_index=True)

print(df.shape)
print('... finished in {}'.format(dt.datetime.utcnow() - t0))
print('Number of texts:', len(df))

text_list = df['text'].tolist()

# make a list of short texts to drop
# For pres, get distributions of words per doc
short_text_drop_index = [index if len(text) > 500 else None for index, text in enumerate(text_list) ]  # ~100 words

t0 = dt.datetime.utcnow()

# TODO: Consider using generator to CV http://stackoverflow.com/a/21600406

# time & size counts, w/ 50 texts:
# 0:01:15 & 202M @ ngram_range=(1, 3), min_df=2, max_features=500
# 0:00:26 & 80M @ ngram_range=(1, 2), analyzer='word', min_df=2, max_features=5000
# 0:00:24 & 81M @ ngram_range=(1, 2), analyzer='word', min_df=2, max_features=50000

# time & size counts, w/ 1823 texts:
# 0:02:18 & 46MB @ ngram_range=(1, 1), analyzer='word', min_df=2, max_features=500000
# 0:2:01 & 47 @ ngram_range=(1, 1), analyzer='word', min_df=2, max_features=1000000

# max features in the lemmatized data set: 551428
max_features = 100000
ngrams = 1
vectorizer = CountVectorizer(ngram_range=(1, ngrams), analyzer='word', 
                             min_df=2, max_features=max_features)
term_document_matrix = vectorizer.fit_transform(text_list)  # input is a list of strings, 1 per document

# save matrix
vector_fp = os.path.expanduser('~/cltk_data/user_data/vectorizer_test_features{0}_ngrams{1}.pickle'.format(max_features, ngrams))
joblib.dump(term_document_matrix, vector_fp)

print('... finished in {}'.format(dt.datetime.utcnow() - t0))

# Put BoW vectors into a new df
term_document_matrix = joblib.load(vector_fp)  # scipy.sparse.csr.csr_matrix

term_document_matrix.shape

term_document_matrix_array = term_document_matrix.toarray() 

dataframe_bow = pandas.DataFrame(term_document_matrix_array, columns=vectorizer.get_feature_names())

ids_list = df['id'].tolist()

len(ids_list)

dataframe_bow.shape

dataframe_bow['id'] = ids_list

authors_list = df['author'].tolist()
dataframe_bow['author'] = authors_list

epithets_list = df['epithet'].tolist()
dataframe_bow['epithet'] = epithets_list

# For pres, give distribution of epithets, including None
dataframe_bow['epithet']

t0 = dt.datetime.utcnow()

# removes 334
#! remove rows whose epithet = None
# note on selecting none in pandas: http://stackoverflow.com/a/24489602
dataframe_bow = dataframe_bow[dataframe_bow.epithet.notnull()]
dataframe_bow.shape

print('... finished in {}'.format(dt.datetime.utcnow() - t0))

t0 = dt.datetime.utcnow()

dataframe_bow.to_csv(os.path.expanduser('~/cltk_data/user_data/tlg_bow.csv'))

print('... finished in {}'.format(dt.datetime.utcnow() - t0))

dataframe_bow.shape

dataframe_bow.head(10)

# write dataframe_bow to disk, for fast reuse while classifying
# 2.3G
fp_df = os.path.expanduser('~/cltk_data/user_data/tlg_bow_df.pickle')
joblib.dump(dataframe_bow, fp_df)




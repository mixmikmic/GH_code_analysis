import os
import pickle
import pandas as pd
import spacy
import spotlight
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

input_dir = 'textbooks/'
data_dir = 'data/'
metadata_file = 'data/metadata.csv'
toc_file = 'toc.pkl'
text_file = 'text.pkl'
spotlight_server = 'http://192.168.99.101:2222/rest/annotate'

isbns = os.listdir(input_dir)

with open(os.path.join(data_dir, toc_file), 'rb') as fp:
    all_toc = pickle.load(fp) 

with open(os.path.join(data_dir, text_file), 'rb') as fp:
    all_text = pickle.load(fp) 
    
nlp = spacy.load('en')

metadata = pd.read_csv(metadata_file, dtype = {'ISBN': 'str'})
metadata['num_pages'] = [len(all_text[isbn]) for isbn in metadata['ISBN']]
metadata

isbn = '9781429242301'

start_page = metadata.loc[metadata['ISBN'] == isbn, 'start_page'].values[0]
end_page = metadata.loc[metadata['ISBN'] == isbn, 'end_page'].values[0]

# Extract the content portion of the textbook, and combine the pages
text = all_text[isbn][(start_page-1):(end_page)]
text = ' '.join(text)

# Parse the textbook with spacy
doc = nlp(text)

# Some spacy examples
token = doc[2]
print(token)
sentence = next(doc.sents)
print(sentence)
print([word.lemma_ for word in sentence])

# Confidence = confidence score for disambiguation / linking
# Support = number of inlinks to the wikipedia entry

# Low support with high confidence

annotations = spotlight.annotate(spotlight_server,
                                 doc.string,
                                 confidence=0.9, support=2)

annotations[1]

annotation_names = [ann['surfaceForm'] for ann in annotations
                   if ann['surfaceForm'] != '/12']
# TODO:
# Exclude '/12' or certain 'types': 'DBpedia:TimePeriod,DBpedia:Year'

annotation_names[:20]

words = [token.text for token in doc if 
         token.is_stop != True and 
         token.is_punct != True and 
         token.is_digit != True and 
         token.pos_ == "NOUN" and
         len(token) > 2]

# five most common tokens
word_freq = Counter(words)
common_words = word_freq.most_common(30)

for word, count in common_words:
    print('{} - {}'.format(word, count))

textbook_text = []

for isbn in isbns:
    start_page = metadata.loc[metadata['ISBN'] == isbn, 'start_page'].values[0]
    end_page = metadata.loc[metadata['ISBN'] == isbn, 'end_page'].values[0]

    # Extract the content portion of the textbook, and combine the pages
    text = all_text[isbn][(start_page-1):(end_page)]
    text = ' '.join(text)

    # Parse the textbook with spacy
    doc = nlp(text)

    text_clean = [token.text for token in doc if 
                  token.is_stop != True and 
                  token.is_punct != True and 
                  token.is_digit != True and 
                  token.pos_ == "NOUN" and
                  len(token) > 2]
    
    textbook_text.append(' '.join(text_clean))

n_features = 1000
n_topics = 10

# Use tf (raw term count) features for LDA
print("Extracting tf features for LDA...")
tf_vectorizer = TfidfVectorizer(min_df=1,
                                max_features=n_features,
                                stop_words='english')


tf = tf_vectorizer.fit_transform(textbook_text)

print("Fitting LDA models with tf features")
lda = LatentDirichletAllocation(n_topics=n_topics, 
                                max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

print("\nTopics in LDA model:")
print()

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()
    
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, 20)


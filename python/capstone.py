import sqlite3
conn = sqlite3.connect('articles.sqlite')
cur = conn.cursor()

# select top authors
top_author = 'SELECT author_unique FROM Counts WHERE count >= 50 ORDER BY count DESC LIMIT 10'

author_doc = []
author_lst = []

for i in cur.execute(top_author) :
    author_lst.append(i[0])
print "number of unique authors: ", len(author_lst)

import itertools
import pandas as pd

d={}
list_authors = []
for i, author_correct in enumerate(author_lst):

    cur.execute('''SELECT abstract, author FROM Articles WHERE author LIKE ? ''', ('%{}%'.format(author_correct), ) )
    all_rows = cur.fetchall()

    authors_list = [x[1] for x in all_rows]
    docs = [x[0] for x in all_rows]

    for author, row in itertools.izip(authors_list, docs) :
        authors = author.split('; ')
        for a in authors:
            if author_correct == a:
                author_doc.append(row)
                list_authors.append(a)
    conn.commit()

df = pd.DataFrame({'author' : list_authors, 'doc': author_doc})
print "size: ", df.shape
print df.head()

cur.close()

print df['author'].value_counts()

vocab = []
lengths = []

for i in range(df.shape[0]):
    doc = df['doc'][i]
    doc_split = doc.split(" ")
    lengths.append(len(doc_split))

max_length = max(lengths)
print "max doc length: ", max_length      

author_doc = []
for i in range(df.shape[0]):
    doc = df['doc'][i]
    doc_split = doc.split(" ")
    # How much to pad each doc
    padding_num = max_length - len(doc_split)
    doc_new = doc + " </PAD>" * padding_num
    words = doc_new.split(" ")
    vocab.append(words)
    author_doc.append(doc_new)
print len(author_doc)

from collections import Counter
import numpy as np

word_counts = Counter(itertools.chain(*vocab))

# Mapping from index to word
vocabulary_inv = [x[0] for x in word_counts.most_common()]

# Mapping from word to index
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

# Map docs and labels (authors) to vectors based on a vocabulary
print "vocab shape: ", len(vocabulary)
x = np.array([ [ vocabulary[word] for word in doc ] for doc in vocab ])
print "x size: ", x.shape

labels = df['author'].astype('category')
labels = labels.cat.codes
labels_unique = np.unique(labels)

from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
lb.fit(labels)
print "classes: ", lb.classes_
labels_tf = lb.transform(labels)
print "labels: ", labels_tf.shape

# Pareto Chart 
import matplotlib.pyplot as plt

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')

import matplotlib

common_words = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

counts_total = dict()

for index, row in df.iterrows():
    words = row.values[1].split()
    
    for word in words:
        counts_total[word] = counts_total.get(word,0) + 1
    for common_word in common_words:
        counts_total.pop(common_word, None)
    
# sort counts in descending order
labels, heights = zip(*sorted(((k, v) for k, v in counts_total.items()), reverse=True))

from paretochart import pareto
fig, ax = plt.subplots(figsize=(40, 20))
pareto(heights, labels, cumplot=False, limit=0.1)


plt.title('Most frequent words in all abstracts', fontsize=40)
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

plt.show()

from wordcloud import WordCloud

abstract = df.iloc[0]
# print abstract.values[1]
# print type(abstract.values[1])

# Generate a word cloud image for all abstracts
lst = list()
for index, row in df.iterrows():
    lst.append(row.values[1])
    
abstracts = ''.join(text for text in lst)
wordcloud = WordCloud().generate(abstracts)

# display wordcloud
fig, ax = plt.subplots(figsize=(30, 30))
wordcloud = WordCloud(max_font_size=35).generate(abstracts)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_index, test_index in sss.split(x, labels_tf):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = labels_tf[train_index], labels_tf[test_index]

vocab_size = len(vocabulary)
doc_size = x_train.shape[1]
print "Train/test split: %d/%d" % (len(y_train), len(y_test))
print 'Train shape:', x_train.shape
print 'Test shape:', x_test.shape

from sklearn import linear_model
from time import time
import string
from nltk import SnowballStemmer

def stem(text):
    stems = []
    exclude = set(string.punctuation)
    words = text.split()
    for word in words:
        # remove punctuation
        word = ''.join(ch for ch in word if ch not in exclude)
        # remove digits
        word = ''.join( [i for i in word if not i.isdigit()] )
        # stem words
        word = SnowballStemmer("english").stem(word)
        stems.append(word)
    return stems

from sklearn.feature_extraction.text import TfidfVectorizer

# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=1000,
                                 min_df=0.05, stop_words='english',
                                 use_idf=True, tokenizer=stem, ngram_range=(1,3))

# fit the vectorizer to synopses
tfidf_matrix = tfidf_vectorizer.fit_transform(df['doc']) 

for train_index, test_index in sss.split(x, labels_tf):
    X_logit_train, X_logit_test = tfidf_matrix[train_index], tfidf_matrix[test_index]   
    labels_train, labels_test = df['author'][train_index], df['author'][test_index]

t0 = time()
clf = linear_model.LogisticRegression(max_iter=500, random_state=42,
                                 multi_class="ovr").fit(X_logit_train, labels_train)
pred = clf.predict(X_logit_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

tt = time()-t0
print("Training Log Regression took: {}").format(round(tt,3))
print "Accuracy score on test data is {}.".format(round(acc,4))

import tensorflow as tf
from time import time

num_classes = len(labels_unique)

x = tf.placeholder(tf.int32, [None, x_train.shape[1]], name="input_x")

# y_ is the correct classes
y_ = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

# Keeping track of l2 regularization loss 
l2_loss = tf.constant(0.0)

embedding_size = 500
filter_sizes = [3,4,5]
num_filters = 500

# Embedding layer
W_emb = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedded_chars = tf.nn.embedding_lookup(W_emb, x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# Create a convolution + maxpool layer for each filter size
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    # Convolution Layer
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]))
    conv = tf.nn.conv2d(
        embedded_chars_expanded,
        W_conv,
        strides=[1, 1, 1, 1],
        padding="VALID")
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(
        h,
        ksize=[1, doc_size - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')
    pooled_outputs.append(pooled)

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(3, pooled_outputs)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
keep_prob = tf.placeholder(tf.float32)
h_drop = tf.nn.dropout(h_pool_flat, keep_prob)

# Unnormalized scores and predictions
W = tf.get_variable(
    "W",
    shape=[num_filters_total, num_classes],
    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
l2_loss += tf.nn.l2_loss(W)
l2_loss += tf.nn.l2_loss(b)
scores = tf.nn.xw_plus_b(h_drop, W, b)
predictions = tf.argmax(scores, 1)

l2_reg_lambda=0.0

losses = tf.nn.softmax_cross_entropy_with_logits(scores, y_)
cross_entropy = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

# Accuracy score
correct_predictions = tf.equal(predictions, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

t0 = time()
for i in range(1000):
    if i%100 == 0:

        train_accuracy = accuracy.eval(session=sess, feed_dict={ x: x_train, y_: y_train, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: x_train, y_: y_train, keep_prob: 0.8})


tt = time()-t0
print("Training TensorFlow NN took: {}").format(round(tt,3))

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x: x_test, y_: y_test, keep_prob: 1.0}))


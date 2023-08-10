import gensim.models.doc2vec as d2v
import os
import pickle
import nltk
from pypet import progressbar
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def load_files(folder):
    """Loads the pickled comments from disk. Unfortunately I cannot publish the 
    scraped data due to copyright problmes"""
    res = []
    for filename in os.listdir(folder):
        print(filename)
        if filename.endswith('.pckl'):
            with open(os.path.join(folder, filename), 'rb') as fh:
                res.append(pickle.load(fh))
    return res


zeit_lists = load_files('./zeit2/')
spon_lists = load_files('./spon/')
fc_lists = load_files('./fc2/')

def concate_dicts(*llist_):
    """Converts data into dictionaries with a publisher key label"""
    articles = {}
    comments = {}
    for list_, key in llist_:
        for db_articles, db_comments in list_:
            for article in db_articles.values():
                article['publisher'] = key
            for comment in db_comments.values():
                comment['publisher'] = key
            articles.update(db_articles)
            comments.update(db_comments)
    return articles, comments


articles, comments = concate_dicts((zeit_lists, 'zeit'), (spon_lists, 'spon'), (fc_lists, 'focus'))
art_ids = list(articles.keys())
com_ids = list(comments.keys())
print(len(articles), len(comments))

def tokenize_body(comments):
    """Tokenizes the comments, tokens are lowercased"""
    tokens = {}
    for idx, com_id in enumerate(comments):
        body = comments[com_id]['body']
        tokenized = [x.lower() for x in nltk.word_tokenize(body, language='german')]
        tokens[com_id] = tokenized
        progressbar(idx, len(comments), reprint=False)
    return tokens
        
    
tokens = tokenize_body(comments)

def create_tagged_objects(tokens):
    """Converts tokens to gensim tagged documents"""
    tagged_docs = {}
    for idx, com_id in enumerate(tokens):
        tagged_doc = d2v.TaggedDocument(words=tokens[com_id], tags=[com_id])
        tagged_docs[com_id]= tagged_doc
        progressbar(idx, len(comments), percentage_step=5, reprint=False)
    return tagged_docs


tagged_docs = create_tagged_objects(tokens)

def make_inverted_index(tagged_docs):
    """Creates and inverted index to quickly find comments containing particular words"""
    inv_index = {}
    for idx, com_id in enumerate(tagged_docs):
        doc = tagged_docs[com_id]
        words = doc.words
        for word in words:
            index_dict = inv_index.setdefault(word, {})
            index_dict[com_id] = index_dict.get(com_id, 0) + 1
        progressbar(idx, len(tagged_docs), reprint=False)
    return inv_index


inv_index = make_inverted_index(tagged_docs)

from scipy.stats import chi2_contingency


def get_aggregation(dicitionary, what='publisher'):
    """Aggregates summary counts of publishers"""
    result_dict = {}
    count=0
    for com_id, item in dicitionary.items():
        which = item[what]
        result_dict[which] = result_dict.get(which, 0) + 1
        #progressbar(count, len(dicitionary))
        count += 1
    return result_dict


def get_word_count(word, index, comments, what='publisher'):
    """Computes a word count per publisher"""
    count_dict = index[word]
    result_count = {}
    for com_id, count in count_dict.items():
        which = comments[com_id][what]
        result_count[which] = result_count.get(which, np.array([0, 0])) + np.array([count, 1])
    return result_count


def get_norm_count(word, index, comments, norm, what='publisher'):
    """Returns a normalized count of a word per publisher
    
    Runs a chi squared test on publisher vs publisher basis.
    """
    tmp = get_word_count(word, index, comments, what)
    result = {}
    for what, counts in tmp.items():
        result[what] = {'frac': counts[1] / float(norm[what]) * 100,
                        'wordcount' : counts[0],
                        'doccount' : counts[1],
                        'total' : norm[what]}
    
    thekeys = list(result.keys())
    tests = {}
    for idx, what in enumerate(thekeys):
        for jdx, what2 in enumerate(thekeys):
            if jdx <= idx:
                continue
            values = [ [result[what]['doccount'], result[what]['total'] - result[what]['doccount']],
                       [result[what2]['doccount'], result[what2]['total'] - result[what2]['doccount']]]
            tests[what + ' vs ' + what2] = chi2_contingency(values)
                
            
    return result, tests


norm_dict = get_aggregation(comments)
norm_dict

# How often is `einigkeit` found in user comments per publisher
get_norm_count('einigkeit', inv_index, comments, norm_dict)

# loads a model from disk, uncomment to train anew
model = d2v.Doc2Vec.load('./zeit_spon_focus_380k_256dim_20epochs_5mincount_8window_5negative_1e4.net')

import random


def get_articles_by_publisher(articles):
    """Returns a nested dict with articles per publisher"""
    articles_by_publisher = {}
    for com_id, article in articles.items():
        publisher = article['publisher']
        publisher_dict = articles_by_publisher.setdefault(publisher, {})
        publisher_dict[com_id] = article
    return articles_by_publisher


def train_test(comments, articles, test_frac = 0.25, model=None):
    """Creates a train test split.
    
    Takes care that comments from the same article are either in the train or test split.
    This avoids data leakage from train to test.
    
    If model is given, recreates training and test set of the model.
    """
    training = {}
    testing = {}
    
    training_articles = set()
    testing_articles = set()
    
    if model is not None:
        print('Using model training set!')
    
    articles_by_publisher = get_articles_by_publisher(articles)
    for publisher, publisher_articles in articles_by_publisher.items():
        narticles = len(publisher_articles)
        up_to = int(narticles*test_frac)
        article_keys = list(publisher_articles.keys())
        random.shuffle(article_keys)
        testing_articles.update(article_keys[:up_to])
        training_articles.update(article_keys[up_to:])
    
    for com_id, comment in comments.items():
        title = comment['title']
        if model is None:
            if title in testing_articles:
                testing[com_id] = comment
            elif title in training_articles:
                training[com_id] = comment
            else:
                print(title)
                print(training_articles)
                raise ValueError('You shall not pass!')
        else:
            if com_id in model.docvecs:
                training[com_id] = comment
            else:
                testing[com_id] = comment
    return training, testing


def count_by_publisher(dictionary):
    """Counts comments per publisher"""
    pub_count = {}
    for comment in dictionary.values():
        publisher = comment['publisher']
        pub_count[publisher] = pub_count.setdefault(publisher, 0) + 1
    return pub_count


random.seed(42)
training, testing = train_test(comments, articles, model=model)
print('Trainig Size: ' + str(count_by_publisher(training)))
print('Test Size:' + str(count_by_publisher(testing)))
training_tagged_docs = [tagged_docs[com_id] for com_id in sorted(training.keys())]
print(len(training_tagged_docs))
print(training_tagged_docs[-3])

if model is None:
    # Train the model if not loaded before
    model = d2v.Doc2Vec(alpha=0.05, min_alpha=0.05, size=256, 
                        window=8, min_count=5, workers=13, sample=1e-4,
                        negative=5)  # use fixed learning rate
    print('Building vocab')
    model.build_vocab(training_tagged_docs)
    print('Vocab length: ' + str(len(model.vocab)))
    epochs = 20
    factor = 0.8
    for epoch in range(epochs):
        # train epoch by epoch
        print('Training epoch %d' % epoch)
        progressbar(epoch, epochs, 2, reprint=False)
        random.shuffle(training_tagged_docs)
        model.train(training_tagged_docs)
        model.alpha *= factor  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
else:
    print('Using exisiting model')

len(model.vocab)

# Uncomment to store newly trained model
#model.save('./zeit_spon_focus_380k_256dim_20epochs_5mincount_8window_5negative_1e4.net')
model.vector_size

model.most_similar(['lügenpresse'])

model.most_similar(['npd'])

model.most_similar(['hitler', 'putin'])

model.most_similar(['brexit', 'griechenland'], ['england'])

model.most_similar(['auto'])     

model.most_similar(['könig', 'frau'], ['mann'])

import numpy as np
from scipy import linalg as la


def pca(data, dims_rescaled_data=2, evecs=None):
    """ Performs a principal component analysis """
    m, n = data.shape
    data = data.copy()
    data -= data.mean(axis=0)
    evals = None
    if evecs is None:
        R = np.cov(data, rowvar=False)
        evals, evecs = la.eigh(R)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        evecs = evecs[:, :dims_rescaled_data]
    return np.dot(evecs.T, data.T).T, evals, evecs

import pandas as pd


def get_doc_vecs(com_ids, model):
    """ Returns the doc2vecs as array. 
    
    In case of training data, the vector is taken from the model's docvecs.
    Test document vectors are inferred using gradient descent.
    """
    dim = model.vector_size
    inputs = np.zeros((len(com_ids), dim))
    for kdx, com_id in enumerate(com_ids):
        try:
            inputs[kdx, :] = model.docvecs[com_id]
        except KeyError:
            # infer the test vector
            inputs[kdx, :] = model.infer_vector(tagged_docs[com_id].words, steps=8)
        progressbar(kdx, len(com_ids), reprint=False)
    return inputs


def comments_to_frame(comments, model):
    """Turns user comments into pandas data frame"""
    com_ids = np.array(sorted(comments.keys()))
    com_range = np.arange(len(com_ids))
    multi_index = pd.MultiIndex.from_arrays([com_ids, com_range])
    columns = list(range(model.vector_size)) + ['publisher', 'rating', 'article_id', 'author']
    data = {}
    docvecs = get_doc_vecs(com_ids, model)
    for col in columns:
        if isinstance(col, int):
            data[col] = docvecs[:, col]
        else:
            data[col] = []
    for com_id in com_ids:
        comment = comments[com_id]
        data['publisher'].append(comment['publisher'])
        data['rating'].append(comment.get('recommendations', np.NaN))
        data['article_id'].append(comment['title'].__hash__())
        data['author'].append(comment['author'])
    data_frame = pd.DataFrame(columns=columns, index=multi_index, data=data)
    return data_frame

training_frame = comments_to_frame(training, model)
testing_frame = comments_to_frame(testing, model)

pcadata, _, evecs = pca(training_frame[list(range(256))])
tepcadata, _, _ = pca(testing_frame[list(range(256))], evecs=evecs)

tepcadata.shape

iszeit = (training_frame['publisher']=='zeit').values
isspon = (training_frame['publisher']=='spon').values
isfocus = (training_frame['publisher']=='focus').values

teiszeit = (testing_frame['publisher']=='zeit').values
teisspon = (testing_frame['publisher']=='spon').values
teisfocus = (testing_frame['publisher']=='focus').values

ssize = 5500  # Number of points to plot
plt.figure(figsize=(5,5))
plt.scatter(pcadata[isspon,0][:ssize],pcadata[isspon,1][:ssize], alpha=0.1, s=5, color='red', label='SPON')
plt.scatter(pcadata[iszeit,0][:ssize],pcadata[iszeit,1][:ssize], alpha=0.1, s=5, label='ZEIT')
plt.scatter(pcadata[isfocus,0][:ssize],pcadata[isfocus,1][:ssize], alpha=0.1, s=5, color='yellow', label='Focus')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend()
plt.grid()

def stratify_frame(data_frame, class_label='publisher', seed=42):
    """Returns stratified data frame with equal amount of comments per publisher"""
    all_classes = {}
    class_col = data_frame[class_label]
    for class_ in class_col:
        all_classes[class_] = all_classes.get(class_, 0) + 1
    min_val = float('inf')
    for class_, count in all_classes.items():
        if count < min_val:
            min_class = class_
            min_val = count
    
    frames = []
    for class_ in all_classes:
        new_frame = data_frame.loc[data_frame[class_label] == class_]
        if class_ != min_class:
            new_frame = new_frame.sample(n=min_val)
        frames.append(new_frame)
    return pd.concat(frames)

sf_training = stratify_frame(training_frame)
sf_testing = stratify_frame(testing_frame)

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plots the confusion matrix"""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=14)
    plt.yticks(tick_marks, labels, fontsize=14)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.grid(True)

# let us train a linear classifier
from sklearn.linear_model import SGDClassifier

sX = sf_training[list(range(model.vector_size))].values
sy = sf_training['publisher'].values

sclf = SGDClassifier(penalty='elasticnet', loss='log', alpha=0.02, n_jobs=3)
sclf.fit(sX,sy)

print('Training Score:' + str(sclf.score(sX, sy)))
tesX = sf_testing[list(range(model.vector_size))].values
tesy = sf_testing['publisher'].values
print('Testing Score: ' + str(sclf.score(tesX, tesy)))

labels = sclf.classes_
scm = confusion_matrix(sy, sclf.predict(sX), labels=labels)
stecm = confusion_matrix(tesy, sclf.predict(tesX), labels=labels)

print(scm)
print(stecm)

plot_confusion_matrix(scm, labels)
plt.figure()
plot_confusion_matrix(stecm, labels)

def print_com_ids(com_ids, comments, howmany=10, maxletter=256):
    """Prints an the first 256 characters of comment with a given `com_id`"""
    for irun in range(min(len(com_ids), howmany)):
        text = comments[com_ids[irun]]['body']
        if len(text) > maxletter:
            text = text[:maxletter] + '...'
        publisher = comments[com_ids[irun]]['publisher']
        rating = comments[com_ids[irun]].get('recommendations', 'N/A')
        print('\n---Comment %s from %s (rating %s) ---' % (com_ids[irun], publisher, str(rating)))
        print(text)


def find_k_best(clf_, frame, k=10, dims=256):
    """Find the k best matching comments per publisher"""
    com_ids = {}
    res_vals = {}
    classes = clf_.classes_
    data = frame[list(range(dims))].values
    try:
        index = frame.index.values
    except ValueError:
        index = []
        for irun in range(len(frame.index)):
            index.append(frame.index[irun])
    decisions = clf_.decision_function(data)
    for kdx, cls_ in enumerate(classes):
        decargs = np.argsort(decisions, 0)
        com_ids[cls_] = []
        res_vals[cls_] = []
        for krun in range(1, k+1):
            com_ids[cls_].append(index[decargs[-krun, kdx]][0])
            res_vals[cls_].append(decisions[decargs[-krun, kdx], :])
    return com_ids, res_vals
        
              
                      

best_com_ids, best_vals = find_k_best(sclf, sf_training)  

# the best classification values are
best_vals

# the best comment ids
best_com_ids

# let's print the prototypes
for class_, theids in best_com_ids.items():
    print('\n\nBEST FOR %s' % class_)
    print_com_ids(theids, comments, maxletter=600)


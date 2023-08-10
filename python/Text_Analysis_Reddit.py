import numpy 
import matplotlib.pyplot
import nltk
import ujson
import re
import time

from __future__ import print_function
from six.moves import zip, range 


get_ipython().magic('matplotlib inline')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn import preprocessing
from collections import Counter, OrderedDict
from nltk.corpus import stopwords
from nltk import SnowballStemmer

get_ipython().magic('matplotlib inline')
nltk.download('stopwords') #download the latest stopwords

def load_reddit(fname, ls_subreddits=[], MIN_CHAR=30):
    """
    Loads Reddit Comments from a json file based on 
    whether they are in the selected subreddits and 
    have more than the MIN_CHARACTERS
    
    Parameters
    ----------
    fname: str
        filename
    ls_subreddits: ls[str]
        list of subreddits to select from 
    MIN_CHAR: int
        minimum number of characters necessary to select
        a comment
        
    Returns
    -------
    corpus: ls[str]
        list of selected reddit comments
    subreddit_id: array[int]
        np.array of indices that match with the ls_subreddit
        index 
    """
    corpus = []
    subreddit_id = []
    with open(fname, 'r') as infile:
        for line in infile:
            dict_reddit_post =  ujson.loads(line)
            subreddit = dict_reddit_post['subreddit']
            n_characters = len( dict_reddit_post['body'] )
            
            if ls_subreddits: #check that the list is not empty
                in_ls_subreddits = subreddit in ls_subreddits
            else:
                in_ls_subreddits = True
            
            grter_than_min = n_characters > MIN_CHAR
            
            if ( grter_than_min and in_ls_subreddits ):
                corpus.append(dict_reddit_post['body'])
                subreddit_id.append(subreddit)
                
    return np.array(corpus), np.array(subreddit_id)

def plot_precision_recall():
    pass

def plot_precision_recall_n(y_true, y_prob, model_name, fname=None):
    """
    create a precision recall curve
    
    Parameters
    ----------
    y_true: ls
        ls of ground truth labels 
    y_prob: ls
        ls of predict probas from model
    model_name: str
        str of model used
    fname: str
        filename to save figure
    
    Returns
    -------
    Plot of precision recall. 
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1] #take every value up to the last one 
    recall_curve = recall_curve[:-1]# take every value up to the last one
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    if(fname):
        plt.savefig(fname)
    plt.show()

get_ipython().run_cell_magic('bash', '#magic function to run a bash command inside of a Jupyter notebook. ', '#unizip the data\ngunzip ./data/RC_2015-05.json.gz')

#grab data from the following subreddits
ls_subreddits = ['SuicideWatch', 'depression']
[corpus, subreddit_id] = load_reddit('./data/RC_2015-05.json', ls_subreddits, MIN_CHAR=30)

#matches are non-word chracters and digits to be replaced with spaces.
RE_PREPROCESS = r'\W+|\d+'  
#get rid of punctuation and make everything lowercase
processed_corpus = np.array( [ re.sub(RE_PREPROCESS, ' ', comment).lower() for comment in corpus] )
Counter(subreddit_id)

#split the data into training and testing sets. 
#refactor this in the test train-split
train_set_size = int(0.8*len(subreddit_id))
train_idx = np.arange(0,train_set_size)
test_idx = np.arange(train_set_size, len(subreddit_id))

train_subreddit_id = subreddit_id[train_idx]
train_corpus = processed_corpus[train_idx]

test_subreddit_id = subreddit_id[test_idx]
test_corpus = processed_corpus[test_idx]

print('Training Labels', Counter(subreddit_id[train_idx]))
print('Testing Labels', Counter((subreddit_id[test_idx])))

#parameters for vectorizer 
ANALYZER = "word" #unit of features are single words rather then phrases of words 
STRIP_ACCENTS = 'unicode'
TOKENIZER = None
NGRAM_RANGE = (0,2) #Range for pharases of words
MIN_DF = 0.01 # Exclude words that have a frequency less than the threshold
MAX_DF = 0.8  # Exclude words that have a frequency greater then the threshold 

vectorizer = CountVectorizer(analyzer=ANALYZER,
                            tokenizer=None, # alternatively tokenize_and_stem but it will be slower 
                            ngram_range=NGRAM_RANGE,
                            stop_words = stopwords.words('english'),
                            strip_accents=STRIP_ACCENTS,
                            min_df = MIN_DF,
                            max_df = MAX_DF)

NORM = None #turn on normalization flag
SMOOTH_IDF = True #prvents division by zero errors
SUBLINEAR_IDF = True #replace TF with 1 + log(TF)
USE_IDF = True #flag to control whether to use TFIDF

transformer = TfidfTransformer(norm = NORM,smooth_idf = SMOOTH_IDF,sublinear_tf = True)

#get the bag-of-words from the vectorizer and
#then use TFIDF to limit the tokens found throughout the text 
start_time = time.time()
train_bag_of_words = vectorizer.fit_transform( train_corpus ) #using all the data on for generating features!! Bad!
test_bag_of_words = vectorizer.transform( test_corpus )
if USE_IDF:
    train_tfidf = transformer.fit_transform(train_bag_of_words)
    test_tfidf = transformer.transform(test_bag_of_words)
features = vectorizer.get_feature_names()
print('Time Elapsed: {0:.2f}s'.format(
        time.time()-start_time))

#relabel our labels as a 0 or 1
le = preprocessing.LabelEncoder() 
le.fit(subreddit_id)
subreddit_id_binary = le.transform(subreddit_id)

#make clear what are the features and what are the labels 
clf = LogisticRegression(penalty='l1')
mdl = clf.fit(train_tfidf, 
              subreddit_id_binary[train_idx])
y_score = mdl.predict_proba( test_tfidf )

auc = roc_auc_score( subreddit_id_binary[test_idx], y_score[:,1])
print("{0:.2f}".format(auc))

plot_precision_recall_n(subreddit_id_binary[test_idx], y_score[:,1], 'LR')

coef = mdl.coef_.ravel()

dict_feature_importances = dict( zip(features, coef) )
orddict_feature_importances = OrderedDict( 
                                sorted(dict_feature_importances.items(), key=lambda x: x[1]) )

ls_sorted_features  = list(orddict_feature_importances.keys())

num_features = 5
subreddit0_features = ls_sorted_features[:5] #SuicideWatch
subreddit1_features = ls_sorted_features[-5:] #depression
print('SuicideWatch: ',subreddit0_features)
print('depression: ', subreddit1_features)

#maybe do something with this crazy indexing: this is python not C!
num_comments = 5
subreddit0_comment_idx = y_score[:,1].argsort()[:num_comments] #SuicideWatch
subreddit1_comment_idx = y_score[:,1].argsort()[-num_comments:] #depression

#convert back to the indices of the original dataset
top_comments_testing_set_idx = np.concatenate([subreddit0_comment_idx, 
                                               subreddit1_comment_idx])

#these are the 5 comments the model is most sure of 
for i in top_comments_testing_set_idx:
    print(
        u"""{}:{}\n---\n{}\n===""".format(test_subreddit_id[i],
                                          y_score[i,1],
                                          test_corpus[i]))

N_TOPICS = 50
N_TOP_WORDS = 10

start = time.time()
corpus_all, subreddit_id_all = load_reddit('./data/RC_2015-05.json',MIN_CHAR=250)
end = time.time()
print('Loading takes {0:2f}s'.format(end-start))

# Get rid of punctuation and set to lowercase  
start = time.time()
processed_corpus_all = [ re.sub( RE_PREPROCESS, ' ', comment).lower() for comment in corpus_all]

#tokenzie the words
bag_of_words_all = vectorizer.fit_transform( processed_corpus_all ) 
end = time.time() 
#grab the features/vocabulary
features_all = vectorizer.get_feature_names()
print("Processing took {}s".format(end - start))

print(Counter(subreddit_id_all), len(subreddit_id_all))

start = time.time()
lda = LatentDirichletAllocation( n_topics = N_TOPICS )
doctopic = lda.fit_transform( bag_of_words_all )
end = time.time() 
print("Processing took {}s".format(end- start)) # takes ~72s for 1 file, ~445s for 5 files


ls_keywords = []
for i,topic in enumerate(lda.components_):
    word_idx = np.argsort(topic)[::-1][:N_TOP_WORDS]
    keywords = ', '.join( features_all[i] for i in word_idx)
    ls_keywords.append(keywords)
    print(i, keywords)
    

num_comments = 50
for comment_id in range(num_comments):
    topic_id = np.argsort(doctopic[comment_id])[::-1][0]
    print('comment_id:',
          comment_id,
          subreddit_id_all[comment_id],
          'topic_id:',
          topic_id,
          'keywords:',
           ls_keywords[topic_id])
    print('---')
    print(corpus_all[comment_id])
    print('===')


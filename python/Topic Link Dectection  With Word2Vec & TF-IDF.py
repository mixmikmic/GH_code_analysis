import os
import numpy as np
from gensim.models.word2vec import Word2Vec
#from __future__ import print_function, division

if not os.path.exists("data/embed.dat"):
    print("Caching word embeddings in memmapped format...")
    
    wv = Word2Vec.load_word2vec_format(
        "/home/skillachie/Downloads/GoogleNews-vectors-negative300.bin",
        binary=True)
    wv.init_sims(replace=False)
    
    fp = np.memmap("data/embed.dat", dtype=np.double, mode='w+', shape=wv.syn0norm.shape)
    fp[:] = wv.syn0norm[:]
    with open("data/embed.vocab", "w") as f:
        for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
            print(w, file=f)
    del fp, wv

W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with open("data/embed.vocab") as f:
    vocab_list = map(str.strip, f.readlines())
    

#print("Features:",  ", ".join(train_vect.get_feature_names()))
vocab_dict = {w: k for k, w in enumerate(vocab_list)}



from parse_tdt5 import *
import sys

print([10*"<B>"])

#Tokenize and clean
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessing(text,stem=False,stop=False,sent=False):
    
    
    # Remove punctuations
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    
    tokens = word_tokenize(text)
    
    if stop:
        stop = stopwords.words('english')
        tokens =[word for word in tokens if word not in stop]
        tokens = [word.lower() for word in tokens]

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
        
            
    if sent:
        tokens = ' '.join(tokens)
        
        
    return tokens

tdt_annotation_dir = "/home/skillachie/Downloads/tdt/annotations/tdt5_topic_annot/data/annotations"
tdt_corpus_dir = "/home/skillachie/Downloads/tdt/data"

link_det_answers = read_link_detection_answer_key(tdt_annotation_dir,'lnk_SR=nwt_TE=man,eng_1000.key')
link_texts = []
link_pairs = []
link_answers = []

for link in link_det_answers:
    doc1 = read_doc(tdt_corpus_dir,link['file1_id'],link['file1_docno'],                            task_type='LINK')
    
    doc2 = read_doc(tdt_corpus_dir,link['file2_id'],link['file2_docno'],                            task_type='LINK')
    
    if doc2 is None:
        continue
    
    if doc1 is None:
        continue 
    
    
    link_texts.append(doc1)
    link_texts.append(doc2)
    
    link_pairs.append((doc1,doc2))
    link_answers.append(link['answer'])
        

print(len(link_answers))

from sklearn.cross_validation import train_test_split
train_articles_pre, test_articles, train_target_pre, test_target =                         train_test_split(link_pairs,link_answers, test_size=0.20, random_state=13)
    
#Divide into Dev and Train
train_articles, dev_articles, train_target, dev_target =    train_test_split(train_articles_pre,train_target_pre, test_size=0.20, random_state=13)

from gensim.models import Doc2Vec, Word2Vec
import gensim.models.doc2vec
import numpy as np

from collections import OrderedDict
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
import sys

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

# Create vectorizer using all the vocab of the dataset
def get_all_articles_text(articles):
    all_articles = []

    for article in articles:
        all_articles.append(article[0])
        all_articles.append(article[1])
    return all_articles

train_articles_txt = get_all_articles_text(train_articles)
dev_articles_txt = get_all_articles_text(dev_articles)

print(train_articles_txt[66])

#train_vect = create_vectorizer(train_articles_txt + test_articles_txt)
print(len(train_articles_txt))
print(len(dev_articles_txt))

from sklearn.feature_extraction.text import TfidfVectorizer

#Test
tfidf_vec_test = TfidfVectorizer(sublinear_tf=True,
                            use_idf=True)


tfidf_vec_test.fit_transform(train_articles_txt)

#Dev
tfidf_vec_dev = TfidfVectorizer(sublinear_tf=True,
                            use_idf=True)

tfidf_vec_dev.fit_transform(dev_articles_txt)

word2vec_model = Word2Vec.load_word2vec_format('/home/skillachie/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
word2vec_model.init_sims(replace=False)


def get_word_vectors(tokens):
    
    article_word_vectors = []
    article_model_vocab = []
    
    for token in tokens:
        #print(token)
        word_vec = []
        try:
            word_vec = word2vec_model[token]
            article_model_vocab.append(token)
            #print(token)
        except Exception:
                #print word
            continue
            #word_vec = np.zeros(300)
                       
    
        #article_word_vectors.append(word_vec)
        
        
    return article_model_vocab


def aggr_word_vectors(word_vectors):
    #print(len(word_vectors))
    stack = np.vstack(word_vectors)
    doc_vec = np.mean(stack,axis=0)
    return doc_vec

def gen_doc2vec_vocab_features(articles,tf_vec):
     
    new_feature_vectors = []
    
    for article_tuple in articles:

        
        #print(article_tuple)
        #print("-------------------------")
        feat_dict = defaultdict(dict)
        
        article1_txt = article_tuple[0]
        article2_txt = article_tuple[1]
        
        if article_tuple[0] is None:
            article1_txt = 'bad data' #hack remove
             
        
        if article_tuple[1] is None:
            article2_txt = 'bad data' #hack remove
        
        
        article1_tf_idf = tf_vec.transform([article1_txt])
        article2_tf_idf = tf_vec.transform([article2_txt])
        
        feat_dict['article1_tfidf'] = article1_tf_idf
        feat_dict['article2_tfidf'] = article2_tf_idf
        
        article1_tokens = word_tokenize(article1_txt)
        article2_tokens = word_tokenize(article2_txt)
        
        article1_wordvec_vobab = get_word_vectors(article1_tokens)
        article2_wordvec_vobab = get_word_vectors(article2_tokens)
        
        #print(article1_wordvec_vobab)
        #sys.exit(1)
          
        #doc1_vec = aggr_word_vectors(article1_wordvecs)
        #doc2_vec = aggr_word_vectors(article2_wordvecs)

        
        #feat_dict['article1_vec'] = doc1_vec
        feat_dict['article1_vocab'] = article1_wordvec_vobab
        #feat_dict['article2_vec'] = doc2_vec
        feat_dict['article2_vocab'] = article2_wordvec_vobab
        
        new_feature_vectors.append(feat_dict)
        
    return new_feature_vectors



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def check_link(articles,method=None):
           
    pred_answer = None
    
    if method is None:
        sim = word2vec_model.n_similarity(articles['article1_vocab'], articles['article2_vocab'])
        
        if sim >= 0.88:
            pred_answer = "TARGET"
        else:
            pred_answer = "NONTARGET"
        
    else:
        sim  = cosine_similarity(articles['article1_tfidf'], articles['article2_tfidf'] )[0][0]
    
        
        if sim >= 0.2:
            pred_answer = "TARGET"
        else:
            pred_answer = "NONTARGET"
        
    return pred_answer

def connect_links(features,method=None):
    
    pred_answers = []
    for article_features in features:
        answer = check_link(article_features,method)
        pred_answers.append(answer)
    return pred_answers

#Dev
word2vec_features_dev = gen_doc2vec_vocab_features(dev_articles,tfidf_vec_dev)


pred_answers_word2vec_dev = connect_links(word2vec_features_dev)
pred_answers_tfidf_dev = connect_links(word2vec_features_dev,"TF-IDF")


#Test
word2vec_features_test = gen_doc2vec_vocab_features(test_articles,tfidf_vec_test)
pred_answers_word2vec_test = connect_links(word2vec_features_test)
pred_answers_tfidf_test = connect_links(word2vec_features_test,"TF-IDF")

print(len(pred_answers_word2vec_dev))
print(len(pred_answers_word2vec_test))
print(len(train_articles))
#print(len(pred_answers_tfidf))


#print(len(test_target))

def event_pair_evaluation(predicted_answers, answers):
    
    neg_correct = 0
    neg_incorrect = 0
    pos_correct = 0
    pos_incorrect = 0
    
    negatives = 0
    positives = 0
    
    total = len(answers)

    for pred_ans,ans in zip(predicted_answers,answers):
        
        if ans == "NONTARGET":
            negatives +=1

            if pred_ans == ans:
                neg_correct += 1
            else:
                neg_incorrect += 1
    
        if ans == "TARGET":
            positives +=1
    
            if pred_ans == ans:
                pos_correct += 1
            else:
                pos_incorrect += 1
    
     
        
    pos_correct = np.float64(pos_correct)
    neg_correct = np.float64(neg_correct)
        
            
    print("Negative: %f " %( neg_correct/negatives) )
    
    print(neg_correct)
    print(negatives)
    

    
    
    
    
    print("Positive: %f " %( pos_correct/positives) )
    print(pos_correct)
    print(positives)
    #print(/np.float64(positives))
    
    correct = neg_correct + pos_correct
        
    return (float(correct) / total)*100   

#word2vec_score_dev = event_pair_evaluation(pred_answers_word2vec_dev,dev_target)
#print(word2vec_score_dev)

word2vec_score_test = event_pair_evaluation(pred_answers_word2vec_test,test_target)
print(word2vec_score_test)

tfidf_score_dev = event_pair_evaluation(pred_answers_tfidf_dev,dev_target)
print(tfidf_score_dev)

#tfidf_score_test = event_pair_evaluation(pred_answers_tfidf_test,test_target)
#print(tfidf_score_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def binarize_answers(answers):
    
    bin_answers = []
    
    for answer in answers:
        if answer == 'TARGET':
            bin_answers.append(1)
        else:
            bin_answers.append(0)
    return bin_answers 


def get_metrics(target,predict):
    
    accuracy = accuracy_score(target,predict)
    recall = recall_score(target,predict)
    f1 = f1_score(target,predict)

    #print(accuracy)
    print("Accuracy: %f , Recall: %f , F1-Score: %f" %(accuracy, recall, f1))
    

    
#Dev    
bin_pred_word2vec_dev = binarize_answers(pred_answers_word2vec_dev)
bin_pred_tfidf_dev = binarize_answers(pred_answers_tfidf_dev)
bin_dev_target = binarize_answers(dev_target)


get_metrics(bin_dev_target,bin_pred_word2vec_dev)
get_metrics(bin_dev_target,bin_pred_tfidf_dev)
    

    
print("-------------------------------")

#Test
bin_pred_word2vec_test = binarize_answers(pred_answers_word2vec_test)
bin_pred_tfidf_test = binarize_answers(pred_answers_tfidf_test)
bin_test_target = binarize_answers(test_target)


get_metrics(bin_test_target,bin_pred_word2vec_test)
get_metrics(bin_test_target,bin_pred_tfidf_test)












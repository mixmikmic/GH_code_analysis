get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.notebook_repr_html", True)
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("poster")

sample_df = pd.read_csv("sample_cases.csv")

sample_df.head()

# training, test data split 
trainingcoln = pd.read_csv('traintestarray.csv',sep=',',header=None).values.ravel()
sample_df['training'] = trainingcoln

import re 
regex1 = r"\(.\)" 

from pattern.en import parse
from pattern.en import pprint
from pattern.en import conjugate, lemma, lexeme
from pattern.vector import stem, PORTER, LEMMA
from sklearn.feature_extraction import text
import string

#stopwords and punctuation
stopwords=text.ENGLISH_STOP_WORDS
punctuation = list('.,;:!?()[]{}`''\"@#$^&*+-|=~_')

def get_parts(opinion):
    oplow = opinion.lower()
    #REMOVING CHARACTERS: we have ugly text, and remove unnecssary characters.
    oplow = unicode(oplow, 'ascii', 'ignore') #remove non-unicode characters 
    oplow = str(oplow).translate(string.maketrans("\n\t\r", "   ")) #remove characters like \n 
    #justices (eg, Justice Breyer) are referred to as J. (eg,Breyer, J.); we remove the J., also JJ. for plural
    oplow = oplow.replace('j.','')
    oplow = oplow.replace('jj.','')
    oplow = oplow.replace('c.','') #remove C. for chief justice 
    oplow = oplow.replace('pp.','') #page numbers
    oplow = oplow.replace('  ','') #multiple spaces
    oplow = ''.join([i for i in oplow if not i.isdigit()]) #remove digits 
    oplow=re.sub(regex1, ' ', oplow)
    #Remove the Justia disclaimer at the end of the case, if it appears in the string
    justiadisclaimer = "disclaimer: official"
    if justiadisclaimer in oplow: 
        optouse = oplow.split(justiadisclaimer)[0]
    else:
        optouse = oplow
    
    #GET A LIST OF PRECEDENTS CITED IN THE OPINION 
    wordslist = optouse.split()
    #find precedents based on string 'v.' (eg, 'Brown v. Board')
    indices = [i for i in range(len(wordslist)) if wordslist[i]=='v.']
    precedents = [wordslist[i-1]+ ' ' + wordslist[i]+ ' ' + wordslist[i+1] for i in indices]
    
    #remove precedents, as we have already accounted for these
    for precedent in precedents:
        optouse = optouse.replace(precedent,'')
    
    #PARSE INTO LIST OF LISTS --> GET WORDS
    parsed = parse(optouse,tokenize=True,chunks=False,lemmata=True).split()
    verbs = [] 
    nouns = [] 
    adjectives = [] 
    foreign = [] 
    i=0
    #Create lists of lists of verbs, nouns, adjectives and foreign words in each sentence.
    for sentence in parsed: #for each sentence 
        verbs.append([])
        nouns.append([])
        adjectives.append([])
        foreign.append([])
        for token in sentence: #for each word in the sentence 
            if token[0] in punctuation or token[0] in stopwords or len(token[0])<=2:
                continue
            wordtouse = token[0]
            for x in punctuation:
                wordtouse = wordtouse.replace(x,' ') #if punctuation in word, take it out
            if token[1] in ['VB','VBZ','VBP','VBD','VBN','VBG']:
                verbs[i].append(lemma(wordtouse)) #append the lemmatized verb (we relemmatize because lemmata in parse does not seem to always work)
            if token[1] in ['NN','NNS','NNP','NNPS']:
                nouns[i].append(lemma(wordtouse))
            if token[1] in ['JJ','JJR','JJS']:
                adjectives.append(lemma(wordtouse))
            if token[1] in ['FW']:
                foreign.append(wordtouse)  
        i+=1  
    #Zip together lists so each tuple is a sentence. 
    out=zip(verbs,nouns,adjectives,foreign)
    verbs2 = []
    nouns2 = []
    adjectives2 = []
    foreign2 = []
    for sentence in out: 
        if sentence[0]!=[] and sentence[1]!=0: #if the sentence has at least one verb and noun, keep it. Otherwise, drop it.
            if type(sentence[0])==list: 
                verbs2.append(sentence[0])
            else: 
                verbs2.append([sentence[0]]) #if verb is a string rather than a list, put string in list
            if type(sentence[1])==list:
                nouns2.append(sentence[1])
            else:
                nouns2.append([sentence[1]])
            if type(sentence[2])==list:
                adjectives2.append(sentence[2])
            else:
                adjectives2.append([sentence[2]])
            if type(sentence[3])==list:
                foreign2.append(sentence[3])
            else:
                foreign2.append([sentence[3]])
    return(verbs2,nouns2,adjectives2,foreign2,precedents)

get_ipython().run_cell_magic('time', '', 'verbwords = []\nnounwords = []\nadjwords = []\nforwords = []\nprecedents_all = []\nfor op in sample_df.text:\n    verbs,nouns,adjectives,foreign,precedents = get_parts(op)\n    verbwords.append(verbs)\n    nounwords.append(nouns)\n    adjwords.append(adjectives)\n    forwords.append(foreign)\n    precedents_all.append(precedents)')

issue_areas = sample_df.issueArea.tolist()

#create precedents vocab
precedents_vocab = list(set([precedent for sublist in precedents_all for precedent in sublist]))
#create other vocabs
verbvocab = list(set([word for sublist in verbwords for subsublist in sublist for word in subsublist]))
nounvocab = list(set([word for sublist in nounwords for subsublist in sublist for word in subsublist]))
adjvocab = list(set([word for sublist in adjwords for subsublist in sublist for word in subsublist]))
forvocab = list(set([word for sublist in forwords for subsublist in sublist for word in subsublist]))

#dictionaries: id --> word
id2prec = dict(enumerate(precedents_vocab))
id2verb = dict(enumerate(verbvocab))
id2noun = dict(enumerate(nounvocab))
id2adj = dict(enumerate(adjvocab))
id2for = dict(enumerate(forvocab))
#dictionaries: word --> id
prec2id = dict(zip(id2prec.values(),id2prec.keys()))
verb2id = dict(zip(id2verb.values(),id2verb.keys()))
noun2id = dict(zip(id2noun.values(),id2noun.keys()))
adj2id = dict(zip(id2adj.values(),id2adj.keys()))
for2id = dict(zip(id2for.values(),id2for.keys()))

#this function takes a list of words, and outputs a list of tuples 
counter = lambda x:list(set([(i,x.count(i)) for i in x]))

#corpus_creator takes a list of lists of lists like verbwords, or a list of lists like precedents_all. 
#It also takes a word2id dictionary.
def corpus_creator(sentence_word_list,word2id):
    counter = lambda x:list(set([(word2id[i],x.count(i)) for i in x]))
    op_word_list = []
    if type(sentence_word_list[0][0])==list: #if list of lists of lists 
        for opinion in sentence_word_list: 
            #for each list (which corresponds to an opinion) in sentence_word_list, get a list of the words
            op_word_list.append([word for sublist in opinion for word in sublist])
    else: #if list of lists 
        op_word_list = sentence_word_list
    corpus = []
    for element in op_word_list: 
        corpus.append(counter(element))
    return(corpus)

import gensim

corpus = corpus_creator(nounwords,noun2id)

get_ipython().run_cell_magic('time', '', '# model with noun corpus, 5 topics\nlda1a = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2noun, num_topics=5, update_every=1, chunksize=200, passes=1)')

lda1a.print_topics()

get_ipython().run_cell_magic('time', '', '# model with noun corpus, 10 topics\nlda1b = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2noun, num_topics=10, update_every=1, chunksize=200, passes=1)')

lda1b.print_topics()

# takes a corpus and a number of words, and returns a matrix in which the element at row i and column j is the number of
# occurrences of word j in document i.
def corpus_to_mat(corpus, num_words):
    n = len(corpus)
    M = np.zeros((n, num_words))
    for i,doc in enumerate(corpus):
        for word,count in doc:
            M[i][word] = count
    return M

get_ipython().run_cell_magic('time', '', "#get noun corpus\nnouncorpus = corpus_creator(nounwords,noun2id)\nnoun_train_corpus = [nouncorpus[i] for i in range(len(nouncorpus)) if sample_df['training'][i]==1]\nnoun_test_corpus = [nouncorpus[i] for i in range(len(nouncorpus)) if sample_df['training'][i]==0]\n\n#get verb corpus\nverbcorpus = corpus_creator(verbwords,verb2id)\nverb_train_corpus = [verbcorpus[i] for i in range(len(verbcorpus)) if sample_df['training'][i]==1]\nverb_test_corpus = [verbcorpus[i] for i in range(len(verbcorpus)) if sample_df['training'][i]==0]\n\n#get adjective corpus\nadjcorpus = corpus_creator(adjwords,adj2id)\nadj_train_corpus = [adjcorpus[i] for i in range(len(adjcorpus)) if sample_df['training'][i]==1]\nadj_test_corpus = [adjcorpus[i] for i in range(len(adjcorpus)) if sample_df['training'][i]==0]\n\n#get foreign corpus\nforcorpus = corpus_creator(forwords,for2id)\nfor_train_corpus = [forcorpus[i] for i in range(len(forcorpus)) if sample_df['training'][i]==1]\nfor_test_corpus = [forcorpus[i] for i in range(len(forcorpus)) if sample_df['training'][i]==0]\n\n#get precedents corpus\npreccorpus = corpus_creator(precedents_all,prec2id)\nprec_train_corpus = [preccorpus[i] for i in range(len(preccorpus)) if sample_df['training'][i]==1]\nprec_test_corpus = [preccorpus[i] for i in range(len(preccorpus)) if sample_df['training'][i]==0]")

from sklearn.feature_extraction.text import TfidfTransformer
#this function takes a training matrix of size n_documents_training*vocab_size and a test matrix
#of size n_documents_test*vocab_size. The function outputs the corresponding tfidf matrices.
#Note that we fit on the training data, and then apply that fit to the test data.
def tfidf_mat_creator(trainmatrix,testmatrix):
    tf_idf_transformer=TfidfTransformer()
    tfidf_fit = tf_idf_transformer.fit(trainmatrix)
    tfidf_train = tfidf_fit.transform(trainmatrix).toarray()
    tfidf_test = tfidf_fit.transform(testmatrix).toarray()
    return(tfidf_train,tfidf_test)

noun_tfidf_mat_train,noun_tfidf_mat_test = tfidf_mat_creator(corpus_to_mat(noun_train_corpus, len(nounvocab)),
                                   corpus_to_mat(noun_test_corpus, len(nounvocab)))

# takes a tfidf matrix and returns the corresponding matrix
def tfidf_to_corpus(tfidf_mat): #takes as input: matrix of size n_documents*vocabulary size
    tfidfcorpus = []
    i=0 #keep track of document you are on
    for doc in tfidf_mat: #for each case
        tfidfcorpus.append([])
        j=0
        for word in doc: #for each word in the vocabulary, append tuple (wordid,num_times_word_used)
            tfidfcorpus[i].append((j,tfidf_mat[i][j])) 
            j+=1
        i+=1
    return(tfidfcorpus) 

tfidf_noun_corpus = tfidf_to_corpus(noun_tfidf_mat_train)

get_ipython().run_cell_magic('time', '', '# model with noun corpus (tf-idf weighted), 5 topics\nlda2a = gensim.models.ldamodel.LdaModel(corpus=tfidf_noun_corpus, id2word=id2noun, num_topics=5, update_every=1, chunksize=200, passes=1)')

lda2a.print_topics()

get_ipython().run_cell_magic('time', '', '# model with noun corpus (tf-idf weighted), 10 topics\nlda2b = gensim.models.ldamodel.LdaModel(corpus=tfidf_noun_corpus, id2word=id2noun, num_topics=10, update_every=1, chunksize=200, passes=1)')

lda2b.print_topics()

for bow in corpus[0:100:5]:
    print lda2a.get_document_topics(bow)
    print " ".join([id2noun[e[0]] for e in bow])
    print "=========================================="

lsi1a = gensim.models.lsimodel.LsiModel(corpus=tfidf_noun_corpus, id2word=id2noun, num_topics=5, chunksize=200)

lsi1a.print_topics()

lsi1b = gensim.models.lsimodel.LsiModel(corpus=tfidf_noun_corpus, id2word=id2noun, num_topics=10, chunksize=200)

lsi1b.print_topics()




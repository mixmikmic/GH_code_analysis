from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from random import shuffle
import numpy as np
import hashlib
import treetaggerwrapper
import sys
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from os.path import basename


reload(sys)  
sys.setdefaultencoding('utf8')

DEPTH_SEARCH=[5,10,30, 50, 100, 200]
NTREES_SEARCH=[5,10,30, 50, 100, 200]
TEST_SIZE=0.33

# load POS tagggers. You can specify the location of the treetagger installation through TAGDIR as a param, but it is better to set the environment variables.
TAGGERS = {'en': treetaggerwrapper.TreeTagger(TAGLANG='en'),
           'fr': treetaggerwrapper.TreeTagger(TAGLANG='fr'),
           'it': treetaggerwrapper.TreeTagger(TAGLANG='it')}

# check if a lexicon entry appears in a given sentence (that is POS annotated). In some cases the dictionaries consists of verbs only, or other non-specific POS
lexicon_subset = lambda dict_entries, tags, filter_val: [token for token in tags if len(token) == 3 and token[1].startswith(filter_val) and token[2] in dict_entries]

'''
    Reads a lexicon where each line contains a single word
'''
def read_lexicon(file_name, filter_key='#'):
    f = open(file_name, 'r')
    lexicon = [line.strip() for line in f if
               not line.startswith(filter_key) and len(line.strip()) != 0 and " " not in line.strip()]
    f.close()
    return lexicon

'''
    Return the tf-idf scores of all the dictionary entries that are present in a statement.
'''
def tfidf_scorer(tf, dict_entries, tfidf_model, tfidf_dict):
    scores = []
    for dict_entry in dict_entries:
        #each entry has the raw token, POS, and the lemmatized token
        if len(dict_entry) != 3 or dict_entry[0].lower() not in tfidf_dict.token2id:
            continue

        raw_token = dict_entry[0].lower()
        scores.append(tf[raw_token] * tfidf_model.idfs[tfidf_dict.token2id[raw_token]])

    return scores

'''
    We need to compute the TF-IDF scores for the tokens in our statements. We do this across languages.
'''

def compute_tfidf_scores():
    tfidf_models = dict()
    corpus_dicts = dict()
    for lang in filenames_raw:
        f = open(filenames_raw[lang], 'rb')
        # this represents our document collection, which we represent as a matrix where
        # each cell represents a word in a document
        statements = [line.strip().split('\t')[-2].lower().split(' ') for line in f]

        # dictionary of tokens
        dct = Dictionary(statements)
        corpus = [dct.doc2bow(statement) for statement in statements]
        model = TfidfModel(corpus)  # fit model
        tfidf_models[lang] = model
        corpus_dicts[lang] = dct

        print 'Finished computing TF-IDF model for %s.' % lang
    return tfidf_models, corpus_dicts

'''
    Annotate a sentence with POS tags and additionally lematize the tokens.
'''
def tag_statement(lang, statement):
    tagger = TAGGERS[lang]
    #annotate the statement with tags, which consist of the "word\tPOS\tlemma".
    tags = tagger.tag_text(statement)
    #parse them into a more structured representation, where each entry consists  word=u'is', pos=u'VBZ', lemma=u'be'

    parsed_tags = treetaggerwrapper.make_tags(tags)
    return parsed_tags
    

filenames={}
filenames_raw={}

"""
to use this, you will need: 
1) features from https://drive.google.com/open?id=1JZu67psmj2Eou2-8wQEJk4kAQfg8GDs2, to be placed in ../fastText_multilingual/features
"""
languages=['en']
language_extended=['english']
feadir='../fastText_multilingual/features/'
rawdir='../data_clean/'
dictdir = '../data/dictionaries/'


def load_languages():
    for lan,lext in zip(languages,language_extended):
        filenames[lan]=feadir+lan+'.tsv' #files with vectors
        filenames_raw[lan]=rawdir+lext+'.tsv' #files with raw text

        # load the dictionary file paths
        dictionaries[lan] = {f.replace('.txt', ''): dictdir + f for f in os.listdir(dictdir) if f.startswith(lan + '_')}

def count_negatives(negatives,positives):
    """
    for balanced data, we need to know how many negatives are out there
    """
    proportion={}
    allneg=0
    for lan in languages:
        proportion[lan]=len(negatives[lan])/float(len(negatives[lan])+len(positives[lan]))
        allneg+=len(negatives[lan])
    print 'proportion of negatives per language'
    print proportion
    return allneg

def get_values_for_crossvalidation(positives,negatives,features):
    """
    positives: list of positives
    negatives: list of negatives
    features: list of feature dictionaries, per type
    """
    values=[]
    y=[]
    ids=[]
    for lan in languages:
        shuffle(positives[lan])
        alldata=set(negatives[lan]+positives[lan][:len(negatives[lan])])
        ids=ids+list(alldata)
        for id in alldata:
            v=[]
            for f in features: #for every type of feature
                if isinstance(f[id], int):
                    v.append(f[id])
                else:
                    for element in f[id]: #append element of feature
                        v.append(element)
            values.append(np.nan_to_num(np.asarray(v)))
            y.append(labels[id])          
    #reshuffle everything for cross_validaton
    ind=range(len(y))
    shuffle(ind)
    y2=[y[i] for i in ind]
    values2=[values[i] for i in ind]
    ids2=[ids[i] for i in ind]
    return y2,values2,ids2

def perform_gridsearch_withRFC(values,y):
    """
    values: list of feature vectors
    y: labels
    returns
    max_ind: depth and estimator values
    max_val: crossval prediction accuracy
    scores: all-scores for each combination of depth and nestimators
    """
    scores={}
    #performs cross_validation in all combiantions
    for d in DEPTH_SEARCH:
        for n in NTREES_SEARCH:
            clf = RandomForestClassifier(max_depth=d, n_estimators=n)
            s = cross_val_score(clf, values, y)
            print s
            scores[str(d)+' '+str(n)]=np.mean(s)
    #computes best combination of parameters
    max_ind=''
    max_val=0
    for s in scores:
        if scores[s]>max_val:
            max_val=scores[s]
            max_ind=s
    print max_ind
    print max_val
    return max_ind,max_val,scores

def train_test_final(val_train,val_test,y_train,d,n):
    """
    just using a Random Forestc classifier on a train/test split for deployment 
    returns model and probability on the test set
    """
    clf = RandomForestClassifier(max_depth=d, n_estimators=n)
    clf.fit(val_train,y_train)
    prob=clf.predict_proba(val_test)
    return clf,prob

def print_top_bottom_sentences(prob,ids_test,y_test,text,labels):
    """
    here we are displaying the 
    """
    pos_proba=(np.asarray(prob).T)[1]
    indexes=np.argsort(-np.asarray(pos_proba))
    for i in indexes[:10]:
        print text[ids_test[i]]
        print y_test[i]
        print labels[ids_test[i]]#checking
    print ('********************************')
    for i in indexes[-10:]:
        print text[ids_test[i]]
        print y_test[i]
        print pos_proba[i]
        print labels[ids_test[i]]#checking

load_languages()

'''
    Load all the dictionaries we use for classifying the statements. We load the dictionaries for all languages
    alongside which we assign a POS tag letter which we use to filter the tokens in a statement when we consider
    if they contain a dictionary entry or not.
'''


def load_dicts():
    if len(dictionaries) == 0:
        load_languages()

    feature_dicts = {}

    for lang in dictionaries:
        feature_dicts[lang] = {}
        for dict_name, dict_file in dictionaries[lang].iteritems():
            dict_data = read_lexicon(dict_file)

            filter_tag = 'V'
            if 'hedges' in dict_name:
                filter_tag = ''
            feature_dicts[lang][dict_name] = (dict_data, filter_tag)

    return feature_dicts

'''
    We need to compute the TF-IDF scores for the tokens in our statements. We do this across languages.
'''
def compute_tfidf_scores():
    tfidf_models = dict()
    for lang in filenames_raw:
        f = open(filenames_raw[lang], 'rb')
        #this represents our document collection, which we represent as a matrix where 
        #each cell represents a word in a document
        statements = [line.strip().lower().split(' ') for line in f]
        
        #dictionary of tokens
        dct = Dictionary(statements)
        corpus = [dct.doc2bow(statement) for statement in statements]
        model = TfidfModel(corpus)  # fit model
        tfidf_models[lang] = model
        
        print 'Finished computing TF-IDF model for %s.' % lang       
        

'''
    For each statement, we check which of the dictionary entries are present. We will consider a flat feature where
    the each entry in a dictionary (along with the dictionary name) is marked as either true or false.
'''


def extract_dict_features(outdir):
    # compute all the tf idf models
    tfidf_models, corpus_dicts = compute_tfidf_scores()
    feature_dicts = load_dicts()

    for lang in filenames_raw:
        f = open(filenames_raw[lang], 'rb')
        fout = open(outdir + '/' + basename(filenames_raw[lang]).replace('.tsv', '') + "_tfidf_features.tsv", 'a+')

        # get the TF-IDF model for this language
        lang_tfidf = tfidf_models[lang]
        lang_corpus_dict = corpus_dicts[lang]

        # get the feature dictionaries for this language
        lang_dict = feature_dicts[lang]
        num_dicts = len(lang_dict)
        
        #write the column header.
        fout.write('line_index\t' + '\t'.join(lang_dict.keys()) + '\n')

        out_str = ''
        for idx, line in enumerate(f):
            print 'Processing line %d' % idx
            data = line.strip().split('\t')
            statement = data[-2]

            # annotate the statement with POS tags
            statement_pos_tags = tag_statement(lang, unicode(statement))

            # count the frequency of the different entries
            statement_token_freq = Counter([token[0].lower() for token in statement_pos_tags])

            # we compute the average TF-IDF scores for each dict entry
            tfidf_scores = np.zeros(num_dicts)

            for dict_idx, dict_name in enumerate(lang_dict):
                dict_data = lang_dict[dict_name][0]
                dict_tag_filter = lang_dict[dict_name][1]

                subset_matching_tokens = lexicon_subset(dict_data, statement_pos_tags, dict_tag_filter)

                if len(subset_matching_tokens) != 0:
                    tfidf_vector = tfidf_scorer(tf=statement_token_freq, dict_entries=subset_matching_tokens,
                                                tfidf_model=lang_tfidf, tfidf_dict=lang_corpus_dict)

                    total = 1 if len(tfidf_vector) == 0 else len(tfidf_vector)
                    tfidf_scores[dict_idx] = sum(tfidf_vector) / total

            out_str += str(idx) + ',' + ','.join(map(str, tfidf_scores)) + '\n'
            if len(out_str) > 10000:
                fout.write(out_str)
                out_str = ''
        fout.write(out_str)
        print 'Finished computing features for %s' % filenames_raw[lang]

"""
raw header is:
entity_id	revision_id	timestamp entity_title	section	start	offset	statement label
feature header is:
entity_id	revision_id	timestamp entity_title	section	start	offset	 label feature
"""
labels={} #whether it needs a citation or not
vectors={} #the word vectors aligned to english
main={} #is it the main section?
factive={} # does the statement contain a factive verb?
implicative={} # does the statement contain an implicative verb?
hedges={} # does the statement contain hedges?
assertive={} # does the statement contain any assertive verb?
report={}
language={} #which language is the article from
pages={} #length of the page
start={} #starting point of the statement in the page
pagelength={} #page length, this is for future use, if we want to track where the statement is placed in the page
positives={}#statements with citation
negatives={}#statements without citation
text={}#raw text
for lan in languages:
    positives[lan]=[] #stores the statements needing a citation
    negatives[lan]=[] #stores the statements without a citation (much less than the positives)
    fraw=open(filenames_raw[lan]) #each line in fraw correspond to the line in f
    #for each line in the vector file, record various parameters and then store the corresponding raw text with the same identifier
    with open(filenames[lan]) as f:
        for line in f:
            unique=hashlib.sha224(line).hexdigest() #unique identifier of this line
            #first, we store the raw statement text from the raw file
            lineraw=fraw.readline() #line with raw text
            rowraw=lineraw[:-1].split('\t')
            text[unique]=rowraw[-2] #where the text is placed in the line
            
            #now, we can get features
            row=line.split('\t')
            labels[unique]=int(row[-2])#where the label sits in the feature file
            txt = unicode(rowraw[-2], errors='ignore')

            #we need to pre-process the statement by tokenizing it and annotating with POS tags.
            statement_pos_tags = tag_statement(lan, txt)

            #first append to lists of positives and negatives depending on the label
            if labels[unique]==1:
                positives[lan].append(unique)
            else:
                negatives[lan].append(unique)
            #store features
            vectors[unique]=[float(r) for r in row[-1].split(',')]
            main[unique]= 1 if row[4]=='MAIN_SECTION'else 0
            
            #TODO: we need to add here the tfidf scores for the dictionary based features
            

            language[unique]=lan
            pages[unique]=int(row[0])
            beginning=int(row[5])
            offset=int(row[6])
            l=beginning+offset
            try:
                base=pagelength[row[0]]
                pagelength[row[0]]=l if l>base else base
            except:
                pagelength[row[0]]=l
            start[unique]=beginning

allneg=count_negatives(negatives,positives)
print allneg

print set(factive.values())

# factive implicative report hedges assertive

print('all')
y,values,ids=get_values_for_crossvalidation(positives,negatives,[factive,implicative,report,hedges,assertive])
max_ind,max_val,scores=perform_gridsearch_withRFC(values,y)
print('all+main')
y,values,ids=get_values_for_crossvalidation(positives,negatives,[factive,implicative,report,hedges,assertive,main])
max_ind,max_val,scores=perform_gridsearch_withRFC(values,y)
print('all+main+vectors')
y,values,ids=get_values_for_crossvalidation(positives,negatives,[factive,implicative,report,hedges,assertive,main,vectors])
max_ind,max_val,scores=perform_gridsearch_withRFC(values,y)

max_ind,max_val,scores=perform_gridsearch_withRFC(values,y)



val_train, val_test, y_train, y_test, ids_train, ids_test = train_test_split(values, y, ids, test_size=TEST_SIZE, random_state=42)

clf,prob=train_test_final(val_train,val_test,y_train,50,200)
print_top_bottom_sentences(prob,ids_test,y_test,text,labels)

y_m,values_m,ids_m=get_values_for_crossvalidation(positives,negatives,[vectors,main])

max_ind,max_val,scores=perform_gridsearch_withRFC(values_m,y_m)

val_train, val_test, y_train, y_test, ids_train, ids_test = train_test_split(values, y, ids, test_size=TEST_SIZE, random_state=42)

clf,prob=train_test_final(val_train,val_test,y_train,100,200)
print_top_bottom_sentences(prob,ids_test,y_test,text,labels)




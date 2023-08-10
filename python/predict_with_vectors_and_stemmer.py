from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from random import shuffle
import numpy as np
import hashlib
from nltk.stem import SnowballStemmer

DEPTH_SEARCH=[5,10,30, 50, 100, 200]
NTREES_SEARCH=[5,10,30, 50, 100, 200]
TEST_SIZE=0.33

filenames={}
filenames_raw={}

"""
to use this, you will need: 
1) features from https://drive.google.com/open?id=1JZu67psmj2Eou2-8wQEJk4kAQfg8GDs2, to be placed in ../fastText_multilingual/features
"""
languages=['en','fr','it']
language_extended=['english','french','italian']
feadir='../fastText_multilingual/features/stemmed_'
rawdir='../data_clean/'

def load_languages():
    for lan,lext in zip(languages,language_extended):
        filenames[lan]=feadir+lan+'.tsv' #files with vectors
        filenames_raw[lan]=rawdir+lext+'.tsv' #files with raw text

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

"""
raw header is:
entity_id	revision_id	timestamp entity_title	section	start	offset	statement label
feature header is:
entity_id	revision_id	timestamp entity_title	section	start	offset	 label feature
"""
labels={} #whether it needs a citation or not
vectors={} #the word vectors aligned to english
main={} #is it the main section?
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
            #first append to lists of positives and negatives depending on the label
            if labels[unique]==1:
                positives[lan].append(unique)
            else:
                negatives[lan].append(unique)
            #store features
            vectors[unique]=[float(r) for r in row[-1].split(',')]
            main[unique]= 1 if row[4]=='MAIN_SECTION'else 0
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

print line
allneg=count_negatives(negatives,positives)

y,values,ids=get_values_for_crossvalidation(positives,negatives,[vectors])

max_ind,max_val,scores=perform_gridsearch_withRFC(values,y)

val_train, val_test, y_train, y_test, ids_train, ids_test = train_test_split(values, y, ids, test_size=TEST_SIZE, random_state=42)

clf,prob=train_test_final(val_train,val_test,y_train,30,200)
print_top_bottom_sentences(prob,ids_test,y_test,text,labels)

y_m,values_m,ids_m=get_values_for_crossvalidation(positives,negatives,[vectors,main])
print max(main.values())

print values_m[2]

max_ind,max_val,scores=perform_gridsearch_withRFC(values_m,y_m)



val_train, val_test, y_train, y_test, ids_train, ids_test = train_test_split(values, y, ids, test_size=TEST_SIZE, random_state=42)

clf,prob=train_test_final(val_train,val_test,y_train,100,200)
print_top_bottom_sentences(prob,ids_test,y_test,text,labels)




import numpy as np
import pandas as pd
import string
import os
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
pd.options.display.max_colwidth = 50
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

train = pd.read_csv('..//bases/training_variants')
test = pd.read_csv('..//bases/test_variants')

train_texts = pd.read_csv('..//bases/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")
test_texts = pd.read_csv('..//bases/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")

train = pd.merge(train, train_texts, how='left', on='ID')
test = pd.merge(test, test_texts, how='left', on='ID')

#Transform Gene Letter to their abbreviation in order to find them in the text
One_to_Three_AA = {'C': 'Cys', 'D': 'Asp', 'S': 'Ser', 'Q': 'Gln', 'K': 'Lys',
         'I': 'Ile', 'P': 'Pro', 'T': 'Thr', 'F': 'Phe', 'N': 'Asn', 
         'G': 'Gly', 'H': 'His', 'L': 'Leu', 'R': 'Arg', 'W': 'Trp', 
         'A': 'Ala', 'V': 'Val', 'E': 'Glu', 'Y': 'Tyr', 'M': 'Met'}
pattern = re.compile('|'.join(One_to_Three_AA.keys()))
##### Get variation types by using regex
def variation_regex(data, pattern): # if you want to not ignore cases, add extra argument to function
    Boolean = [not bool(re.search(pattern, i, re.IGNORECASE)) for i in data.Variation]
    data_no_regex = data[Boolean]  # 182 Fusions => 495 over 
    not_Boolean = [not i for i in Boolean]  
    data_regex = data[not_Boolean]
    
    return (data_regex, data_no_regex)

#### process the train and test set together
data_all = pd.concat((train, test), axis=0, ignore_index=True)
data_all_backup = data_all[:] ##### We keep backup because we want dummy variables of Gene & Text 
# TODO maybe also use Variation function of Gene from a database, and other suggestions. Also can use Count_sub as feature

def find_sub(data):

    ##### The normal case is around 2080 out of the 2644
    
    
    Boolean = [data.Variation[i] in data.Text[i] or #normal case
               data.Variation[i][:-1] in data.Text[i] or #case 1.
               pattern.sub(lambda x: One_to_Three_AA[x.group()], data.Variation[i][:-1]) # case2
               in data.Text[i]  for i in data.index] ## because new indexing we use 
    
    #TODO could also match insensitive as a next step for more info.
    #Shorter Boolean below = the normal version
    
    #Boolean = [trainSub.Variation[i] in trainSub.Text[i] #normal case
    #           for i in trainSub.ID] ## because new indexing we use ID
    #           
            
    sub_in_text = data[Boolean]
    not_Boolean = [not i for i in Boolean]  

    sub_not_in_text = data[not_Boolean]
#    sub_in_text['Count'] = [sub_in_text.Text[i].count(sub_in_text.Variation[i][:-1])
#                    +sub_in_text.Text[i].count(pattern.sub(lambda x: One_to_Three_AA[x.group()], sub_in_text.Variation[i][:-1]))
#                    for i in sub_in_text.index]
    
    return sub_in_text, sub_not_in_text
##### For subs that are not find in text: use regex to account for a different number
##### TODO: things you can further try - with AA name replacement, searching for the number only etc.
def find_sub_noText(data):
    Booleans = []
    for i in data.index:
        split_variation = re.split('(\d+)', data.Variation[i]) # split based on a number
        first_Amino = re.escape(split_variation[0]) #re.escpae uses variable as regex
        last_Amino = re.escape(split_variation[-1])
        #first_number = re.escape(split_variation[1][0])
        #new_regex = r"[^a-zA-Z0-9]" + first_Amino + first_number
        new_regex  = first_Amino + r"\d+" + last_Amino
        Boolean = bool(re.search(new_regex, data.Text[i]))
        Booleans.append(Boolean)
    
    sub_number_in_text = data[Booleans]
    not_Boolean = [not i for i in Booleans]  

    sub_again_no_text = data[not_Boolean]
    return sub_again_no_text, sub_number_in_text

##### Next we use a window to extract sentences
def get_sentences_sub(data, splitted_sentences, window_left, window_right):
    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
    data.index = range(len(data))
    sentences_with_sub = [[] for _ in range(len(data))]
    background=[[] for _ in range(len(data))]
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        one_to_three_variation = pattern.sub(lambda x: One_to_Three_AA[x.group()], data.Variation[i][:-1])
        Variation = data.Variation[i][:-1]        
        for j in range(len(sentences)):                              
            if (Variation in sentences[j]) or (one_to_three_variation in sentences[j]):
                new_regex = re.escape(Variation) + r"[\S]*" ### Means no white space 0 or more
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) #case 1
                new_regex = re.escape(one_to_three_variation) + r"[\S]*"
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) #case 2
                sentences_with_sub[i].extend(sentences[max(j-window_left,0) : min(j+1+window_right, len(sentences)-1)])
                
                ### We add space to ' placeholderMutation' because sometimes there are letters in front of it
                # position_sentences[i].append(j) # not used for the moment
            else:
                background[i].extend(sentences[max(j-window_left,0) : min(j+1+window_right, len(sentences)-1)])
                
    return sentences_with_sub,background   ### This might take a while because it's looping through all sentences
def get_sentences_sub_noText(data, splitted_sentences, window_left, window_right):
    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
    data.index = range(len(data))
    sentences_with_sub = [[] for _ in range(len(data))]
    background=[[] for _ in range(len(data))]
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i] 
        for j in range(len(sentences)):
            split_variation = re.split('(\d+)', data.Variation[i]) # split based on a number
            first_Amino = re.escape(split_variation[0]) #re.escpae uses variable as regex
            last_Amino = re.escape(split_variation[-1])
            new_regex  = first_Amino + r"\d+" + last_Amino
            
            #counter=len(re.findall(new_regex, sentences[j]))
            
            Boolean = bool(re.search(new_regex, sentences[j]))
            if Boolean:
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) # Might help catch sy
                sentences_with_sub[i].extend(sentences[max(j-window_left,0) : min(j+1+window_right,len(sentences)-1)])
                # position_sentences[i].append(j) # not used for the moment
            else:
                background[i].extend(sentences[max(j-window_left,0) : min(j+1+window_right, len(sentences)-1)])
    return sentences_with_sub,background   ### This might take a while because it's looping through all sentences

#### Converts list of sentences into one string of sentences for each document => to use for tfidf etc.
def sentences_to_string(sentences_list):
    sentence_strings = []
    for sentences in sentences_list:
        sentence_string =  ' '.join(str(sentence) for sentence in sentences)
        sentence_strings.append(sentence_string)
    
    return sentence_strings ### This doesn't take such a long time to run

######### First find those that have the format of being a substitution in data
data_all['Substitutions_var'] = data_all.Variation.apply(lambda x: bool(re.search('^[A-Z]\\d+[A-Z*]$', x))*1) #multiplying by 1 converts True to 1, False to 0 => Maybe modify this later?
data_all['Stop_codon_var'] = data_all.Variation.apply(lambda x: bool(re.search('[*]', x))*1) #multiplying by 1 converts True to 1, False to 0
data_sub = data_all[data_all['Substitutions_var']==1] ### Now we know the index of where a substitution occurs - the data_sub

sub_in_text, sub_not_in_text = find_sub(data_sub)
sub_in_text_backup = sub_in_text[:] ## index gets changed by text_processing if we don't make a copy
##### INVESTIGATION: Why do some subs don't appear in Text?: Try to automize this and find out
### Substitutions can appear as SAME_PREFIX - Other number - SAME_SUFFIX

sub_again_no_Text, sub_noText = find_sub_noText(sub_not_in_text) # 108 such cases out of 411 = nice improvement
sub_noText_backup = sub_noText[:]

#nltk.download("popular")

no_sub=data_all[data_all['Substitutions_var']==0]
sub_again_no_Text=sub_again_no_Text

def custom_find(data):
    l=["mutat","variat","fusion","deleti","amplific","framesh",
    "truncat","fs","exon","dupl"]
    Boolean={}
    for word in l:
        Boolean[word] = [re.search(word,data.Text[i]) for i in data.index] ## because new indexing we use 
    Bool=[any(tup) for tup in zip(Boolean["mutat"],Boolean["variat"],Boolean["fusion"],
        Boolean["deleti"],Boolean["amplific"],Boolean["truncat"],
        Boolean["fs"],Boolean["framesh"],Boolean["exon"],Boolean["dupl"])]
    no_sub_in_text=data[Bool]
    no_Bool=[not i for i in Bool]
    no_sub_not_in_text=data[no_Bool]
    return no_sub_in_text,no_sub_not_in_text

no_sub_yes,no_sub_no=custom_find(no_sub)

def custom_find_bis(data):
    l=["mutat","variat"]
    Boolean={}
    for word in l:
        Boolean[word] = [re.search(word,data.Text[i]) for i in data.index] ## because new indexing we use 
    Bool=[any(tup) for tup in zip(Boolean["mutat"],Boolean["variat"])]
    no_sub_in_text=data[Bool]
    no_Bool=[not i for i in Bool]
    no_sub_not_in_text=data[no_Bool]
    return no_sub_in_text,no_sub_not_in_text

sub_again_yes,sub_again_no=custom_find_bis(sub_again_no_Text)

def get_sentences_nosub(data, splitted_sentences, window_left, window_right):
    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
    data.index = range(len(data))
    sentences_with_sub = [[] for _ in range(len(data))]
    background=[[] for _ in range(len(data))]
    
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        l=["mutat","variat","fusion","deleti","amplific","framesh",
    "truncat","fs","exon","dupl"]     
        for j in range(len(sentences)):    
            for keyword in l:
                if keyword in sentences[j]:
                    sentences_with_sub[i].extend(sentences[max(j-window_left,0) : min(j+1+window_right, len(sentences)-1)])
                else:
                    background[i].extend(sentences[max(j-window_left,0) : min(j+1+window_right, len(sentences)-1)])
    return sentences_with_sub,background   ### This might take a while because it's looping through all sentences

NLTK_nosub = [sent_tokenize(no_sub_yes.Text[i]) for i in no_sub_yes.index]

nosub_sentences,nosub_background = get_sentences_nosub(no_sub_yes, NLTK_nosub, window_left = 0, window_right = 0) # Retrieves sentences where keyword is included
nosub_sentences = [sorted(set(sentences), key = sentences.index) for sentences in nosub_sentences] # only use unique 
nosub_background = [sorted(set(sentences), key = sentences.index) for sentences in nosub_background] # only use unique 


def get_sentences_sub_again_notext(data, splitted_sentences, window_left, window_right):
    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
    data.index = range(len(data))
    sentences_with_sub = [[] for _ in range(len(data))]
    background=[[] for _ in range(len(data))]
    
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        l=["mutat","variat"]    
        for j in range(len(sentences)):    
            for keyword in l:
                if keyword in sentences[j]:
                    sentences_with_sub[i].extend(sentences[max(j-window_left,0) : min(j+1+window_right, len(sentences)-1)])
                else:    
                    background[i].extend(sentences[max(j-window_left,0) : min(j+1+window_right, len(sentences)-1)])
    return sentences_with_sub,background   ### This might take a while because it's looping through all sentences

NLTK_sub_again_notext = [sent_tokenize(sub_again_yes.Text[i]) for i in sub_again_yes.index]

sub_again_sentences,sub_again_background = get_sentences_sub_again_notext(sub_again_yes, NLTK_sub_again_notext, window_left = 0, window_right = 0) # Retrieves sentences where keyword is included
sub_again_sentences = [sorted(set(sentences), key = sentences.index) for sentences in sub_again_sentences] # only use unique 
sub_again_background = [sorted(set(sentences), key = sentences.index) for sentences in sub_again_background] # only use unique 

NLTK_sub_noText = [sent_tokenize(sub_noText.Text[i]) for i in sub_noText.index]
sub_noText_sentences,sub_noText_background = get_sentences_sub_noText(sub_noText, NLTK_sub_noText, window_left = 0, window_right = 0) # Retrieves sentences where subsitution mutation is included
sub_noText_sentences = [sorted(set(sentences), key = sentences.index) for sentences in sub_noText_sentences] # only use unique sentences
sub_noText_background = [sorted(set(sentences), key = sentences.index) for sentences in sub_noText_background] # only use unique 

NLTK_sub = [sent_tokenize(sub_in_text.Text[i]) for i in sub_in_text.index] # takes a long time to run tokenizer => use pickle to save
sub_sentences,sub_background = get_sentences_sub(sub_in_text, NLTK_sub, window_left = 0, window_right = 0) 
# Retrieves sentences where subsitution mutation is included.
# window_left and window_right specify which sentences to keep at the left side or right side of the sub sentences.
# IMPORTANT: I used also placeholderMutation to replace the original sub mutations here
sub_sentences = [sorted(set(sentences), key = sentences.index) for sentences in sub_sentences]
sub_background = [sorted(set(sentences), key = sentences.index) for sentences in sub_background]

sub_sentences_string = sentences_to_string(sub_sentences)
sub_noText_string = sentences_to_string(sub_noText_sentences)
nosub_sentences_string=sentences_to_string(nosub_sentences)
sub_again_string=sentences_to_string(sub_again_sentences)

#Creating the background text (Text is now the frontline text)

#data_all.loc[sub_in_text.index,"Background"]=sentences_to_string(sub_background)
#data_all.loc[sub_noText.index,"Background"]=sentences_to_string(sub_noText_background)
#data_all.loc[no_sub_yes.index,"Background"]=sentences_to_string(nosub_background)
#data_all.loc[sub_again_yes.index,"Background"]=sentences_to_string(sub_again_background)
#data_all.loc[sub_again_no.index,"Background"]="nbckgr"
#data_all.loc[no_sub_no.index,"Background"]="nbckgr"

#data_all.Background[data_all.Background==""]="nbckgr"
#data_all.Background[data_all.Background.isnull()]="nbckgr"

data_all.Text[sub_in_text.index] = sub_sentences_string
data_all.Text[sub_noText.index] = sub_noText_string
data_all.Text[no_sub_yes.index] = nosub_sentences_string
data_all.Text[sub_again_yes.index] = sub_again_string
data_all.Text[sub_again_no.index]="null"
data_all.Text[no_sub_no.index]="null"
data_all.Text[data_all.Text==""]="null"
data_all.Text[data_all.Text.isnull()]="null"

##############################################################################################################################
############## Non-subs preprocessing of data set #######################

#def find_mutation_type(row, pattern):  ##### TODO: make clearer by using a function instead of lambda
#    return bool(re.search('^fusion', row, re.IGNORECASE)) *1. Also for subs

####### Fusions : 'Fusions' ############
data_all['Fusion_var'] = data_all.Variation.apply(lambda x: bool(re.search('^fusion', x, re.IGNORECASE))*1) #multiplying by 1 converts True to 1, False to 0
new_fusion, new_data_all = variation_regex(data_all, '^fusion') 

###### Fusions: 'Gene-Gene fusion' ########
data_all['gene_fusion_var'] = data_all.Variation.apply(lambda x: bool(re.search('fusion', x, re.IGNORECASE))*1) 
_ , new_data_all = variation_regex(new_data_all, 'fusion') 
###### Notice that NaN introduced for places where splicing occured => replace after NaN with 0's when complete

####### Deletions: 'Deletions' ############
data_all['Deletion_var'] = data_all.Variation.apply(lambda x: bool(re.search('^del', x, re.IGNORECASE))*1) 
new_del, new_data_all = variation_regex(new_data_all, '^del') 

####### Deletions & Insertions wheteher together or seperately (doesn't make a big difference IMO)
data_all['del_or_ins_var'] = data_all.Variation.apply(lambda x: bool(re.search('del|ins', x, re.IGNORECASE))*1) 
# we could also later divide it into del, ins if we want to

###### Amplifications #########
data_all['Amplification_var'] = data_all.Variation.apply(lambda x: bool(re.search('ampl', x, re.IGNORECASE))*1) 

###### Truncations ########### Don't forget there are 'Truncating mutations' = 95 and '_trunc' = 4
data_all['Truncation_var'] = data_all.Variation.apply(lambda x: bool(re.search('trunc', x, re.IGNORECASE))*1) 

####### Exons #########
data_all['exon_var'] = data_all.Variation.apply(lambda x: bool(re.search('exon', x, re.IGNORECASE))*1) 

####### Frameshift mutations ########
data_all['frameshift_var'] = data_all.Variation.apply(lambda x: bool(re.search('fs', x, re.IGNORECASE))*1) 

####### Duplications ##############
data_all['dup_var'] = data_all.Variation.apply(lambda x: bool(re.search('dup', x, re.IGNORECASE))*1) 




#TODO : #Sentence Tokenizer for non-subs that are not in text, AND
#subs that are not in text

#Not used
################ The dummy variables for Gene and Text ##################
## TODO: also use dummy for Text? There are 135 shared Genes and 142 shared Text between train and Leaks!  len(set(train.Text) & set(Leaks.Text))
#data_all_dummy = data_all_backup[['Gene', 'Text']] # drop those columns we don't need as dummy.
#X_dummy = pd.get_dummies(data_all_dummy) # converts categorical variables into dummy variable. From len set => 269 genes + 2090 texts
#X_dummy_train = X_dummy[:train.shape[0]]
#X_dummy_test = X_dummy[train.shape[0]:]
#dummy_names = X_dummy.columns.values #### To remember names if you want to check again what Gene or Text used
#X_dummy = X_dummy.values

###### Use the variation types 
#variation_types = data_all.drop(['ID', 'Gene', 'Class', 'Text', 'Variation'], axis =1)
#X_variation_train = variation_types[:train.shape[0]]
#X_variation_test = variation_types[train.shape[0]:]
#variation_names = variation_types.columns.values 

stop = set(stopwords.words('english'))
exclude = set('!"#$%&\'()*+:;<=>?@[\\]^_`{|}~0123456789') 
lemma = WordNetLemmatizer()
def clean(doc,lemmatiz=False):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free_0 = [re.sub(",|\.|/"," ",ch) for ch in stop_free]
    if lemmatiz:
        punc_free_lem="".join(ch for ch in punc_free_0 if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free_lem.split())
        return normalized
    else:
        punc_free = "".join(ch for ch in punc_free_0 if ch not in exclude)
        return punc_free

#No lemmatization for the moment, be careful not to lemmatize then w2vec
data_all.Text = [clean(doc) for doc in data_all.Text]  
#data_all.Background = [clean(doc) for doc in data_all.Background]

ID_train=train.ID
ID_test=test.ID

train = data_all.iloc[:len(train)]
test = data_all.iloc[len(train):]

#tfidf = TfidfVectorizer(
 #       min_df=3, max_features=5000, strip_accents=None, lowercase = False,
  #      analyzer='word', token_pattern=r'\w+', ngram_range=(1,3), use_idf=True,
   #     smooth_idf=True, sublinear_tf=True
    #    ).fit(train["Background"])

#X_train_background = tfidf.transform(train["Background"])
#X_test_background = tfidf.transform(test["Background"])

#tsvd=TruncatedSVD(n_components=100,n_iter=10,random_state=26)
#tsvd_train=tsvd.fit_transform(X_train_background)
#tsvd_test=tsvd.transform(X_test_background)
    

#for i in range(100):
#   train['tsvd_'+str(i)] = tsvd_train[:, i]
#   test['tsvd_'+str(i)] = tsvd_test[:, i]
    

data_all=pd.concat((train, test), axis=0, ignore_index=True)

#Some counters on the gene and variation, here not using the 3letters abreviations
#that means if the text contains only the gene or variation but with the 
#3letters format, the variable will capture it
data_all["Gene_Share"] = data_all.apply(lambda r: sum([1 for w in r["Gene"].split(" ") if w in r["Text"].split(" ")]), axis=1)
data_all["Variation_Share"] = data_all.apply(lambda r: sum([1 for w in r["Variation"].split(" ") if w in r["Text"].split(" ")]), axis=1)

data_all["Text_words"] = data_all["Text"].map(lambda x: len(str(x).split(" ")))

figure_counter=["fig","figure"]
for fig in figure_counter: 
        data_all["Figure_counter"]=(data_all["Text"].map(lambda x : str(x).count(fig)))

data_all["Text"]=[re.sub("fig|figure"," ",doc) for doc in data_all["Text"]]

train_1 = data_all.iloc[:len(train)]
test_1 = data_all.iloc[len(train):]

train_2=pd.get_dummies(train_1,columns=["Gene"])

test_2=pd.get_dummies(test_1,columns=["Gene"])

#add and remove gene dummies on test set to match train

def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0
def fix_test_columns( d, columns ):  

    add_missing_dummy_columns( d, columns )

    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print ("extra columns:", extra_cols)

    d = d[ columns ]
    return d

test_3=fix_test_columns(test_2,train_2.columns)

train_2.to_csv("checkpoints_databases/w_working_train.csv",index=False,encoding="utf8")
test_3.to_csv("checkpoints_databases/w_working_test.csv",index=False,encoding="utf8")




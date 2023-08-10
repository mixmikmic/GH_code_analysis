import numpy as np
import pandas as pd
import string
import os
from collections import Counter
import copy

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
pd.options.display.max_colwidth = 50
pd.set_option('display.max_rows', 500)
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

new_test=pd.read_csv('..//bases/new_test_variants.csv')
new_test_texts = pd.read_csv('..//bases/new_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")
new_test_final=pd.merge(new_test,new_test_texts,how="left",on="ID")

leaks=pd.read_csv('..//bases/s1_add_train.csv')
leaks_1=pd.DataFrame([leaks["ID"],leaks.drop("ID",axis=1).idxmax(axis=1).map(lambda x: x.lstrip('class'))])
leaks_2=leaks_1.T
leaks_2.columns=["ID","Class"]

train = pd.read_csv('..//bases/training_variants')
test = pd.read_csv('..//bases/test_variants')

train_texts = pd.read_csv('..//bases/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")
test_texts = pd.read_csv('..//bases/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")

train = pd.merge(train, train_texts, how='left', on='ID')
test = pd.merge(test, test_texts, how='left', on='ID')

leaks_3=pd.merge(leaks_2,test[test.ID.isin(leaks_2.ID)])
leaks_final=pd.merge(leaks_3,test_texts[test_texts.ID.isin(leaks_3.ID)])

train_all = pd.concat([train,leaks_final]) #adding first stage

merge_match = new_test.merge(train_all, left_on=['Gene', 'Variation'], right_on = ['Gene', 'Variation'])
Index_leak = merge_match.ID_x - 1
new_test_index = [item for item in new_test_final.index if item not in list(Index_leak)]
test_no_leaks = new_test_final.iloc[new_test_index]
test_no_leaks

train_all['Substitutions_var'] = train_all.Variation.apply(lambda x: bool(re.search('^[A-Z]\\d+[A-Z*]$', x))*1)
new_train = train_all[train_all['Substitutions_var']==1]

#### process the train and test set together
data_all = pd.concat((new_train, test_no_leaks), axis=0, ignore_index=True)
data_all = data_all[['Class', 'Gene', 'ID', 'Variation', 'Text']] # just reordering
data_all_backup = data_all[:] ## We keep backup in case we need to use again
data_all

data_all.iloc[3107,:]

#Transform Amino Acid (AA) Letter to their three-letter abbreviation in order to find them in the text when they appear
One_to_Three_AA = {'C': 'Cys', 'D': 'Asp', 'S': 'Ser', 'Q': 'Gln', 'K': 'Lys',
         'I': 'Ile', 'P': 'Pro', 'T': 'Thr', 'F': 'Phe', 'N': 'Asn', 
         'G': 'Gly', 'H': 'His', 'L': 'Leu', 'R': 'Arg', 'W': 'Trp', 
         'A': 'Ala', 'V': 'Val', 'E': 'Glu', 'Y': 'Tyr', 'M': 'Met'}
pattern = re.compile('|'.join(One_to_Three_AA.keys()))

# find_sub return the substituions that are in text and those that are not
def find_sub(data):    
    Boolean = [data.Variation[i][:-1] in data.Text[i] or #case 1.
               pattern.sub(lambda x: One_to_Three_AA[x.group()], data.Variation[i][:-1]) # case2
               in data.Text[i]  for i in data.index] ## because new indexing we use 
    
    sub_in_text = data[Boolean]
    not_Boolean = [not i for i in Boolean]  
    sub_not_in_text = data[not_Boolean]
    
    return sub_in_text, sub_not_in_text

# find_sub_numberChange searches for other number of a substitution i.e. G12V -> G_V because sometimes mistake in entry
# Is currently without One_to_three substitution or Variation[:-1] only the full variation
def find_sub_numberChange(data):
    Booleans = [] #will contain the different Booleans if found in text
    for i in data.index:
        split_variation = re.split('(\d+)', data.Variation[i]) # split based on a number
        first_Amino = re.escape(split_variation[0]) #re.escpae uses variable as regex
        last_Amino = re.escape(split_variation[-1])
        new_regex  = first_Amino + r"\d+" + last_Amino
        Boolean = bool(re.search(new_regex, data.Text[i]))
        Booleans.append(Boolean)
        
    sub_number_in_text = data[Booleans]
    not_Boolean = [not i for i in Booleans]  
    sub_number_no_text = data[not_Boolean]
    
    return sub_number_in_text, sub_number_no_text

# for substitutions that are still not found, use other keywords
def find_sub_pattern(data):    
    Boolean = [('mutat' in data.Text[i]) or ('variant' in data.Text[i]) or (data.Gene[i] in data.Text[i]) for i in data.index] ## because new indexing we use 
    
    sub_in_text = data[Boolean]
    not_Boolean = [not i for i in Boolean]  
    sub_not_in_text = data[not_Boolean]
    
    return sub_in_text, sub_not_in_text

##### get_sentences_sub use a window to extract sentences where the subs appear. 
# If window_left & window_right = 0 => just taking the sentences with subs

def get_sentences_sub(data, splitted_sentences):
    data.index = range(len(data)) #makes sure that index is right
    sentences_with_sub = [[] for _ in range(len(data))]
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        one_to_three_variation = pattern.sub(lambda x: One_to_Three_AA[x.group()], data.Variation[i][:-1])
        Variation = data.Variation[i][:-1]    
        for j in range(len(sentences)):                              
            if (Variation in sentences[j]) or (one_to_three_variation in sentences[j]):
                new_regex = re.escape(Variation) + r"[\S]*" ###  r"[\S]*" because we look for Variation[:-1] not just Variation
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) #case 1
                ### We add the space to ' placeholderMutation' because sometimes there are letters in front of it
                new_regex = re.escape(one_to_three_variation) + r"[\S]*"
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) #case 2
                sentences_with_sub[i].extend(sentences[j:j+1])
                
    return sentences_with_sub

##### get_sentences_sub_number use a window to extract sentences where the subs appear that have different number i.e. G12V -> G_V

def get_sentences_sub_number(data, splitted_sentences):
    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
    data.index = range(len(data))
    sentences_with_sub_number = [[] for _ in range(len(data))]
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        split_variation = re.split('(\d+)', data.Variation[i]) # split based on a number
        first_Amino = re.escape(split_variation[0]) #re.escpae uses variable as regex
        last_Amino = re.escape(split_variation[-1])
        new_regex  = first_Amino + r"\d+" + last_Amino
        
        for j in range(len(sentences)):
            Boolean = bool(re.search(new_regex, sentences[j]))            
            if Boolean:
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) # Again replacing the sentence with placeholder
                sentences_with_sub_number[i].extend(sentences[j:j+1])
    
    return sentences_with_sub_number

# for substitutions that are still not found, use other keywords
def get_sentences_pattern(data, splitted_sentences):
    data.index = range(len(data)) #makes sure that index is right
    sentences_with_sub = [[] for _ in range(len(data))]
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        gene_name = data.Gene[i]
        for j in range(len(sentences)):                              
            if ('mutat' in sentences[j]) or ('variant' in sentences[j]) or (gene_name in sentences[j]):
                sentences[j] = re.sub(gene_name, ' placeholderGene', sentences[j]) # This time we replace for the gene because specific mutation not found
                sentences_with_sub[i].extend(sentences[j:j+1])               
    
    return sentences_with_sub

###### We use a window when sentences are too low. For 5 <= LENGTH <= 10: window of 1, for <= 5: window of 2 =, for 1 : window of 3
###### 

def get_window_sub(data, splitted_sentences, lengths):
    data.index = range(len(data)) #makes sure that index is right
    sentences_with_sub = [[] for _ in range(len(data))]
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        length = lengths[i]
        if length == 1:
            window = 6
        elif length == 2:
            window = 3
        elif length >= 5:
            window=1
        else:
            window=2
        
        one_to_three_variation = pattern.sub(lambda x: One_to_Three_AA[x.group()], data.Variation[i][:-1])
        Variation = data.Variation[i][:-1] 
        all_sentences = []
        for j in range(len(sentences)):
            if (Variation in sentences[j]) or (one_to_three_variation in sentences[j]):
                new_regex = re.escape(Variation) + r"[\S]*" ###  r"[\S]*" because we look for Variation[:-1] not just Variation
                other_regex = re.escape(one_to_three_variation) + r"[\S]*"
                for sentence in sentences[max(j-window,0) : min(j+1+window, len(sentences)-1)]: # to account if start or end of text
                    sentence = re.sub(new_regex, ' placeholderMutation', sentence) #case 1
                    sentence = re.sub(other_regex, ' placeholderMutation', sentence) #case 2 after case 1
                    all_sentences.append(sentence)
        sentences_with_sub[i] = all_sentences
                    
    return sentences_with_sub

def get_window_sub_number(data, splitted_sentences, lengths):
    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
    data.index = range(len(data))
    sentences_with_sub_number = [[] for _ in range(len(data))]
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i] 
        length = lengths[i]
        if length == 1:
            window = 6
        elif length == 2:
            window = 3
        elif length >= 5:
            window=1
        else:
            window=2
            
        split_variation = re.split('(\d+)', data.Variation[i]) # split based on a number
        first_Amino = re.escape(split_variation[0]) #re.escpae uses variable as regex
        last_Amino = re.escape(split_variation[-1])
        new_regex  = first_Amino + r"\d+" + last_Amino    
        
        all_sentences = []
        for j in range(len(sentences)):
            Boolean = bool(re.search(new_regex, sentences[j]))            
            if Boolean:
                for sentence in sentences[max(j-window,0) : min(j+1+window, len(sentences)-1)]:
                    sentence = re.sub(new_regex, ' placeholderMutation', sentence)
                    all_sentences.append(sentence)
        sentences_with_sub_number[i] = all_sentences
    
    return sentences_with_sub_number

def get_window_pattern(data, splitted_sentences, lengths):
    data.index = range(len(data)) #makes sure that index is right
    sentences_with_sub = [[] for _ in range(len(data))]
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        length = lengths[i]
        if length == 1:
            window = 6
        elif length == 2:
            window = 3
        elif length >= 5:
            window=1
        else:
            window=2
        
        gene_name = data.Gene[i]
        all_sentences = []
        for j in range(len(sentences)):
            if ('mutat' in sentences[j]) or ('variant' in sentences[j]) or (gene_name in sentences[j]):
                for sentence in sentences[max(j-window,0) : min(j+1+window, len(sentences)-1)]: # to account if start or end of text
                    sentence = re.sub(gene_name, ' placeholderGene', sentence) #case 1
                    all_sentences.append(sentence)
        sentences_with_sub[i] = all_sentences
                    
    return sentences_with_sub



#### Converts list of sentences into one string of sentences for each document => to use for tfidf etc.
def sentences_to_string(sentences_list):
    sentence_strings = []
    for sentences in sentences_list:
        sentence_string =  ' '.join(str(sentence) for sentence in sentences)
        sentence_strings.append(sentence_string)
    
    return sentence_strings 

######### First find those that have the format of being a substitution in data
data_all['Substitutions_var'] = data_all.Variation.apply(lambda x: bool(re.search('^[A-Z]\\d+[A-Z*]$', x))*1) #multiplying by 1 converts True to 1, False to 0 => Maybe modify this later?
data_all['null'] = data_all.Variation.apply(lambda x: bool(re.search('null', x))*1)
data_all = data_all[(data_all['Substitutions_var']==1) | (data_all['null']==1) ] ### Now we know the index of where a substitution occurs - the data_sub
data_all = data_all.loc[:, 'Class':'Text']
data_sub = data_all
print("Length of total subs: %i" %len(data_sub)) # other ways to process it like finding the word 'mutation' 
data_all

## First consider the subs that appear in text
sub_text, sub_no_text = find_sub(data_sub) 

## use tokenizer to split into sentences of all the subs in text 
NLTK_sub = [sent_tokenize(sub_text.Text[i]) for i in sub_text.index] 

NLTK_window = copy.deepcopy(NLTK_sub) # a deep copy is necessary besauce something magical and strange happening without it
NLTK_copy = copy.deepcopy(NLTK_sub) # again for backup purposes

# extract window for the sub sentences where they appear
# !! Use [:] because it makes a copy and doesn't change anything to the original indexes
sub_sentences = get_sentences_sub(sub_text[:], NLTK_copy[:]) # choosing for window 0 as default now
sub_sentences_string = sentences_to_string(sub_sentences)

# Replace text in data_all
data_all.Text.loc[sub_text.index] = sub_sentences_string
data_all.Text.loc[sub_text.index]

######### We rerun for window. Length<2 : w = 3, Length >=6: w = 1, Length in between: w = 2
indexes = [index for index, sentences in enumerate(sub_sentences) if len(sentences) <= 10]
sentence_lengths = [len(sentences) for index, sentences in enumerate(sub_sentences) if len(sentences) <= 10]
NLTK_sub_window = [NLTK_window[i] for i in indexes]

new_index = sub_text.index[indexes]
sub_window = sub_text.loc[new_index] # gets the subs with the low length
sub_window #1524 cases 

sub_window_sentences = get_window_sub(sub_window[:], NLTK_sub_window[:], sentence_lengths) 
sub_window_sentences = [sorted(set(sentences), key = sentences.index) for sentences in sub_window_sentences] # removes duplicates
sub_window_string = sentences_to_string(sub_window_sentences)

sub_window_sentences[3] # only thing we can do about sentences like this one is look at the full text words now

# Finally: replacing those texts in data_all
data_all.Text.loc[new_index] = sub_window_string
data_all.Text.loc[new_index]

# now the subs that don't appear in text: one reason is different number in the substitution. f.e. G12V -> G13V 
sub_number_text, sub_number_no_text = find_sub_numberChange(sub_no_text) # 131 cases with number change

## use tokenizer to split into sentences of subs that have different number in text 
NLTK_sub_number = [sent_tokenize(sub_number_text.Text[i]) for i in sub_number_text.index]
NLTK_sub_number_window = copy.deepcopy(NLTK_sub_number) # a deep copy is necessary besauce something magical and strange happening without it
NLTK_sub_number_copy = copy.deepcopy(NLTK_sub_number) # again for backup purposes

# extract window for the sub sentences where they appear
# !! Use [:] because it makes a copy and doesn't change anything to the original indexes
sub_number_sentences = get_sentences_sub_number(sub_number_text[:], NLTK_sub_number_copy[:])
sub_number_string = sentences_to_string(sub_number_sentences)

data_all.Text.loc[sub_number_text.index] = sub_number_string #iloc for indexing based on integers
data_all.Text.loc[sub_number_text.index]

######### ######### We rerun for window. Length<2 : w = 3, Length >=6: w = 1, Length in between: w = 2
indexes = [index for index, sentences in enumerate(sub_number_sentences) if len(sentences) <= 10]
sentence_lengths = [len(sentences) for index, sentences in enumerate(sub_number_sentences) if len(sentences) <= 10]
NLTK_window = [NLTK_sub_number[i] for i in indexes]

new_index = sub_number_text.index[indexes]
sub_number_window = sub_number_text.loc[new_index] # gets the subs with the low length
sub_number_window #87 cases 

sub_window_sentences = get_window_sub_number(sub_number_window[:], NLTK_window[:], sentence_lengths) 
sub_window_sentences = [sorted(set(sentences), key = sentences.index) for sentences in sub_window_sentences] # removes duplicates
sub_window_string = sentences_to_string(sub_window_sentences)
sub_window_sentences[24]

data_all.Text.loc[new_index] = sub_window_string
data_all.Text.loc[new_index]

data_all.loc[len(new_train):].Text[data_all.loc[len(new_train):].index[4]]

sub_number_no_text.shape # 365 left

sub_number_no_text.loc[len(new_train):].shape # 62 in test

texts_to_analyze = list(sub_number_no_text.loc[len(new_train):].Text)

len(set(texts_to_analyze)) # basically 31 texts the same!

texts_to_analyze[8]

sub_number_no_text.loc[len(new_train):] # A lot of sentences starting with 'Among...'

# subs that are still not found, look based on a pattern like the word 'mutat', 'variat' or the gene in text
sub_pattern_text, sub_pattern_no_text = find_sub_pattern(sub_number_no_text) # 

sub_pattern_no_text # only 13 texts that still don't include the sentences of which three nulls and low text_words length, replace with nulls

data_all.Text.loc[sub_pattern_no_text.index] = ''
data_all.Text.loc[sub_pattern_no_text.index]

## use tokenizer to split into sentences of subs that have keywords or gene
NLTK_sub_pattern = [sent_tokenize(sub_pattern_text.Text[i]) for i in sub_pattern_text.index]

# !! Use [:] because it makes a copy and doesn't change anything to the original indexes
sub_pattern_sentences = get_sentences_pattern(sub_pattern_text[:], NLTK_sub_pattern[:])
sub_pattern_string = sentences_to_string(sub_pattern_sentences)

data_all.Text.iloc[sub_pattern_text.index] = sub_pattern_string #iloc for indexing based on integers
data_all.Text.iloc[sub_pattern_text.index]

######### ######### We rerun for window. Length<2 : w = 3, Length >=6: w = 1, Length in between: w = 2
indexes = [index for index, sentences in enumerate(sub_pattern_sentences) if len(sentences) <= 10]
sentence_lengths = [len(sentences) for index, sentences in enumerate(sub_pattern_sentences) if len(sentences) <=10]
NLTK_window = [NLTK_sub_pattern[i] for i in indexes]

sentence_lengths # 19 cases where we still look again, some of them from same text

new_index = sub_pattern_text.index[indexes]
sub_pattern_window = sub_pattern_text.loc[new_index] # gets the subs with the low length
sub_pattern_window #87 cases 

sub_window_sentences = get_window_pattern(sub_pattern_window[:], NLTK_window[:], sentence_lengths) 
sub_window_sentences = [sorted(set(sentences), key = sentences.index) for sentences in sub_window_sentences] # removes duplicates
sub_window_string = sentences_to_string(sub_window_sentences)

data_all.Text.loc[new_index] = sub_window_string
data_all.Text.loc[new_index]

test_all = data_all.loc[len(new_train):]
all_text = list(test_all.Text)
test_all

positions_machine = []
all_together = []
for i in range(len(all_text)):
    text = all_text[i]
    if text.count('placeholderMutation') < 11:
        all_together.append((i, text.count('placeholderMutation')))
        positions_machine.append(i)

important_texts = [all_text[i] for i in positions_machine]
test_all.iloc[positions_machine]

all_together

test_important = test_all.iloc[positions_machine]
test_important

test_important.Text[test_important.index[20]]



shared_genes = list(set(train_all.Gene).intersection(set(test_all.Gene))) # the shared genes are those that appear in both
count_important = 0
for gene in test_important.Gene:
    if gene in shared_genes:
        count_important += 1

count_important # so 104 EXAMPLES in test where same gene appears as in the impo

len(shared_genes)

shared_genes

for gene in shared_genes:
    data_all[gene] = 0
data_all

for i in data_all.index:
    gene = data_all.Gene[i]
    if gene in shared_genes:
        data_all.loc[i, gene] = 1

data_all # it's not all zeros, it looks like it but genes like BRAF and EFGR have more 1's because they occur more

# We decided to go for 15 dummies for genes:
N_components = 15
svd = TruncatedSVD(n_components= N_components, n_iter=18, random_state=18)
data_text = data_all[['Class', 'Gene', 'ID', 'Variation','Text']] #5th is Text, just before the genes 
one_hot_gene = data_all[shared_genes]

truncated_one_hot_gene = svd.fit_transform(one_hot_gene.values)
print(svd.explained_variance_ratio_.sum())

genes_df =pd.DataFrame(truncated_one_hot_gene,columns=["tsvd_gene"+ str(x) for x in range(N_components)])
genes_df

data_final = pd.merge(data_text.reset_index(), genes_df.reset_index()).drop("index",axis=1)

data_final

# Replace column name for text of window, and add new column of full text
data_final.insert(loc=4, column='Full_Text', value=data_all_backup.Text)
data_final.columns.values[5] = 'Window_Text'
data_final

new_train_1 = data_final.iloc[:len(new_train)]
new_test_1 = data_final.iloc[len(new_train):]

new_test_1

new_train_1.shape

new_test_1.shape

new_train_1.to_csv("checkpoints_databases/new_working_train.csv",index=False,encoding="utf8")
new_test_1.to_csv("checkpoints_databases/new_working_test.csv",index=False,encoding="utf8")






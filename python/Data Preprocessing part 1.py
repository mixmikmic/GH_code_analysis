from copy import deepcopy
import re
import numpy as np
import pandas as pd
import matplotlib as plt
import pickle

colnames = ['sentence_id', 'text', 'pair_id', 'drug1_id', 'drug1', 'drug1_type', 'drug2_id', 'drug2', 'drug2_type', 'ddi', 'ddi_type']
train_data = pd.read_csv('./ddi_train.csv', header = None, names = colnames)
test_data = pd.read_csv('./ddi_test.csv', header = None, names = colnames)

train_data['drug1'] = train_data['drug1'].apply(lambda x: x.lower())
train_data['drug2'] = train_data['drug2'].apply(lambda x: x.lower())
test_data['drug1'] = test_data['drug1'].apply(lambda x: x.lower())
test_data['drug2'] = test_data['drug2'].apply(lambda x: x.lower())

#mini set to test functions with
mini_set = deepcopy(train_data.loc[1535:1545])

def anonymize_drugs(data):
    '''replaces the drug mentions in the sentences with drug1, drug2, or drug0. The drug pair of interest is
    replaced with drug1 and drug2 while other drug mentions in the sentence that are not part of the pair
    are replaced with drug0
    
    Example:
    
    laboratory tests response to plenaxis should be monitored by measuring serum total testosterone 
    concentrations just prior to administration on day 29 and every 8 weeks thereafter.
    
    If the pair of interest is plenaxis and testosterone, this sentence becomes:
    
    laboratory tests response to drug1 should be monitored by measuring serum total drug2 
    concentrations just prior to administration on day 29 and every 8 weeks thereafter.
    '''
    sentences = data['text']
    drug1_list = data['drug1']
    drug2_list = data['drug2']
    drug_list = np.unique(np.concatenate([drug1_list, drug2_list]))
    anonymized_text = []
    if 'drug' in drug_list:
        drug_list = np.delete(drug_list, np.where(drug_list == 'drug')[0][0])
    for i in range(len(sentences)):
        sentence = sentences.iloc[i]
        #print(f'{drug1_list.iloc[i]}')
        #print(f'{drug2_list.iloc[i]}')
        #print(sentence)
        drug1 = drug1_list.iloc[i]
        drug2 = drug2_list.iloc[i]
        if sentence.find(drug1) != -1:
            try:
                regex_drug1 = re.compile(f'{drug1_list.iloc[i]}[^a-zA-Z0-9]')

                for m in re.finditer(regex_drug1, sentence):
                    last_char_drug1 = sentence[m.end() - 1]
                    if last_char_drug1 != ' ':
                        sentence = regex_drug1.sub(f'drug1{last_char_drug1}', sentence, count = 1)
                        break
                    sentence = regex_drug1.sub(f'drug1 ', sentence, count = 1)
                    break
            except: #this is to avoid 'nothing to repeat' errors that occassionally occur for some reason when compiling an re
                sentence = sentence.replace(drug1, 'drug1', 1) 
        
        if sentence.find(drug2) != -1:
            try:
                regex_drug2 = re.compile(f'{drug2_list.iloc[i]}[^a-zA-Z0-9]')
                for m in re.finditer(regex_drug2, sentence):
                    last_char_drug2 = sentence[m.end() - 1]
                    if last_char_drug2 != ' ':
                        sentence = regex_drug2.sub(f'drug2{last_char_drug2}', sentence, count = 1)
                        break
                    sentence = regex_drug2.sub(f'drug2 ', sentence, count = 1)
                    break
            except: #this is to avoid 'nothing to repeat' errors that ocassionally occur for some reason when compiling an re
                sentence = sentence.replace(drug2, 'drug2', 1)
        
        for drug in drug_list:
            if sentence.find(drug) != -1:
                try:
                    regex_drug0 = re.compile(f'{drug}\W')
                    last_chars = []
                    for m in re.finditer(regex_drug0, sentence):
                        last_chars.append(sentence[m.end() - 1])
                    for chars in last_chars:
                        sentence = regex_drug0.sub(f'drug0{chars}', sentence, count = 1)
                except: #this is to avoid 'nothing to repeat' errors that ocassionally occur for some reason when compiling an re
                    sentence = sentence.replace(drug, 'drug0')
        
        anonymized_text.append(sentence)
    
    data['anonymized_text'] = anonymized_text
            

def identify_same_drug(row):
    '''Returns 1 if the drugs in a given pair are the same, 0 otherwise'''
    if row['drug1'].strip().lower() == row['drug2'].strip().lower():
        return 1
    return 0

def identify_negative_phrases(row):
    '''Returns 1 if there is the presence of a negation word or phrase such as 'no', "n't", or 'not'.'''
    negative_regex = re.compile("no[^a-zA-Z0-9]|not[^a-zA-Z0-9]|.*n't[^a-zA-Z0-9]")
    match = re.search(negative_regex, row['text'].lower())
    if match:
        return 1
    return 0

def identify_series(row):
    '''Returns 1 if 'drug1' and 'drug2' appear in the same coordinate phrase, 0 otherwise'''
    patterns = [re.compile("drug1[,;](|\s)([a-zA-Z0-9]*,(|\s))+(|or\s|and\s)drug2"),
                re.compile("drug1[,;](|\s)drug2"),
                re.compile("drug2[,;](|\s)([a-zA-Z0-9]*,(|\s))+(|or\s|and\s)drug1"),
                re.compile("drug2[,;](|\s)drug1")]
    
    for re_pattern in patterns:
        if re.search(re_pattern, row['anonymized_text']):
            return 1
    return 0

def identify_special_cases(row):
    '''Returns 1 if drug1 is a special case of drug2 or vice versa. An example of this is when a drug is
    describing a class of drugs. For example, the phrase 'drug1 such as drug2' should return 1.'''
    patterns = [re.compile('drug1(|\s)\(drug2\)(|\W)'),
                re.compile('drug2(|\s)\(drug1\)(|\W)'),
                re.compile('drug1 such as drug2'),
                re.compile('drug2 such as drug1')]
    
    for re_pattern in patterns:
        if re.search(re_pattern, row['anonymized_text']):
            return 1
    return 0

def preprocess(data):
    '''goes through all the preprocesing steps and returns the resulting dataframe'''
    d = deepcopy(data)
    anonymize_drugs(d)
    d['same_drug'] = d.apply(identify_same_drug, axis = 1)
    d['negative'] = d.apply(identify_negative_phrases, axis = 1)
    d['in_series'] = d.apply(identify_series, axis = 1)
    d['special_cases'] = d.apply(identify_special_cases, axis = 1)
    d['removal_flag'] = d.apply(lambda x: max(x['same_drug'], x['negative'], x['in_series'], x['special_cases']), axis = 1)
    return d

train_data_preprocessed = preprocess(train_data)

train_data_preprocessed.head()

test_data_preprocessed = preprocess(test_data)

def write_to_pickle(data, file_name):
    pickle_out = open(file_name, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()

write_to_pickle(train_data_preprocessed, 'train_data_preprocessed.pickle')

write_to_pickle(test_data_preprocessed, 'test_data_preprocessed.pickle')


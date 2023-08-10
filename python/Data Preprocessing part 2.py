import nltk
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
import re

def open_pickle(data):
    pickle_in = open(data, 'rb')
    return pickle.load(pickle_in)

def write_to_pickle(data, file_name):
    pickle_out = open(file_name, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()

train_data = open_pickle('./train_data_preprocessed.pickle')
test_data = open_pickle('./test_data_preprocessed.pickle')

len(train_data[train_data['removal_flag'] == 1]['ddi_type'])

train_data['ddi_type'].value_counts()

train_data[train_data['removal_flag'] == 1]['ddi_type'].value_counts()

train_data[train_data['removal_flag'] == 0]['ddi_type'].value_counts()

train_data[train_data['negative'] == 1]['ddi_type'].value_counts()

train_data[train_data['in_series'] == 1]['ddi_type'].value_counts()

def identify_negatives(row):
    '''returns 1 if there is a presence of a negative phrase, 0 otherwise'''
    pattern = re.compile("no effect|not affect|no significant|no clinically|not significantly|not alter|no affected|not have|not altered|not influence|not result|no pharmacokinetic|no evidence|not a|no formal|not appear|not apparent|not significant|not been|not known")
    match = re.search(pattern, row['text'].lower())
    if match:
        return 1
    return 0

train_data['negative_phrase'] = train_data.apply(identify_negatives, axis = 1)
test_data['negative_phrase'] = test_data.apply(identify_negatives, axis = 1)

train_data[train_data['negative_phrase'] == 1]['ddi_type'].value_counts()

sum(train_data['negative_phrase'] == 1)

train_data['removal_flag_2'] = train_data.apply(lambda x: max(x['same_drug'], x['negative_phrase'], x['in_series'], x['special_cases']), axis = 1)
test_data['removal_flag_2'] = test_data.apply(lambda x: max(x['same_drug'], x['negative_phrase'], x['in_series'], x['special_cases']), axis = 1)

train_data[train_data['removal_flag_2'] == 1]['ddi_type'].value_counts()

def preprocess(data, train = True):
    d = deepcopy(data)
    d['ddi_type'] = d['ddi_type'].fillna('none')
    output_columns = ['text','tokenized_sentences', 'drug1', 'drug2', 'ddi', 'ddi_type']
    if train:
        d = d[d['removal_flag_2'] == 0]
    
    d['tokenized_sentences'] = d.apply(lambda row: " ".join(nltk.word_tokenize(row['anonymized_text'])), axis = 1)
    
    if not train:
        d_pos = d[d['removal_flag_2'] == 0]
        d_neg = d[d['removal_flag_2'] == 1]
        return (d_pos[output_columns], d_neg[output_columns])
    return d[output_columns]
                

train_data_final = preprocess(train_data, train = True)
test_data_final, test_data_negatives = preprocess(test_data, train = False)

len(train_data_final)

len(test_data_final)

len(test_data_negatives)

def series_2(row):
    patterns = [re.compile("drug1 [;,] ([a-zA-Z0-9*]* [;,])+ drug2"), 
                re.compile("drug1 [;,] (([a-zA-Z0-9*]* ){1,4}[;,])+ drug2")]
    for pattern in patterns:
        if re.search(pattern, row['tokenized_sentences']):
            return 1
    return 0

def preprocess_second_pass(data, train = True):
    output_columns = ['text','tokenized_sentences', 'drug1', 'drug2', 'ddi', 'ddi_type']
    data['series_flag'] = data.apply(lambda row: series_2(row), axis = 1)
    if train:
        return data[data['series_flag'] == 0][output_columns]
    else:
        negs = data[data['series_flag'] == 1][output_columns]
        pos = data[data['series_flag'] == 0][output_columns]
        return (pos, negs)
        

train_data_final_2 = preprocess_second_pass(train_data_final, train = True)
test_data_final_2, test_data_negatives_2 = preprocess_second_pass(test_data_final, train = False)
test_negatives = test_data_negatives.append(test_data_negatives_2)

test_negatives = test_negatives.reset_index(drop = True)
train = train_data_final_2.reset_index(drop = True)
test = test_data_final_2.reset_index(drop = True)

counts_after = train['ddi_type'].value_counts()

counts_before = train_data['ddi_type'].value_counts()
counts_before = counts_before.append(pd.Series([len(train_data) - sum(counts_before)]))
counts_before.index = ['effect', 'mechanism', 'advise', 'int', 'none']

counts_before

counts_after

counts_before - counts_after

counts_after/counts_before

write_to_pickle(train, 'train_complete_processed.pickle')
write_to_pickle(test, 'test_complete_processed.pickle')
write_to_pickle(test_negatives, 'test_negatives_processed.pickle')


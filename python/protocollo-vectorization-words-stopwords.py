import random
import json
from stop_words import get_stop_words

random.seed(12345)

stopwords = get_stop_words('it')
print("the first 20 stopword %s of %d" % (stopwords[:20],len(stopwords)))

input_filename = './protocollo_2017.csv'
output_filename = './protocollo_2017_vectorized_no_stopwords.csv'

def read_data(input_path, separator=",", contains_header=False):
    with open(input_filename, 'r') as f:
        lines = f.read().splitlines()
    
    if contains_header:
        headers = lines[0].split(separator)
        lines = lines[1:]
    else:
        headers = []
    
    data = [l.split(separator) for l in lines]
    return headers, data

headers, data = read_data(input_filename, separator="|", contains_header=True)

headers

data

def preprocess(text,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True, split = ' ', stopwords_lang = 'it'):
    """apply the transformations to the given text.

    # Arguments
        text: Input text (string).
        filters: Sequence of characters to filter out.
        lower: Whether to convert the input to lowercase.
        split: Sentence split marker (string).
        
        return a string of token separated by split
    """
    translate_map = str.maketrans(filters, split * len(filters))
    text = text.translate(translate_map)
    
    if stopwords_lang is not None:
        stopwords = get_stop_words('it')
        text_tokens = [w for w in text.split(split) if w not in stopwords]
        text = ' '.join(text_tokens)
        
    return text.strip()

column = 2
    
def preprocess_dataset(data, column=2):
    data_processed = []
    for v in data:
        v[column] = preprocess(v[column])
        data_processed.append(v)
    return data_processed

data_processed = preprocess_dataset(data, column)
data_processed = preprocess_dataset(data, column=3)
data_processed

def extract_tokens_dict(data, column=2):
    """
        return a pair (index_to_token, token_to_word)
    """
    tokens = set()
    for row in data:
        for v in row[column].split(' '):
            tokens.add(v)
    random.shuffle(list(tokens))
    
    index_to_token = dict(enumerate(tokens))
    index_to_token[len(index_to_token)] = 'UNK'
    token_to_index = {v:k for k,v in index_to_token.items()}
    return index_to_token, token_to_index

index_to_token, token_to_index = extract_tokens_dict(data_processed)

print(index_to_token)
print(token_to_index)

with open('index_to_token_no_stopwords.json','w') as f:
    json.dump(index_to_token,f)
    
with open('token_to_index_no_stopwords.json', 'w') as f:
    json.dump(token_to_index,f)

def vectorize(token_to_index, data, column=2):
    data_transformed = []

    for row in data:
        transformed = [token_to_index[v] for v in row[column].split(' ') if v in token_to_index]
        row[column] = ' '.join([str(x) for x in transformed])
        data_transformed.append(row)
    
    return data_transformed

data_transformed = vectorize(token_to_index, data)

data_transformed

with open(output_filename, 'w') as f:
    for line in data_transformed:
        f.write(','.join(line) + '\n')


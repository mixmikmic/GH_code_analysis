import pickle

def get_corpus_length(path):
    return len(open(path, 'r').readlines())

def get_corpus(path):
    raw = open(path, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            all_x.append(point[:-1])
            point = []  
    return all_x

def get_maxlen(all_x):
    lengths = [len(x) for x in all_x]
    maxlen = max(lengths)
    return maxlen

def build_data(all_x):
    X = [[c[0] for c in x] for x in all_x]
    tags = [[c[1] for c in y] for y in all_x]
    chunks = [[c[2] for c in z] for z in all_x]
    return X, tags, chunks

def build_word_idx_dicts(X):
    all_text = [c for x in X for c in x]
    words = list(set(all_text))
    word2ind = {word: (index+1) for index, word in enumerate(words)}
    ind2word = {(index+1): word for index, word in enumerate(words)}
    return word2ind, ind2word

def build_label_idx_dicts(tags):
    labels = list(set([c for x in tags for c in x]))
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    return label2ind, ind2label

# traing set 
all_x_train = get_corpus('train.txt')
maxlen_train = get_maxlen(all_x_train)
X_train, tags_train, chunks_train = build_data(all_x_train)

# test set 
all_x_test = get_corpus('test.txt')
maxlen_test = get_maxlen(all_x_test)
X_test, tags_test, chunks_test = build_data(all_x_test)

X = X_train + X_test
tags = tags_train + tags_test
chunks = chunks_train + chunks_test

maxlen = max(maxlen_train, maxlen_test)
word2ind, ind2word = build_word_idx_dicts(X)
label2ind, ind2label = build_label_idx_dicts(tags)

# print(X[0])
# print(len(X))
# print(len(word2ind), len(label2ind))
# print(maxlen, maxlen_train, maxlen_test)

data = {'full':{}, 'train':{}, 'test':{}, 'stats':{}}

data['full']['X'] = X
data['full']['tags'] = tags
data['full']['chunks'] = chunks

data['train']['X'] = X_train
data['train']['tags'] = tags_train
data['train']['chunks'] = chunks_train

data['test']['X'] = X_test
data['test']['tags'] = tags_test
data['test']['chunks'] = chunks_test

data['stats']['maxlen'] = maxlen
data['stats']['word2ind'] = word2ind
data['stats']['ind2word'] = ind2word
data['stats']['label2ind'] = label2ind
data['stats']['ind2label'] = ind2label

with open('pos_conll.pkl', 'wb') as f:
    pickle.dump(data, f)


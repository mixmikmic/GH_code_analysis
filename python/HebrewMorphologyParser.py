import sys, collections, os, re

from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from math import sqrt
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

from tf.fabric import Fabric

DATABASE = '~/github'
BHSA = 'bhsa/tf/c'

TF = Fabric(locations=[DATABASE], modules=[BHSA], silent=False )

api = TF.load('''
    lex g_cons sp pfm vbs nme uvf prs vbe language
''')

api.loadLog()
api.makeAvailableIn(globals())

def prepare_data_train(n_examples):
    n_verbs = 0
    max_con = 0
    max_an = 0
    wo_list = []
    info_dict = {}
    len_dict = {}
    alphabet = set()
    for word in F.otype.s('word'):
        if n_verbs < n_examples and not T.bookName(word) in {'Jonah', 'Ruth'} and F.language.v(word) == 'hbo':       
            if F.sp.v(word) != 'nmpr': # proper nouns are excluded
                n_verbs += 1
                vbs = F.vbs.v(word)
                if vbs in {'', 'n/a', 'absent'}:
                    vbs = 'n'
                pfm = F.pfm.v(word)
                if pfm in {'', 'n/a', 'absent'}:
                    pfm = 'n'
                vbe = F.vbe.v(word)
                if vbe in {'', 'n/a', 'absent'}:
                    vbe = 'n'
                nme = F.nme.v(word)
                if nme in {'', 'n/a', 'absent'}:
                    nme = 'n'
                prs = F.prs.v(word)
                if prs in {'', 'n/a', 'absent'}:
                    prs = 'n'
                root = F.lex.v(word).strip('/').strip('[').strip('=')
                an_length = len(vbs) + len(pfm) + len(root) + len(vbe) + len(nme) + len(prs)
                cons = F.g_cons.v(word)
                for elem in [vbs, pfm, cons, vbe, nme, root, prs]:
                    for char in elem:
                        alphabet.add(char)

                con_length = len(cons)
                wo_list.append(word)
                info_dict[word] = [cons, vbs, pfm, root, vbe, nme, prs]
                len_dict = [con_length, an_length]
                if an_length > max_an:
                    max_an = an_length
                if con_length > max_con:
                    max_con = con_length
                    
    alphabet.add(' ')
    alphabet.add('+')
    print('max_an = ' ,max_an)
    return wo_list, info_dict, len_dict, max_con, max_an, list(alphabet)                

def prepare_data_test(n_examples):
    n_verbs = 0
    max_con = 0
    max_an = 0
    wo_list = []
    info_dict = {}
    len_dict = {}
    for word in F.otype.s('word'):
        if n_verbs < n_examples and T.bookName(word) in {'Jonah', 'Ruth'}:       
            if F.sp.v(word) != 'nmpr':
                n_verbs += 1
                vbs = F.vbs.v(word)
                if vbs in {'', 'n/a', 'absent'}:
                    vbs = 'n'
                pfm = F.pfm.v(word)
                if pfm in {'', 'n/a', 'absent'}:
                    pfm = 'n'
                vbe = F.vbe.v(word)
                if vbe in {'', 'n/a', 'absent'}:
                    vbe = 'n'
                nme = F.nme.v(word)
                if nme in {'', 'n/a', 'absent'}:
                    nme = 'n'
                prs = F.prs.v(word)
                if prs in {'', 'n/a', 'absent'}:
                    prs = 'n'
                    
                root = F.lex.v(word).strip('/').strip('[').strip('=')
                an_length = len(vbs) + len(pfm) + len(root) + len(vbe) + len(nme) + len(prs)
                cons = F.g_cons.v(word)

                con_length = len(cons)
                wo_list.append(word)
                info_dict[word] = [cons, vbs, pfm, root, vbe, nme, prs]
                len_dict = [con_length, an_length]

    return wo_list, info_dict, len_dict             

# convert data to strings
def to_string(wo_list, info_dict, len_dict, max_con, max_an):
    
    ystr = list()
    for wo in wo_list:
        strp = info_dict[wo][1] + '+' + info_dict[wo][2] + '+' + info_dict[wo][3] + '+' + info_dict[wo][4] + '+' + info_dict[wo][5] + '+' + info_dict[wo][6]
        strp2 = ''.join([' ' for _ in range(max_an - len(strp))]) + strp
        ystr.append(strp2)
    
    Xstr = list()
    for wo in wo_list:
        strp = info_dict[wo][0]
        conson = ''.join([' ' for _ in range(max_con - len(strp))]) + strp
        Xstr.append(conson)
    return Xstr, ystr

# integer encode strings
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc

# one hot encode
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc

# generate an encoded dataset
def generate_data(wo_list, info_dict, len_dict, max_con, max_an, alphabet):
    
    X, y = to_string(wo_list, info_dict, len_dict, max_con, max_an)
    
    X, y = integer_encode(X, y, alphabet)
    
    X, y = one_hot_encode(X, y, len(alphabet))
    
    X, y = array(X), array(y)
    
    return X, y

# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)

wo_list, info_dict, len_dict, max_con, max_an, alphabet = prepare_data_train(400000)

n_chars = len(alphabet)

n_in_seq_length = max_con

n_out_seq_length = max_an + 5

# define LSTM
model = Sequential()
model.add(LSTM(315, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(265, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

X, y = generate_data(wo_list, info_dict, len_dict, max_con, max_an + 5, alphabet)
print(X.shape)
print(y.shape)
model.fit(X, y, epochs=1, batch_size=32)

# evaluate LSTM
wo_list, info_dict, len_dict = prepare_data_test(2000)
X, y = generate_data(wo_list, info_dict, len_dict, max_con, max_an + 5, alphabet)
loss, acc = model.evaluate(X, y, verbose=0)
print('Loss: %f, Accuracy: %f' % (loss, acc*100))

for _ in range(100):
    # generate an input-output pair
    X, y = generate_data(wo_list, info_dict, len_dict, max_con, max_an + 4, alphabet)
    # make prediction
    yhat = model.predict(X, verbose=0)
    # decode input, expected and predicted
    in_seq = invert(X[_], alphabet)
    out_seq = invert(y[_], alphabet)
    predicted = invert(yhat[_], alphabet)
    print('%s = %s (expect %s)' % (in_seq, predicted, out_seq))


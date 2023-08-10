from keras import objectives, backend as K
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import keras
from keras.layers import Input, Dense, Lambda, Layer
import numpy as np
import uuid
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
import pickle
import itertools

class VAE(object):
    
    def _build_decoder(self, encoded, vocab_size, max_length):
        repeated_context = RepeatVector(max_length)(encoded)

        h = LSTM(500, return_sequences=True, name='dec_lstm_1')(repeated_context)
        h = LSTM(500, return_sequences=True, name='dec_lstm_2')(h)

        decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)

        return decoded
    def _build_encoder(self, x, latent_rep_size=100, max_length=300, epsilon_std=0.01):
        h = Bidirectional(LSTM(500, return_sequences=True, name='lstm_1'), merge_mode='concat')(x)
        h = Bidirectional(LSTM(500, return_sequences=False, name='lstm_2'), merge_mode='concat')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))


    def create(self, vocab_size=1000, max_length=50, latent_rep_size=100):
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        x = Input(shape=(max_length,))
        x_embed = Embedding(vocab_size, 64, input_length=max_length)(x)

        vae_loss, encoded = self._build_encoder(x_embed, latent_rep_size=latent_rep_size, max_length=max_length)
        self.encoder = Model(inputs=x, outputs=encoded)

        encoded_input = Input(shape=(latent_rep_size,))


        decoded = self._build_decoder(encoded_input, vocab_size, max_length)
        self.decoder = Model(encoded_input, decoded)

        self.autoencoder = Model(inputs=x, outputs=[self._build_decoder(encoded, vocab_size, max_length)])
        self.autoencoder.compile(optimizer='Adam',
                                 loss=[vae_loss],
                                 metrics=['accuracy'])

        

MAX_LENGTH

from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

NUM_WORDS=1000
MAX_LENGTH=15
VALIDATION_SPLIT =.3
_EOS = "endofsent"

def sent_parse(sentences,tokenizer=None,build_indices=True):
    if build_indices:
        tokenizer = Tokenizer(nb_words=NUM_WORDS)
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=MAX_LENGTH)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
    else:
        sequences = tokenizer.texts_to_sequences(sentences)
        data = pad_sequences(sequences, maxlen=MAX_LENGTH)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
    return tokenizer,data


    

def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec


def interpolate_b_points(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample


def sent_2_sent(sent1,sent2, model,tokenizer=None):
    _,a = sent_parse([sent1],tokenizer,build_indices=False)
    _,b = sent_parse([sent2],tokenizer,build_indices=False)
    encode_a = model.encoder.predict(a)
    encode_b = model.encoder.predict(b)
    test_hom = interpolate_b_points(encode_a, encode_b, 100)
    index_word = {v: k for k, v in tokenizer.word_index.items()}

    for point in test_hom:
        words=[]
        deco=model.decoder.predict(point)
        #print(deco)
        for seq in deco[0]:
            words.append(index_word[np.argmax(seq)])
            words.append(' ')
        print(''.join(words))
       
        #print_sentence_with_w2v(p,index_word)

import nltk
from nltk.corpus import brown
def split_into_sent (text):
    strg = ''
    for word in text:
        strg += word
        strg += ' '
    strg_cleaned = strg.lower()
    for x in ['\n','"',"!", '#','$','%','&','(',')','*','+',',','-','/',':',';','<','=','>','?','@','[','^',']','_','`','{','|','}','~','\t']:
        strg_cleaned = strg_cleaned.replace(x, '')
    sentences = sent_tokenize(strg_cleaned)
    return sentences

fiction_text=brown.words(categories=['fiction','humor', 'learned', 'lore', 'mystery', 'news'])
sents=split_into_sent(fiction_text)

###### APT Text#############

#with open('C:\\Users\\Vinod\\projects\\keras\\APT_sanitized.txt',"r",encoding='utf-8') as f:
#    texts=f.readlines()


#eostxts=[]
#for txt in texts:
#    eostxts.append(txt + " " + _EOS)



tokenizer,data=sent_parse(sents)

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

X_train = data[:-nb_validation_samples]
X_test = data[-nb_validation_samples:]





print("Training data")
print(X_train.shape)

print("Number of words:")
print(len(np.unique(np.hstack(X_train))))



temp = np.zeros((X_train.shape[0], MAX_LENGTH, NUM_WORDS))
temp[np.expand_dims(np.arange(X_train.shape[0]), axis=0).reshape(X_train.shape[0], 1), np.repeat(np.array([np.arange(MAX_LENGTH)]), X_train.shape[0], axis=0), X_train] = 1

X_train_one_hot = temp

temp = np.zeros((X_test.shape[0], MAX_LENGTH, NUM_WORDS))
temp[np.expand_dims(np.arange(X_test.shape[0]), axis=0).reshape(X_test.shape[0], 1), np.repeat(np.array([np.arange(MAX_LENGTH)]), X_test.shape[0], axis=0), X_test] = 1

x_test_one_hot = temp

def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' +                 model_name
               #model_name + "-{epoch:02d}-{val_decoded_mean_acc:.2f}-{val_pred_loss:.2f}.h5"
    directory = os.path.dirname(filepath)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=False)

    return checkpointer

index_word = {v: k for k, v in tokenizer.word_index.items()}
index_word

from keras.callbacks import TensorBoard
from time import time
def train():
    model = VAE()
    model.create(vocab_size=NUM_WORDS, max_length=MAX_LENGTH)
    
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    checkpointer = create_model_checkpoint('models', 'rnn_ae')
    
    model.autoencoder.fit(x=X_train, y={'decoded_mean': X_train_one_hot},
                          batch_size=100, epochs=10, callbacks=[checkpointer,tensorboard],
                          validation_data=(X_test, {'decoded_mean': x_test_one_hot}))
    return model

model=train()

sent_2_sent(sents[27],sents[28],model,tokenizer=tokenizer)

print(sents[27])
print(sents[29])

sent1 = 'Explorer plugin open source'
_,a = sent_parse([sent1],tokenizer,build_indices=False)
z=model.encoder.predict(a)

deco=model.decoder.predict(z)

np.argmax(deco[0][1])

words=[]
for seq in deco[0]:
    if len(seq):
        words.append(word_index.get(seq[np.argmax(seq)]))
    else:
        words.append(' ')
print(''.join(words))

index_word = {v: k for k, v in tokenizer.word_index.items()}
for seq in deco[0]:
    print(index_word[np.argmax(seq)])

index_word






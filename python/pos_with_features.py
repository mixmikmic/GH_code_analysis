## Read in the data
import json
import random
import numpy

def read_labeled_data(json_file):
    with open(json_file) as f:
        data=json.load(f)
        texts=[one_example["text"] for one_example in data]  #list of texts
        labels=[one_example["tags"] for one_example in data] # list of lists of output labels
    return texts,labels

texts_train,labels_train=read_labeled_data("data/pos_train_fi.json")
texts_devel,labels_devel=read_labeled_data("data/pos_devel_fi.json")       

## Read in pre-trained embeddings
from gensim.models import KeyedVectors

# English model: wiki-news-300d-1M.vec
# Finnish model: pb34_wf_200_v2_skgram.bin
# these models are under /home/bio in the classroom machines
#                        /home/ginter on the virtual server
#                         ...don't make a copy of that file on the virtual server, just use it from that path
#                         ...if you run things locally on your laptop, you can scp this model from the virtual machine
vector_model=KeyedVectors.load_word2vec_format("data/pb34_wf_200_v2_skgram.bin", binary=True, limit=100000)
word_embeddings=vector_model.vectors # these are the vectors themselves

# Just checking all is fine
print("word_embeddings shape=",word_embeddings.shape)
print("embeddings=",word_embeddings)

import keras.utils
# The embeddings have one row for every word, and they are indexed from 0 upwards
# For our tagger, we need words with index 0 and 1 to have a special meaning
#       0 is the mask
#       1 is OOV (out of vocabulary)
# We need to make space for the two words:
# 1) Add two rows into the word_embeddings matrix
# 2) Renumber indices in the gensim model by 2, so that what was word 0 is now word 2, word 1 becomes word 3, etc...

# ad 1:
# Two rows with the right number of columns, and filled with random numbers
two_random_rows=numpy.random.uniform(low=-0.01, high=0.01, size=(2,word_embeddings.shape[1]))
# stack the two rows, and the embedding matrix on top of each other
word_embeddings=numpy.vstack([two_random_rows,word_embeddings])

# Normalize to unit length, works better this way
word_embeddings=keras.utils.normalize(word_embeddings)

# Alternative normalization code
#norm=numpy.linalg.norm(word_embeddings,axis=1,keepdims=True) #magnitude of every row
#word_embeddings/=norm #divide every row by magnitude, results in unit length vectors

# Ad 2:
# Now renumber all word indices, shifting them up by two
for word_record in vector_model.vocab.values():
    word_record.index+=2

print("New embeddings shape=",word_embeddings.shape)
print(word_embeddings)

def vectorize(texts,word_vocab,feature_vocab):
    vectorized_texts=[] # List of sentences, each sentence is a list of words, and each word is a list of features
    for one_text in texts:
        vectorized_text=[] # One sentence, ie list of words, each being a list of features
        for one_word in one_text:
            # feature vector of this one word
            # [ word_itself, last_character, last_two_characters, last_three_characters, 
            #                first character, first_two_characters, first_three_characters, ...]
            one_word_feature_vector=[]
            if one_word in word_vocab:
                one_word_feature_vector.append(word_vocab[one_word].index) # the .index comes from gensim's vocab
            else:
                one_word_feature_vector.append(1) # OOV
            #as a future-proof idea, let us mark the word with a beginning and end marker
            marked="^"+one_word+"$"
            for affix_length in range(2,5): #2,3,4
                suffix=marked[-affix_length:]  # g$  og$  dog$
                prefix=marked[:affix_length]   # ^d  ^do  ^dog
                if len(suffix)==affix_length: #if len(suffix) is less than the desired length, the word is too short
                    one_word_feature_vector.append(feature_vocab.setdefault(suffix,len(feature_vocab)))
                else:
                    one_word_feature_vector.append(1) #No such suffix
                if len(prefix)==affix_length: #if len(prefix) is less than the desired length, the word is too short
                    one_word_feature_vector.append(feature_vocab.setdefault(prefix,len(feature_vocab)))
                else:
                    one_word_feature_vector.append(1) #No such prefix
            
            #Done with the word
            vectorized_text.append(one_word_feature_vector)
        #Done with the text
        vectorized_texts.append(vectorized_text)
    return numpy.array(vectorized_texts)

feature_vocab={"<SPECIAL>":0,"<NOSUCHSUFFIX>":1} #these are just to reserve the indices 0 and 1
vectorized_train=vectorize(texts_train,vector_model.vocab,feature_vocab)
print("First 10 features",list(feature_vocab.items())[:10]) #first 10 features
print("Some text:",vectorized_train[100])

vectorized_devel=vectorize(texts_devel,vector_model.vocab,feature_vocab)

import tensorflow as tf
### Only needed for me, not to block the whole GPU, you don't need this stuff
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
### ---end of weird stuff

from keras.preprocessing.sequence import pad_sequences

# Now we pad the sequences to get everything into the right sizes
padded_train_data=pad_sequences(vectorized_train,padding="post")
print("Padded train shape (texts x words x features):",padded_train_data.shape)
_,longest_train_sent,_=padded_train_data.shape
padded_devel_data=pad_sequences(vectorized_devel,maxlen=longest_train_sent,padding="post")
print("Padded devel shape (texts x words x features):",padded_devel_data.shape)

# Now the training data input part is done ... labels needed yet
# Easiest way is to make our own vectorizer
def vectorize_labels(labels,label_dictionary):
    vectorized=[]
    for one_text_labels in labels: #List like ["NOUN","VERB","VERB","PUNCT"]
        one_text_vectorized=[] #numerical indices of the labels
        for one_label in one_text_labels:
            one_text_vectorized.append(label_dictionary.setdefault(one_label,len(label_dictionary)))
        vectorized.append(one_text_vectorized) #done with the sentence
    return numpy.array(vectorized)

label_dictionary={}
vectorized_train_labels=vectorize_labels(labels_train,label_dictionary)
padded_train_labels=pad_sequences(vectorized_train_labels,padding="post")
print("padded_train_labels shape=",padded_train_labels.shape)
vectorized_devel_labels=vectorize_labels(labels_devel,label_dictionary)
padded_devel_labels=pad_sequences(vectorized_devel_labels,padding="post",maxlen=longest_train_sent)
print("padded_devel_labels shape=",padded_devel_labels.shape)

# Almost there ... we yet need the mask, telling which parts of each padded sequence are real words
# and which are only the padding which should be ignored in the output

#                           where(condition,value_if_true,value_if_false)
# padded_train_data[:,:,0]  -> returns the first feature of every word, i.e. the index of this word in the vocabulary
# here zero means padding
sentence_mask_train = numpy.where(padded_train_data[:,:,0]>0,1,0)
print(sentence_mask_train[:3])

sentence_mask_devel = numpy.where(padded_devel_data[:,:,0]>0,1,0) 

# phew, finally everything in place:

print("padded_train_data.shape",padded_train_data.shape)
print("padded_train_labels.shape",padded_train_labels.shape)
print("padded_devel_data.shape",padded_devel_data.shape)
print("padded_devel_labels.shape",padded_devel_labels.shape)

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Activation, Conv1D, TimeDistributed
from keras.layers import Bidirectional, Concatenate,Flatten,Reshape
from keras.optimizers import SGD, Adam
from keras.initializers import Constant
from keras.layers import CuDNNLSTM as LSTM  #massive speedup on graphics cards
#from keras.layers import LSTM
from keras.callbacks import EarlyStopping



example_count, sequence_len, feature_count = padded_train_data.shape
_,word_embedding_dim=word_embeddings.shape
feature_embedding_dim=100 #we need to decide on an embedding for the features

word_input=Input(shape=(sequence_len,))
feature_input=Input(shape=(sequence_len,feature_count-1)) #first feature is word, so feature_count-1
word_embeddings_layer=Embedding(len(vector_model.vocab)+2,                     word_embedding_dim, mask_zero=False,                     trainable=False, weights=[word_embeddings])(word_input)
feature_embeddings_layer=Embedding(len(feature_vocab),feature_embedding_dim,embeddings_initializer=Constant(value=0.1))(feature_input)
feature_embeddings_layer_concat=Reshape((sequence_len,(feature_count-1)*feature_embedding_dim))(feature_embeddings_layer)
word_and_f_emb_layer=Concatenate()([word_embeddings_layer,feature_embeddings_layer_concat])
hidden_layer=TimeDistributed(Dense(100,activation="tanh"))(word_and_f_emb_layer)  #Simple
outp_layer=TimeDistributed(Dense(len(label_dictionary),activation="softmax"))(hidden_layer)


model=Model(inputs=[word_input,feature_input], outputs=[outp_layer])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", sample_weight_mode='temporal',weighted_metrics=["acc"])

print(model.summary())
word_input_data_train=padded_train_data[:,:,0]
feature_input_data_train=padded_train_data[:,:,1:]
labels_output_train=numpy.expand_dims(padded_train_labels,-1)

word_input_data_devel=padded_devel_data[:,:,0]
feature_input_data_devel=padded_devel_data[:,:,1:]
labels_output_devel=numpy.expand_dims(padded_devel_labels,-1)

print("word input shape",word_input_data_train.shape)
print("feature input shape",feature_input_data_train.shape)
print("output shape",labels_output_train.shape)
# train
# stop early
es_callback=EarlyStopping(monitor='val_weighted_acc', min_delta=0, patience=2, verbose=1, mode='auto')
hist=model.fit([word_input_data_train,feature_input_data_train],[labels_output_train],               validation_data=([word_input_data_devel,feature_input_data_devel],[labels_output_devel],sentence_mask_devel),               batch_size=200,sample_weight=sentence_mask_train,verbose=1,epochs=20,callbacks=[es_callback])


import json
import random
with open("data/imdb_train.json") as f:
    data=json.load(f)
random.shuffle(data) 
print(data[0])

# We need to gather the texts, into a list
texts=[one_example["text"] for one_example in data]
labels=[one_example["class"] for one_example in data]
print(texts[:2])
print(labels[:2])

from gensim.models import KeyedVectors

vector_model=KeyedVectors.load_word2vec_format("data/wiki-news-300d-1M.vec", binary=False, limit=1000000)

# sort based on the index to make sure they are in the correct order
words=[k for k,v in sorted(vector_model.vocab.items(), key=lambda x:x[1].index)]
print("Words from embedding model:",len(words))
print("First 50 words:",words[:50])

print("Before normalization:",vector_model.get_vector("in")[:10])
vector_model.init_sims(replace=True)
print("After normalization:",vector_model.get_vector("in")[:10])

from sklearn.feature_extraction.text import CountVectorizer
import numpy
analyzer=CountVectorizer(lowercase=False).build_analyzer() # includes tokenizer and preprocessing
print(analyzer(texts[0]))


# init the vectorizer vocabulary using words from the embedding model
def init_vocabulary(vocab, text, text_analyzer):
    for word in analyzer(text):
        vocab.setdefault(word, len(vocab))
    return vocab

words_from_model=" ".join(words[:50000]) # use 50K words from the embedding model to initialize the vocabulary --> expands the learned vocabulary
vocabulary={"<SPECIAL>": 0} # zero has a special meaning in sequence models, prevent using it for a normal word
vocabulary=init_vocabulary(vocabulary, words_from_model, analyzer)
print("Words from embedding model:",len(vocabulary))

def vectorizer(vocab, texts):
    vectorized_data=[] # turn text into numbers based on our vocabulary mapping
    for one_example in texts:
        vectorized_example=[]
        for word in analyzer(one_example):
            vocab.setdefault(word, len(vocab)) # add word to our vocabulary if it does not exist
            vectorized_example.append(vocab[word])
        vectorized_data.append(vectorized_example)
    
    vectorized_data=numpy.array(vectorized_data) # turn python list into numpy matrix
    return vectorized_data, vocab

vectorized_data, vocabulary=vectorizer(vocabulary, texts)

# now vectorized data is the same as feature_matrix, but in different format
print("Words in vocabulary:",len(vocabulary))
print("Vectorized data shape:",vectorized_data.shape)
print("First example vectorized:",vectorized_data[0])
inversed_vocabulary={value:key for key, value in vocabulary.items()} # inverse the dictionary
print("First example text:",[inversed_vocabulary[idx] for idx in vectorized_data[0]])
        

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder=LabelEncoder() #Turns class labels into integers
one_hot_encoder=OneHotEncoder(sparse=False) #Turns class integers into one-hot encoding
class_numbers=label_encoder.fit_transform(labels)
print("class_numbers shape=",class_numbers.shape)
print("class_numbers",class_numbers)
print("class labels",label_encoder.classes_)
#And now yet the one-hot encoding
classes_1hot=one_hot_encoder.fit_transform(class_numbers.reshape(-1,1))
print("classes_1hot",classes_1hot)

def load_pretrained_embeddings(vocab, embedding_model):
    """ vocab: vocabulary from our data vectorizer, embedding_model: model loaded with gensim """
    pretrained_embeddings=numpy.random.uniform(low=-0.05, high=0.05, size=(len(vocab),embedding_model.vectors.shape[1])) # initialize new matrix (words x embedding dim)
    found=0
    for word,idx in vocab.items():
        if word in embedding_model.vocab:
            pretrained_embeddings[idx]=embedding_model.get_vector(word)
            found+=1
            
    print("Found pretrained vectors for {found} words.".format(found=found))
    return pretrained_embeddings

pretrained=load_pretrained_embeddings(vocabulary, vector_model)
print("Shape of pretrained embeddings:",pretrained.shape)
print("Vector for the word 'in':",pretrained[vocabulary["in"]][:10])

import tensorflow as tf
### Only needed for me, not to block the whole GPU, you don't need this stuff
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
### ---end of weird stuff

from keras.preprocessing.sequence import pad_sequences
print("Old shape:", vectorized_data.shape)
vectorized_data_padded=pad_sequences(vectorized_data, padding='post')
print("New shape:", vectorized_data_padded.shape)
print("First example:", vectorized_data_padded[0])

from keras.layers import Layer
from keras import backend as K

class MaskedAveragePooling(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedAveragePooling, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, perm=[0,2,1])
            x = x * mask
        return K.sum(x, axis=1) / K.sum(mask, axis=1)

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return input_shape[0], input_shape[2]

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Activation
from keras import backend as K
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam


example_count,sequence_len=vectorized_data_padded.shape
example_count,class_count=classes_1hot.shape

vector_size=pretrained.shape[1] # embedding dim ("hidden layer") must be the same as in the pretrained model

inp=Input(shape=(sequence_len,))
embeddings=Embedding(len(vocabulary), vector_size, mask_zero=True, weights=[pretrained])(inp)
average_embeddings=MaskedAveragePooling()(embeddings) # custom layer we defined above
tanh=Activation("tanh")(average_embeddings)
outp=Dense(class_count, activation="softmax")(tanh)
model=Model(inputs=[inp], outputs=[outp])

optimizer=Adam(lr=0.0001) # define the learning rate
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])

print(model.summary())

# train
hist=model.fit(vectorized_data_padded,classes_1hot,batch_size=100,verbose=1,epochs=50,validation_split=0.1)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print("History:",hist.history["val_acc"])
print("Max accuracy:",numpy.max(hist.history["val_acc"]))
plt.ylim(0.85,1.0)
plt.plot(hist.history["val_acc"],label="Validation set accuracy")
plt.plot(hist.history["acc"],label="Training set accuracy")
plt.legend()
plt.show()


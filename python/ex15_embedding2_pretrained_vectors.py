import os
imdb_dir="aclImdb"
train_dir=os.path.join(imdb_dir,"train")

labels=[]
texts=[]
for label_type in ["neg","pos"]:
    dir_name=os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:]==".txt":
            f=open(os.path.join(dir_name,fname),encoding="UTF-8")
            texts.append(f.read())
            f.close()
            if label_type=="neg":
                labels.append(0)
            else:
                labels.append(1)
print(labels[0])
print(texts[0])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen=20
max_words=10000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)

data=pad_sequences(sequences,maxlen=maxlen)
labels=np.asarray(labels)
indexs=np.arange(data.shape[0])
np.random.shuffle(indexs)
x_train=data[indexs];y_train=labels[indexs]

glove_dir="glove.6B.100d"
embedding_index={}
f=open(os.path.join(glove_dir,"glove.6B.100d.txt"),encoding="UTF-8")
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype="float32")##turn string into float32
    embedding_index[word]=coefs
f.close()
print("Found %s words. "%len(embedding_index))

embedding_dim=100
embedding_matrix=np.zeros((max_words,embedding_dim))
word_index=tokenizer.word_index
for word,i in word_index.items():
    embedding_vector=embedding_index.get(word)
    if i<max_words and embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense

model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
#model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False

model.compile(optimizer="rmsprop",
             loss="binary_crossentropy",
             metrics=["acc"])
history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2)




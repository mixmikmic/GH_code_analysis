from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)
print("Training data' shape:"+str(train_data.shape)+". Training labels' shape:"+str(train_labels.shape))
print("Testing data' shape:"+str(test_data.shape)+". Testing labels' shape:"+str(test_labels.shape))
print("The class nums is "+str(np.max(train_labels)+1))

dic=reuters.get_word_index()
dic_rev=dict([(v,k) for (k,v) in dic.items()])
decoded_newswires=" ".join([dic_rev.get(i-3,"?") for i in train_data[0]])
print("The content is:"+decoded_newswires)
print("The label is:"+str(train_labels[0]))

import numpy as np
from keras.utils.np_utils import to_categorical

def vectorize(seqs,dim=10000):
    outs=np.zeros((len(seqs),dim))
    for i,seq in enumerate(seqs):
        outs[i,seq]=1
    return outs

x_train=vectorize(train_data);y_train=to_categorical(train_labels)
x_test=vectorize(test_data);y_test=to_categorical(test_labels)

print("One sample of x_train:"+str(x_train[0]))
print("One sample of y_train:"+str(y_train[0])+" The label is:"+str(np.argmax(y_train[0])))

from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(64,activation="relu",input_shape=(10000,)))
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(46,activation="softmax"))

model.compile(optimizer="rmsprop",
             loss="categorical_crossentropy",
             metrics=["accuracy"])

history=model.fit(x_train[1000:],y_train[1000:],
                 epochs=20,batch_size=512,
                 validation_data=(x_train[:1000],y_train[:1000]))

import matplotlib.pyplot as plt
history=history.history
train_acc=history["acc"];val_acc=history["val_acc"]
train_loss=history["loss"];val_loss=history["val_loss"]
epochs=np.arange(1,21)
plt.plot(epochs,train_acc,"b",label="Training Acc");plt.plot(epochs,val_acc,"r",label="Validation Acc")
plt.legend(loc="best");plt.title("ACC");plt.xlabel("Epochs");plt.ylabel("ACC");plt.show()

plt.plot(epochs,train_loss,"b",label="Training Loss");plt.plot(epochs,val_loss,"r",label="Validation Loss")
plt.legend(loc="best");plt.title("Loss");plt.xlabel("Epochs");plt.ylabel("Losses");plt.show()

results=model.evaluate(x_test,y_test)

print(results)




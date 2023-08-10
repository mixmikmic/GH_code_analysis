import json
import random
random.seed(0)
with open("data/reuters_51cls.json") as f:
    data=json.load(f)
random.shuffle(data) #play it safe!
print(data[0]) #Every item is a dictionary with `text` and `class` keys, here's one:

# We need to gather the texts, into a list
texts=[one_example["text"] for one_example in data]
labels=[one_example["class"] for one_example in data]
print(texts[:2])
print(labels[:2])

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


vectorizer=CountVectorizer(max_features=100000,binary=True,ngram_range=(1,2))
feature_matrix=vectorizer.fit_transform(texts)
print("shape=",feature_matrix.shape)
print(feature_matrix)
#print(feature_matrix.todense())



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

import h5py
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint

def save_model(file_name,model,label_encoder,vectorizer):
    """Saves model structure and vocabularies"""
    model_json = model.to_json()
    with open(file_name+".model.json", "w") as f:
        print(model_json,file=f)
    with open(file_name+".vocabularies.json","w") as f:
        classes=list(label_encoder.classes_)
        vocab=dict(((str(w),int(idx)) for w,idx in vectorizer.vocabulary_.items()))
        json.dump((classes,vocab),f,indent=2)
        
example_count,feature_count=feature_matrix.shape
example_count,class_count=classes_1hot.shape

inp=Input(shape=(feature_count,))
hidden=Dense(300,activation="tanh")(inp)
outp=Dense(class_count,activation="softmax")(hidden)
model=Model(inputs=[inp], outputs=[outp])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

# Save model and vocabularies
save_model("models/reuters_51cls_bow",model,label_encoder,vectorizer)
# Callback function to save weights during training
save_cb=ModelCheckpoint(filepath="models/reuters_51cls_bow.weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
hist=model.fit(feature_matrix,classes_1hot,batch_size=100,verbose=1,epochs=10,validation_split=0.1,callbacks=[save_cb])

import numpy
from sklearn.metrics import classification_report, confusion_matrix

#Validation data used during training:
val_instances,val_labels_1hot,_=hist.validation_data

print("Network output=",model.predict(val_instances))
predictions=numpy.argmax(model.predict(val_instances),axis=1)
print("Maximum class for each example=",predictions)
gold=numpy.nonzero(val_labels_1hot)[1] #undo 1-hot encoding
conf_matrix=confusion_matrix(list(gold),list(predictions))
print(conf_matrix)

### FIXED VERSION (thanks for reporting the bug during the lecture!)
### 
gold_labels=label_encoder.inverse_transform(list(gold))
predicted_labels=label_encoder.inverse_transform(list(predictions))
print("Gold labels=",gold_labels)
print("Predicted labels=",predicted_labels)
print(classification_report(gold_labels,predicted_labels))


import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec

#import relevant data
with_time = True
if with_time:
    control_data = pd.read_csv("./cleaned_data/control_w_time.csv", index_col = 0)
    case_data = pd.read_csv("./cleaned_data/case_w_time.csv", index_col = 0)
    all_events = pd.read_csv("./cleaned_data/events_id_w_time.csv", index_col = 0)
    word2vec_model = Word2Vec.load("./word2vec_model/w2vmodel_wt")
    
    control_demo = pd.read_csv("./cleaned_data/control_demo_wt.csv", index_col = 0)
    case_demo = pd.read_csv("./cleaned_data/case_demo_wt.csv", index_col = 0)
else:
    control_data = pd.read_csv("./cleaned_data/control.csv", index_col = 0)
    case_data = pd.read_csv("./cleaned_data/case.csv", index_col = 0)
    all_events = pd.read_csv("./cleaned_data/events_id.csv", index_col = 0)
    word2vec_model = Word2Vec.load("./word2vec_model/w2vmodel")
    
    control_demo = pd.read_csv("./cleaned_data/control_demo.csv", index_col = 0)
    case_demo = pd.read_csv("./cleaned_data/case_demo.csv", index_col = 0)

#construct an input list of arrays with event index (start from 1)
control_temp = control_data.groupby("SUBJECT_ID").apply(lambda x: x.EVE_INDEX.values)
case_temp = case_data.groupby("SUBJECT_ID").apply(lambda x: x.EVE_INDEX.values)

control_temp = control_temp.sort_index()
case_temp = case_temp.sort_index()

control_patients = control_temp.index.values
case_patients = case_temp.index.values
all_patients = np.concatenate([control_patients,case_patients])

#construct labels
Y_control = np.zeros(len(control_patients))
Y_case = np.ones(len(case_patients))
Y = np.concatenate([Y_control,Y_case])

#set up demographic input layer 
#group by operation automatic sorts the subject id so demographic data is in the same order as events data)
X_demo_control = control_demo.sort_values(by="SUBJECT_ID")
X_demo_case = case_demo.sort_values(by="SUBJECT_ID")
X_demo = np.concatenate([X_demo_control.ix[:,1:3].values ,X_demo_case.ix[:,1:3].values])

#find maximum number of events(used for set the parameters of embedding layer
#all sequences of events are padded to the max length)
c_max = control_data.groupby("SUBJECT_ID")["EVE_INDEX"].count().max()
ca_max = case_data.groupby("SUBJECT_ID")["EVE_INDEX"].count().max()
max_num_event_patient = np.max([c_max,ca_max])

#contruct training set of sequences with paddings, so all the sequence has the same length of max_length, 
#with 0s padded before in shorter sequences
np.random.seed(seed=6250)
from keras.preprocessing.sequence import pad_sequences

X_control = [np.array(events).astype("int") for events in control_temp]
X_case = [np.array(events).astype("int") for events in case_temp]
X_all = np.concatenate([X_control,X_case])

X = pad_sequences(X_all, maxlen=None)

X.shape

print(X[0].shape[0])
print (X.shape)
print (Y.shape)

#shuffle
shuffled_index = np.random.permutation(len(all_patients))

#split train, dev, test set 7:1:2
train_index = shuffled_index[:int(len(all_patients)*0.7)]
dev_index =  shuffled_index[int(len(all_patients)*0.7):int(len(all_patients)*0.8)]
test_index =  shuffled_index[int(len(all_patients)*0.8):]

print (shuffled_index.shape)
print (train_index.shape, dev_index.shape, test_index.shape)

all_patients_shuffle = all_patients[shuffled_index]

Y_train = Y[train_index]
Y_dev = Y[dev_index]
Y_test = Y[test_index]

X_train = X[train_index]
X_dev = X[dev_index]
X_test = X[test_index]

X_demo_train = X_demo[train_index]
X_demo_dev = X_demo[dev_index]
X_demo_test = X_demo[test_index]

#contruct embedding matrix dim of (number of different events, dim of embedding)
num_events = len(all_events)
dim_embedding = len(word2vec_model.wv["1"])
#adding the dummy row for padding at index "0"
embedding_matrix = np.zeros((num_events+1, dim_embedding))
for i in range(1, num_events+1):
    embedding_matrix[i] = word2vec_model.wv[str(i)]

embedding_matrix.shape

print (embedding_matrix.shape)

max_num_event_patient

#set up the model
from keras.layers import Conv1D, Dense, Input,GlobalMaxPooling1D, MaxPooling1D, concatenate, Embedding,BatchNormalization, Dropout
from keras.optimizers import SGD, Adam, RMSprop,Nadam
from keras.models import Model
from keras.models import load_model
from keras.initializers import glorot_normal

input_events = Input(shape=(max_num_event_patient,))
init = glorot_normal(seed = 6250)

#embedding layer
embedding_raw = Embedding(num_events+1, dim_embedding,embeddings_initializer = init, input_length=max_num_event_patient)(input_events)
embedding = BatchNormalization()(embedding_raw)
#1D conv
x_2 = Conv1D(filters = 4, kernel_size = 2,padding = "valid", activation='relu')(embedding)
x_3 = Conv1D(filters = 4, kernel_size = 3,padding = "valid", activation='relu')(embedding)
x_4 = Conv1D(filters = 4, kernel_size = 4,padding = "valid", activation='relu')(embedding)
x_5 = Conv1D(filters = 4, kernel_size = 5,padding = "valid", activation='relu')(embedding)

pool_2 = GlobalMaxPooling1D()(x_2)
pool_3 = GlobalMaxPooling1D()(x_3)
pool_4 = GlobalMaxPooling1D()(x_4)
pool_5 = GlobalMaxPooling1D()(x_5)

#need to adjust the shape for demographics if demo feature changes
#now its age, sex (tried ethnicity(one hot dim = 5) but not working well)

input_demo = Input(shape=(2,))

#concatenate max_pooling results
patient_embed = concatenate([input_demo,pool_2, pool_3,pool_4,pool_5])

#fully connected part
dense1 = Dense(6, activation = "relu")(patient_embed)
dense1 = Dropout(0.2)(dense1)
output = Dense(1, activation = "sigmoid" )(dense1)

model = Model(inputs=[input_events, input_demo], outputs=output)
model.compile(optimizer=Nadam(lr = 0.0005),loss='binary_crossentropy',metrics=['accuracy'])

model.fit([X_train, X_demo_train],Y_train, epochs=20,batch_size =64,shuffle=True,validation_data=([X_dev, X_demo_dev], Y_dev))

get_ipython().system('rm cnn-model.h5')
model.save('cnn-model.h5')

model.evaluate([X_test, X_demo_test], Y_test)

from sklearn import metrics
print metrics.roc_auc_score(Y_test, model.predict([X_test, X_demo_test]))

#get embedding
patient_embedding=Model(inputs=[input_events, input_demo], outputs=patient_embed).predict([X, X_demo])

patient_embedding.shape

from sklearn.manifold import TSNE
from time import time

#use t-sne to visualize patient representations in control group and case group
print("Computing t-SNE embedding")
tsne = TSNE(n_components=2, perplexity =2,init='pca', random_state=0,method='exact')
t0 = time()
patient_tsne = tsne.fit_transform(patient_embedding)

t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.utils import check_random_state

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1,1,1)
plt.scatter(patient_tsne[:,0], patient_tsne[:,1], c=Y,cmap=plt.cm.rainbow_r)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()



#get case embeddings
case_embedding = patient_embedding[Y==1]

case_embedding.shape

from sklearn.cluster import KMeans
get_ipython().magic('matplotlib inline')

kmeans_wss=[]
kmeans_idx=[]
for i in range(2,20):

    kmeans_clustering = KMeans( n_clusters = i).fit(case_embedding)
    kmeans_idx.append(kmeans_clustering.labels_)
    kmeans_wss.append(kmeans_clustering.inertia_)


#find good # of clusters 
plt.plot(range(2,20), kmeans_wss)

case_class = KMeans(n_clusters = 3).fit(case_embedding).labels_ +1

#visualize and tag tsne graph using kmeans result 
print("Computing t-SNE embedding")
tsne = TSNE(n_components=3, perplexity =10,init='pca', learning_rate=500,early_exaggeration=2.0,random_state=6250,method='exact')  
t0 = time()
case_tsne = tsne.fit_transform(case_embedding)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
x_min, x_max = np.min(case_tsne, 0), np.max(case_tsne, 0)
case_tsne = (case_tsne - x_min) / (x_max - x_min)

plt.figure(figsize=(8,6))
ax = plt.subplot(111)
for i in range(case_tsne.shape[0]):
    plt.text(case_tsne[i, 0], case_tsne[i, 1], str(case_class[i]),
                color=plt.cm.Set1(case_class[i]/10.0),
                fontdict={'weight': 'bold', 'size': 9})

plt.xticks([]), plt.yticks([])
plt.title("t-SNE: %.2g sec" % (t1 - t0))

from nltk import ngrams
import operator

def topk(X_case, case_class, cluster,k,n):
    codes_class=[]
    d={}
    for i in np.where(case_class==cluster)[0]:
        codes_class += list(ngrams(X_case[i],n))
    for e in codes_class:
        d[e] = d.get(e,0)+1
    d2 = {k: v/float(sum(d.values())) for (k,v) in d.items()}

    sorted_code = sorted(d2.items(), key=operator.itemgetter(1),reverse=True)

    return [x[0] for x in sorted_code[:k]]

event_id_dict = dict(zip(all_events.EVE_INDEX,all_events.EVENTS))

def top_grams(event_pairs):
    return [tuple(event_id_dict[x] for x in tup) for (i,tup) in enumerate(event_pairs)]



import pprint
pp = pprint.PrettyPrinter()
for c in range(1,4):
    pp.pprint('cluster:' + str(c))
    onegram = topk(X_case,case_class,c,10,1)
    twogram = topk(X_case,case_class,c,10,2)
    threegram = topk(X_case,case_class,c,10,3)
    pp.pprint(top_grams(onegram))
    pp.pprint(top_grams(twogram))
    pp.pprint(top_grams(threegram))
    print("")
    print("")
    

#model2 = Model(inputs = input_events_r,outputs = x_2_r)
model2 = Model(inputs = input_events,outputs = x_2)
model3 = Model(inputs = input_events,outputs = x_3)
model4 = Model(inputs = input_events,outputs = x_4)

model2.predict(X).shape

#the temperal dimention of output size is max_events - n 
patient_feature_weights2 = model2.predict(X).reshape(4, 1944, 491)
patient_feature_weights3 = model3.predict(X).reshape(4, 1944, 490)
patient_feature_weights4 = model4.predict(X).reshape(4, 1944, 489)

def get_top_events(filter_output, top_num = 500, filter_size = 2):
    top = filter_output.flatten().argsort()[-top_num:]
    
    top_feature_med = []
    top_feature_diag = []
    top_feature_proc = []
    
    event_width = filter_output.shape[1]
    for idx in top:
        patient_idx = int(np.floor(idx/event_width))
        event_idx =  idx%event_width
        real_patient_id = all_patients_shuffle[patient_idx]
        real_event_id = X[patient_idx,event_idx:event_idx+2]
        if np.sum(real_event_id) !=0:
            for eve in real_event_id:
                if eve >= 1612:
                    top_feature_med.append(eve)
                elif eve < 1612 and eve >= 944:
                    top_feature_proc.append(eve)
                elif eve < 944 and eve > 0:
                    top_feature_diag.append(eve)
                
    top_counts_med = pd.Series(top_feature_med).value_counts()
    top_counts_diag = pd.Series(top_feature_diag).value_counts()
    top_counts_proc = pd.Series(top_feature_proc).value_counts()
    
    top_counts_med = pd.DataFrame({"index": top_counts_med.index, "counts":top_counts_med})
    top_counts_diag = pd.DataFrame({"index": top_counts_diag.index, "counts":top_counts_diag})
    top_counts_proc = pd.DataFrame({"index": top_counts_proc.index, "counts":top_counts_proc})
    
    return top_counts_med, top_counts_diag, top_counts_proc

eve_list_med = []
eve_list_diag = []
eve_list_proc = []
for i in range(4):
    te2_med, te2_diag, te2_proc = get_top_events(patient_feature_weights2[i])
    te3_med, te3_diag, te3_proc = get_top_events(patient_feature_weights3[i])
    te4_med, te4_diag, te4_proc = get_top_events(patient_feature_weights4[i])
    
    eve_list_med.append(te2_med)
    eve_list_med.append(te3_med)
    eve_list_med.append(te4_med)
    
    eve_list_diag.append(te2_diag)
    eve_list_diag.append(te3_diag)
    eve_list_diag.append(te4_diag)
    
    eve_list_proc.append(te2_proc)
    eve_list_proc.append(te3_proc)
    eve_list_proc.append(te4_proc)
    

eve_counts_d = pd.concat(eve_list_diag, axis = 0).groupby("index").sum().sort_values(by = "counts", ascending = False)[:10]
final_top_diag = eve_counts_d.merge(all_events, how = "left", left_index = True, right_on = "EVE_INDEX")

eve_counts_p = pd.concat(eve_list_proc, axis = 0).groupby("index").sum().sort_values(by = "counts", ascending = False)[:10]
final_top_proc = eve_counts_p.merge(all_events, how = "left", left_index = True, right_on = "EVE_INDEX")

eve_counts_m = pd.concat(eve_list_med, axis = 0).groupby("index").sum().sort_values(by = "counts", ascending = False)[:10]
final_top_med = eve_counts_m.merge(all_events, how = "left", left_index = True, right_on = "EVE_INDEX")

final_top_diag 

final_top_med



final_top_output = pd.DataFrame({"Med": final_top_med["EVENTS"].values
                                 ,"Diag": final_top_diag["EVENTS"].apply(lambda x: x[-3:]).values
                                 ,"Proc":final_top_proc["EVENTS"].apply(lambda x: x[-3:]).values}, index = np.arange(10))

raw_icd_diag = pd.read_csv("raw_mimic_data/D_ICD_DIAGNOSES.csv")
raw_icd_proc = pd.read_csv("raw_mimic_data/D_ICD_PROCEDURES.csv")

final_top_output.to_csv("strong_indicators.csv")

final_top_output
















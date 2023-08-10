#in case you forgot, this is how one can load the embeddings

from gensim.models import KeyedVectors

# load embedding model from a file
# binary: True if saved as binary file (.bin), False if saved as text file (.vectors or .txt for example)
# limit: How many words to read from the model
model_english=KeyedVectors.load_word2vec_format("data/gigaword-and-wikipedia.bin", binary=True, limit=100000)
model_finnish=KeyedVectors.load_word2vec_format("data/pb34_wf_200_v2_skgram.bin", binary=True, limit=100000)

print("Most similar words for 'locomotive':")
print(model_english.most_similar("locomotive",topn=10))
print()
print("Most similar words for 'veturi':")
print(model_finnish.most_similar("veturi",topn=10))
print()

from googletrans import Translator
translator=Translator()
translations=translator.translate(["locomotive","milk"],src="en",dest="fi")
for t in translations:
    print("origin=",t.origin,"text=",t.text)

print("English vocab",len(model_english.vocab))
print(model_english.vocab.__class__)
print(model_english.vocab["car"])
#We need a list, in order of frequency
words=sorted(model_english.vocab.items(),key=lambda word_dim:word_dim[1].count,reverse=True)
print(words[:5])
words_freq_sorted=[w for w,_ in words]
print("Freq sorted",words_freq_sorted[:5])

import re
english_word_re=re.compile("^[a-zA-Z]+$") #about as stupid simplification as you can get!
final_word_list=[]
for w in words_freq_sorted:
    if english_word_re.match(w):
        final_word_list.append(w)
print(final_word_list[:20])

import re

#same thing as above, nicely packed into a function
def clean_vocab(gensim_model,regexp):
    words=sorted(gensim_model.vocab.items(),key=lambda word_dim:word_dim[1].count,reverse=True)
    words_freq_sorted=[w for w,_ in words]
    word_re=re.compile(regexp)
    final_word_list=[]
    for w in words_freq_sorted:
        if word_re.match(w):
            final_word_list.append(w)
    return final_word_list

finnish_vocab=clean_vocab(model_finnish,'^[a-zA-ZäöåÖÄÅ]+$')
english_vocab=clean_vocab(model_english,'^[a-zA-Z]+$')
print("Final Finnish",finnish_vocab[:15],"...",finnish_vocab[2000:2015])
print("Final English",english_vocab[:15],"...",english_vocab[2000:2015])

#Little test
import time
def translate(words,src,dest,batch_size=1000):
    result=[] #[("dog","koira"),....]
    translator=Translator()
    for idx in range(0,len(words),batch_size):
        batch=words[idx:idx+batch_size]
        try:
            translations=translator.translate(batch,src=src,dest=dest)
            for t in translations:
                result.append((t.origin,t.text))
            time.sleep(0.2) #sleep between batches
            print(src,"->",dest,"batch at",idx,"....OK")
        except: #we end here, if the lines between try ... except throw an error
            print(src,"->",dest,"batch at",idx,"....FAILED")
            time.sleep(61) #sleep a little longer so Google is not angry
            print(src,"->",dest,"...RESTARTING")
            
    return result

x=translate(english_vocab[:50],"en","fi",20) # a small test

print(x)

import json
en_fi=translate(english_vocab,"en","fi",batch_size=150)
with open("en_fi_transl.json","wt") as f:
    json.dump(en_fi,f)
fi_en=translate(finnish_vocab,"fi","en",batch_size=150)
with open("fi_en_transl.json","wt") as f:
    json.dump(fi_en,f)

#dump 10K words at a time into a file, which can be fed to google translate
def build_files(words,fname,batch_size):
    for idx in range(0,len(words),batch_size):
        batch=words[idx:idx+batch_size]
        with open("trdata/{}_batch_{}.txt".format(fname,idx),"wt") as f:
            print("\n".join(batch),file=f)

build_files(english_vocab,"en-fi-source",10000)
build_files(finnish_vocab,"fi-en-source",10000)

get_ipython().run_cell_magic('bash', '', '\nls trdata/fien_* trdata/enfi_*\nwc -l trdata/fien_* trdata/enfi_*\necho "FI -> EN"\npaste trdata/fien_source_all.txt trdata/fien_target_all.txt  | head -n 10\necho "EN -> FI"\npaste trdata/enfi_source_all.txt trdata/enfi_target_all.txt  | head -n 10\n')

fien=[] #list of (fin,eng) pairs obtained from the fin -> eng direction
enfi=[] #list of (fin,eng) pairs, this time obtained from  the eng->fin direction
with open("trdata/fien_source_all.txt") as fi_file, open("trdata/fien_target_all.txt") as en_file:
    for fi,en in zip(fi_file,en_file):
        fi=fi.strip()
        en=en.strip()
        if fi and en:
            fien.append((fi,en))

with open("trdata/enfi_target_all.txt") as fi_file, open("trdata/enfi_source_all.txt") as en_file:
    for fi,en in zip(fi_file,en_file):
        fi=fi.strip()
        en=en.strip()
        if fi and en:
            enfi.append((fi,en))

fien_set=set(fien)
enfi_set=set(enfi)
common=fien_set&enfi_set #keep only pairs which are shared
print("Len fien",len(fien_set))
print("Len enfi",len(enfi_set))
print("Len common",len(common))
print(list(common)[:300])

#Making sure all we found is in the top 100K - just crosschecking really
print(len(set(finnish_vocab)&set(fi for fi,en in common)))
print(len(set(english_vocab)&set(en for fi,en in common)))

#Making sure all words are there exactly once - no risk of mixing train and validation
print(len(set(fi for fi,en in common)))
print(len(set(en for fi,en in common)))
print("...all these four numbers should be the same")

import random
pairs=[(fi,en) for fi,en in common if fi!=en] #Only keep pairs where source does not equal target
print("Left with",len(pairs),"after removing identical pairs")
random.shuffle(pairs) #always, always make sure to shuffle!

print("Shuffled pairs",pairs[:20])

#Now we need to grab the vectors for the words in question
en_indices=[model_english.vocab[en].index for fi,en in pairs] #English
fi_indices=[model_finnish.vocab[fi].index for fi,en in pairs] #Finnish
print("Indices:",en_indices[:10],fi_indices[:10])
#...and the vectors are hidden in the models
print("English model.vectors shape:",model_english.vectors.shape)
print("Finnish model.vectors shape:",model_finnish.vectors.shape)
en_vectors=model_english.vectors[en_indices] #Selects the rows in just the correct order
fi_vectors=model_finnish.vectors[fi_indices] #Selects the rows in just the correct order
print("English selected vectors shape:",en_vectors.shape)
print("Finnish selected vectors shape:",fi_vectors.shape)




import tensorflow as tf
### Only needed for me, not to block the whole GPU, you don't need this stuff
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
### ---end of weird stuff

from keras.models import Model
from keras.layers import Input, Dense



inp=Input(shape=(en_vectors.shape[1],)) #input is 200-dim
outp=Dense(fi_vectors.shape[1])(inp) #Simple linear transformation of the input

model=Model(inputs=[inp], outputs=[outp])
model.summary()

model.compile(optimizer="adam",loss="mse")
hist=model.fit(en_vectors,fi_vectors,batch_size=100,verbose=1,epochs=30,validation_split=0.1)

val_en,val_fi,_=hist.validation_data #This we saw before - the validation data
predicted_fi=model.predict(val_en) #Transform the English vectors in the validation data
for en,fi,pred_fi in list(zip(val_en,val_fi,predicted_fi))[:30]:
    print(model_english.similar_by_vector(en,topn=1)) #This is the original English word
    print(model_finnish.similar_by_vector(fi,topn=1)) #This is the target Finnish word
    print(model_finnish.similar_by_vector(pred_fi,topn=5)) # Top five closest hits to the transformed vector
    print("\n")
    

def eval(src_model,tgt_model,src_vecs,tgt_vecs,predicted_vecs):
    top1,top5,top10,total=0,0,0,0
    for src_v,tgt_v,pred_v in zip(src_vecs,tgt_vecs,predicted_vecs):
        src_word=src_model.similar_by_vector(src_v)[0][0]
        tgt_word=tgt_model.similar_by_vector(tgt_v)[0][0]
        hits=list(w for w,sim in tgt_model.similar_by_vector(pred_v,topn=10))
        total+=1
        if tgt_word==hits[0]:
            top1+=1
        if tgt_word in hits[:5]:
            top5+=1
        if tgt_word in hits[:10]:
            top10+=1
    print("Top1",top1/total*100,"percent correct")
    print("Top5",top5/total*100,"percent correct")
    print("Top10",top10/total*100,"percent correct")
eval(model_english,model_finnish,val_en,val_fi,predicted_fi)

# Extra stuff - a function to query the translations, so we can play around
def top_n(word,source_model,target_model,transformation_model,topn=5):
    try:
        source_idx=source_model.vocab[word].index
    except:
        print("Cannot retrieve vector for",word)
        return None
    mapped=transformation_model.predict(source_model.vectors[source_idx,:].reshape(1,-1))
    return target_model.similar_by_vector(mapped[0])
    
seen_words=set(en for fi,en in common) #These words were seen during training or validation
while True:
    wrd=input("word> ")
    if wrd=="end":
        break
    if wrd in seen_words:
        print("    WARNING: this word was seen during training")
    hits=top_n(wrd,model_english,model_finnish,model)
    for word,sim in hits:
        print("  ",word,"  ",sim)
    print()


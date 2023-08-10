lines=["This is line 1\t POSITIVE",        "this is line 2\t POSITIVE",        "I am line 3\t NEGATIVE"]
data_x=[]
data_y=[]
for line in lines:
    sent, tag = line.split("\t")
    data_x.append(sent)
    data_y.append(tag)

print data_x
print data_y
    

from collections import defaultdict

def get_word_space(data_x):
    word_space=defaultdict(int)
    for sent in data_x:
        #lowercase all words
        sent=sent.lower()
        words=sent.split()
        for w in words:
            if w not in word_space:
                word_space[w]=len(word_space)
    return word_space

#---------------------
space =get_word_space(data_x)
for w in space:
    print w, space[w]

import numpy as np
# print x
# print space["am"]
#x[space["am"]]=1
#print x
sent="I like this"
def get_space(sent, space):
    vector= np.zeros(len(space))
    words=sent.lower().split()
    for w in words:
        if w in space:
            vector[space[w]]=1
    return vector

v= get_space("Let's all go home this", space)
print v

        
    



import numpy as np

sentences= ["Mapping the geographical diffusion of new words line","Challenges of studying and processing This dialects in sm"]

def get_space_vec(sent, space):
    vec= np.zeros(len(space))
    #print vec
    for w in sent.lower():
        if w in space:
            #print w, "--->" , space[w]
            vec[space[w]]= 1
    return vec

data_vecs=[]
for sent in sentences:
    #print sent
    sent=sent.split()
    vector= get_space_vec(sent, space)
    #print "New disney vector: ",
    #print vector
    data_vecs.append(vector)

print data_vecs





sent="Hey there, people"
x=sent.split()
print x




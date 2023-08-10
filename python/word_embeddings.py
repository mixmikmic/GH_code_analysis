from gensim.models import KeyedVectors

# load embedding model from a file
# binary: True if saved as binary file (.bin), False if saved as text file (.vectors or .txt for example)
# limit: How many words to read from the model
model=KeyedVectors.load_word2vec_format("data/gigaword-and-wikipedia.bin", binary=True, limit=100000)

# basic functions
print("Most similar words for 'locomotive':")
print(model.most_similar("locomotive",topn=10))
print()

print("Similarity of 'ship' and 'ferry':")
print(model.similarity("ship", "ferry"))
print("Similarity of 'ship' and 'sweetness':")
print(model.similarity("ship", "sweetness"))

print("Embedding matrix type:", type(model.vectors))
print("Embedding matrix shape:", model.vectors.shape)
print("First 2 vectors:", model.vectors[:2])
print()

print("Vocabulary type:", type(model.vocab))
print("Word locomotive in the vocabulary:", model.vocab["locomotive"])
print("Index for the word locomotive:", model.vocab["locomotive"].index)
print("Vector for the word locomotive:", model.vectors[model.vocab["locomotive"].index])


import numpy
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

cutoff=100000

#numpy.savetxt("vectors_giga_in.txt",model.vectors[:cutoff,:]) # save vectors for a file

# run tsne locally, input is an embedding matrix, output is an embedding matrix with reduced dimensions
#os.system("python3 bhtsne/bhtsne.py --input vectors_giga_in.txt --output vectors_giga_out.txt")

m2d=numpy.loadtxt("vectors_giga_out.txt") # read new vectors from a file

# plot
fig, ax = plt.subplots(figsize=(50, 50), dpi=250)
plt.scatter(m2d[:,0],m2d[:,1],marker=",",color="yellow")
words=[(k,w.index) for k,w in model.vocab.items()][:100000] # read words from the embedding model
words=[w for w,i in sorted(words, key=lambda x:x[1])] # sort based on the index to make sure the order is correct
print(words[:50])

for i,(v,w) in enumerate(zip(m2d,words)):
    if i%200!=0: # show only every 200th word
        continue
    txt=plt.text(v[0],v[1],w,size=30,ha="center",va="center",color="black")
    txt.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
                   path_effects.Normal()])
    

plt.show()


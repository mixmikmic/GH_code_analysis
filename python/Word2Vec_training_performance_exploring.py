import cython #ENSURE cython package is installed on computer/canopy
import numpy as np
from gensim.models import word2vec
from gensim.models import phrases 
from gensim import corpora, models, similarities #calc all similarities at once, from http://radimrehurek.com/gensim/tut3.html
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import csv
from statistics import mean
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import string
from string import digits
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#np.set_printoptions(threshold=np.inf) #set to print full output

dat= pd.read_csv('examiner-date-tokens.csv') #this file should be in your working directory - the same folder where this Jupyter Notebook is saved
dat #view what it looks like

dat2= [str(i[1]) for i in dat.values] #only need the text of each headline in the 1st index, not the date, in the 0th index
translator = str.maketrans(string.ascii_letters, string.ascii_letters, string.digits)

sentences=[]
for i in dat2:
    headline= i.translate(translator)
    headline= headline.strip()
    sentences.append(headline.split())
    
print(sentences[1])
print(len(sentences)) # check that it is actually around 3 million headlines here as supposed to be. its ok if not perfectly clean (i.e.  some words that are nonsensical) for these purposes

#does sentences look ok? then delete dat and dat2 to save up some space
del(dat2)
del(dat)

sentences= open('your_tokenized_data_file_here.txt').read() 

#your "sentences" object with the cleaned text data. 
bigram_transformer = phrases.Phrases(sentences) 
bigram= phrases.Phraser(bigram_transformer)

modelA_ALLYEARS= word2vec.Word2Vec(bigram[sentences], workers=4, sg=0,size=500, min_count=40,window=5, sample=1e-3)

modelB_ALLYEARS= word2vec.Word2Vec(bigram[sentences], workers=4, sg=1, size=500, min_count=40, window=10, sample=1e-3)

modelC_ALLYEARS= word2vec.Word2Vec(bigram[sentences],  workers=4, sg=0, hs=1,size=500, min_count=40, window=10, sample=1e-3)

modelD_ALLYEARS= word2vec.Word2Vec(bigram[sentences], workers=4, sg=1, hs=1, size=500, min_count=40, window=10, sample=1e-3)

modelA_ALLYEARS.init_sims(replace=True) #Precompute L2-normalized vectors. If replace is set to TRUE, forget the original vectors and only keep the normalized ones. Saves lots of memory, but can't continue to train the model.

modelA_ALLYEARS.save("modelA_ALLYEARS_500dim_10CW") #save your model for later use! change the name to something to remember the hyperparameters you trained it with

currentmodel=  Word2Vec.load("Word2VecModels/modelA_ALLYEARS_500dim_10CW") #name of YOUR model here, or file path to your model. 

currentmodel = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True) 

#Make sure you have downloaded three files into your working directory from this from the Github repo https://github.com/arsena-k/Word2Vec-bias-extraction: questions_words_pasted.txt, testing.py, and question-words.txt
accuracy=currentmodel.accuracy('questions_words_pasted.txt') 

accuracy_labels= ['world_capitals1', 'world_capitals2', 'money', 'US_capitals', 'family', 
                  'adj_to_adverbs', 'opposites', 'comparative', 'superlative','present_participle',
                 'nationality', 'past_tense', 'plural', 'plural_verbs', 'world_capitals3']

accuracy_tracker=[]
for i in range(0, len(accuracy)):
    sum_corr = len(accuracy[i]['correct'])
    sum_incorr = len(accuracy[i]['incorrect'])
    total = sum_corr + sum_incorr
    print("Accuracy on " + str(accuracy_labels[i]) + ": "  + str(float(sum_corr)/(total)))
    accuracy_tracker.append(float(sum_corr)/(total))

print('\033[1m' + "Average Accuracy: " + str(mean(accuracy_tracker)) + '\033[0m')

len(currentmodel.vocab) 

currentmodel['woman'] 

currentmodel.most_similar('woman', topn=10) 

print(1 - spatial.distance.cosine(currentmodel['woman'], currentmodel['man'])) 

currentmodel.most_similar(negative=['big']) 

print(currentmodel.wv.most_similar(positive=['woman', 'king'], negative=['man'])) #man:king as woman:_?___ QUEEN! 

print(currentmodel.wv.most_similar(positive=['girl', 'man'], negative=['boy'])) #boy:man as girl:_?___ WOMAN!

currentmodel.doesnt_match("noodle chicken turkey beef".split()) #does it get that noodle is not a meat?

currentmodel.doesnt_match("tomato potato toe".split()) #does it get that a toe is not a fruit or veggie?

print(currentmodel.wv.most_similar(positive=['woman'], negative=['man'], topn=10)) #what are the most feminine words?

print(currentmodel.wv.most_similar(positive=['man'], negative=['woman'], topn=10)) #what are the most masculine words?

my_word_list=[]
my_word_vectors=[]
label=[]

words_to_explore= ['woman', 'man', 'queen', 'king', 'human', 'person', 'girl', 'child', 'boy', 'salad', 'lettuce', 'tomatoe', 'soup', 'turnip', 'arugula', 'pepper', 'greens', 'barley', 'bean', 'stew', 'carrot']

for i in words_to_explore:   
    if my_word_list not in my_word_list:
        my_word_list.append(i)
        my_word_vectors.append(currentmodel.wv[i])

my_word_list=[]
my_word_vectors=[]
label=[]

for i in currentmodel.wv.most_similar(positive=['woman'], negative=['man'], topn=30):   #30 most feminine words
    if my_word_list not in my_word_list:
        my_word_list.append(i[0])
        my_word_vectors.append(currentmodel.wv[i[0]])
        label.append('fem')

for i in currentmodel.wv.most_similar(positive=['man'], negative=['woman'], topn=30):   #30 most masculine words
    if my_word_list not in my_word_list:
        my_word_list.append(i[0])
        my_word_vectors.append(currentmodel.wv[i[0]])
        label.append('masc')

#review of tsne: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
#modified code from https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne/code

def tsne_plot(words, vectors, iterations, seed, title): 
    "Creates and TSNE model and plots it"
    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=iterations, random_state=seed) #you may need to tune these, epsecially the perplexity. #Use PCA to reduce dimensionality to 2-D, an "X" and a "Y 
    new_values = tsne_model.fit_transform(vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(10, 10)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(words[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.ylabel("Latent Dimension 1") #Some pyplot reminders: https://matplotlib.org/users/pyplot_tutorial.html
    plt.xlabel("Latent Dimension 2")
    plt.title(title)
    plt.show()

tsne_plot(my_word_list, my_word_vectors, 3000, 23,  "TSNE Visualization of Word-Vectors") #don't use too many words, won't finish running




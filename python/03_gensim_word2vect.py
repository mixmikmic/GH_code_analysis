# Header
import gensim, logging
import os

# Access to the pretrained embedings
data_path = '/Users/jorge/data/training/text/'

# Access to the evaluation embedings file
module_path = '/Users/jorge/anaconda/envs/tm/lib/python3.5/site-packages/gensim/test'

# Example code to buils a word2vect embedings froma a corpus

# To show in the output the internal messages of the word2vect process
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
# My little corpus    
sentences = [['first', 'sentence'], ['second', 'sentence']]

# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)

# Load a more big corpus
from nltk.corpus import brown

print('Corpus sentences len:', len(brown.sents()))
print('Corpus words len:', len(brown.words()))

# Train a word2 vect model over the new corpus

from gensim.models import Word2Vec

model = Word2Vec(brown.sents(), size=100, window=5, min_count=5, workers=4)

#Persist the model

model.save('/tmp/brown_word2vect_model.bin')

# Load a trained model

model = Word2Vec.load('/tmp/brown_word2vect_model.bin')  # you can continue training with the loaded model!

# Access to the embedings

model.wv['the']  # Vector embeding of a word. Numpy array

# Model functionalities

print('Similars to woman:', model.wv.most_similar_cosmul(positive=['woman']), '\n')

print("Indetify the word that doesn't match in a list:", model.wv.doesnt_match("breakfast cereal dinner lunch".split()), '\n')

print('Words similarity (woman - man):', model.wv.similarity('woman', 'man'))

# Check the accuracy of the builded embedings over a standar evaluation list of relations

model.wv.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))

# If you finish to train the model. Save embedings and delete model.

word_vectors = model.wv

del model

# Explore the embedings

print('Num of embedings:', len(word_vectors.vocab.keys()), '\n')

print('Sample of words available (20 first):', list(word_vectors.vocab.keys())[:20], '\n')

print('Vocab word attributes for "Oregon" word:', word_vectors.vocab['Oregon'], '\n')

print('Word embedings for "Oregon" word:', word_vectors['Oregon'])

# Vocabulary frequency. List the words with freq > 1000

for k in word_vectors.vocab.keys():
    if word_vectors.vocab[k].count > 1000:
        print(k, word_vectors.vocab[k].count)



# Load pretrained embedings

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(os.path.join(data_path, 'lexvec.enwiki+newscrawl.300d.W.pos.vectors'),
                                          unicode_errors='ignore')

print('Sample of one embeding')
dog = model['dog']
print('Shape of one embeding:', dog.shape)
print('First 10 embedings of "dog":', dog[:10], '\n')


# Some predefined functions that show content related information for given words
print('woman + king - man = ', model.most_similar(positive=['woman', 'king'], negative=['man']), '\n')

print("Doesn't match:", model.doesnt_match("breakfast cereal dinner lunch".split()), '\n')

print('Similarity woman - man:', model.similarity('woman', 'man'))
    

# Test the accuracy of this model
model.wv.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))

# use in a model
import numpy as np

# Load data. Sentiment model in movies reviews.
# Reference: http://www.aclweb.org/anthology/P11-1015 

X_trn = np.load(os.path.join(data_path, 'sentiment', 'sentiment_X_trn.npy')) 
X_tst = np.load(os.path.join(data_path, 'sentiment', 'sentiment_X_tst.npy'))
y_trn = np.load(os.path.join(data_path, 'sentiment', 'sentiment_y_trn.npy')) # 1: pos, 0:neg
y_tst = np.load(os.path.join(data_path, 'sentiment', 'sentiment_y_tst.npy')) # 1: pos, 0:neg

print(X_trn[:2])
print(y_trn[:20])

# Represent each sentence by the average embeding of the words located in the embedings dictionary

def encode_text(corpus, model):
    '''
    Function to encode text sentences into one embbeding by sentence
        input: A list of sentences (corpus) and a embeddings model (model)
        output: One embeding for each sentence (average of embedings of the words in the sentence)
    '''
    features_list = []
    for s in corpus:
        features = []
        for t in s:
            if str.lower(t) in model.vocab.keys():
                features += [model[str.lower(t)]]
        features_list += [np.mean(features, axis=0)] 
    return np.array(features_list)

# Check embedings shape
embeds_trn = encode_text(X_trn, model)
print('Embeds trn shape:', embeds_trn.shape)

embeds_tst = encode_text(X_tst, model)
print('Embeds tst shape:', embeds_tst.shape)

# Build a model and evaluate it
from sklearn.svm import LinearSVC

# Train
text_clf_svm = LinearSVC()
text_clf_svm.fit(embeds_trn, y_trn)

#Evaluate test data
predicted = text_clf_svm.predict(embeds_tst)
print('Test accuracy:', np.mean(predicted == y_tst))




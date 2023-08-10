import numpy as np
from fasttext import FastVector

languages=['en','it']
language_extended=['english','french','italian']
"""
to use this, you will need: 
1) alignment matrices from https://github.com/Babylonpartners/fastText_multilingual - place in alignemnt_matrices/
2) Vectors from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md - place in vectors/
"""

matrix_dir='alignment_matrices/'
dic_dir='vectors/wiki.'
rawdir='../data_clean/'
infile='translations.tsv'

dictionary={}
filenames={}
words={}

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

def load_words():
    with open(infile,'rU') as f:
        for line in f:
            print line
            row=line[:-1].split('\t')
            words[row[0]]=row[1]

def load_dictionaries():
    for lan in languages:
        #load word vector dictionaries
        dictionary[lan]= FastVector(vector_file=dic_dir+lan+'.vec')
        #aligning all vectors to english
        if lan !='en':
            dictionary[lan].apply_transform(matrix_dir+lan+'.txt')

#first load variables and dictionaries
words={}
load_words()
#load_dictionaries()
print words

l=len(dictionary['en']['hi'])
translations={}
#for every language, generate aligned vectors for clean sentences and write to file
cw=0
cc=0
for w in words:
    out=dictionary['it'].translate_nearest_neighbour(dictionary['en'][w])
    translations[w]=out
    if words[w]==translations[w]:
        cc+=1
    cw+=1
print cc/float(cw)

print words




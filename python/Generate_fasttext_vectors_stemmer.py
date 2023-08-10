import numpy as np
from fasttext import FastVector
from nltk.stem import SnowballStemmer
import copy

languages=['it','fr','en']
language_extended=['italian','french','english']
"""
to use this, you will need: 
1) alignment matrices from https://github.com/Babylonpartners/fastText_multilingual - place in alignemnt_matrices/
2) Vectors from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md - place in vectors/
"""

matrix_dir='alignment_matrices/'
dic_dir='vectors/wiki.'
rawdir='../data_clean/'
feadir='features/stemmed_'

dictionary={}
filenames={}
outfiles={}

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

def load_filenames():
    for lan,lext in zip(languages,language_extended):
        #load clean data files
        filenames[lan]=rawdir+lext+'.tsv'
        #load output feature files
        outfiles[lan]=feadir+lan+'.tsv'

def load_dictionaries():
    for lan,lext in zip(languages,language_extended):
        substitutes={}
        #load word vector dictionaries
        dictionary[lan]= FastVector(vector_file=dic_dir+lan+'.vec')
        stemmer = SnowballStemmer(lext) # Choose a language
        for w in dictionary[lan].word2id:
            try:
                wo=stemmer.stem(w)
                substitutes[wo]=w
            except:
                pass
        for wo in substitutes:
            dictionary[lan].word2id[wo]=dictionary[lan].word2id[substitutes[wo]]       
        #aligning all vectors to engglish
        if lan !='en':
            dictionary[lan].apply_transform(matrix_dir+lan+'.txt')

#first load variables and dictionaries
load_filenames()
load_dictionaries()

l=len(dictionary['en']['hi'])
#for every language, generate aligned vectors for clean sentences and write to file
for lan,lext in zip(languages,language_extended):
    #open outfile for writing
    fo=open(outfiles[lan],'w')
    stemmer = SnowballStemmer(lext) # Choose a language
    cnt=0
    with open(filenames[lan]) as f:
        #for every sentence in the clean filename
        for line in f:
            #isolate the text
            row=line[:-1].split('\t')
            text=row[-2]
            #split into words
            words=text.split()
            #populate vector with sum of word vectors
            outvec=np.zeros(l)
            count=0
            for w in words:
                cnt=cnt+1
                try:
                    outvec+=dictionary[lan][w]
                    count=count+1
                except:
                    try:
                        wo=stemmer.stem(w)
                        try:
                            outvec+=dictionary[lan][wo]
                            count=count+1
                            print('here')
                        except:
                            pass
                    except:
                    #there is no matching word in the dictionary
                        pass
            #divide by the total number of matching vetors
            if count>0:
                outvec /=count
            outvec[outvec == np.nan] =0
            outvec[outvec == np.inf] = 0
            outvec[outvec == -np.inf] = 0
            #build a comma-separated string for the sentence vectors
            out=','.join([str(c) for c in outvec])
            #rebuild output string
            outstring='\t'.join(row[:-2])+'\t'+row[-1]+'\t'+out+'\n'
            #writes to file
            fo.write(outstring)
    fo.close()
            

print

print outfiles




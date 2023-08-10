#7000 wiki is the set of wikipedia articles crawled
#with the ck12 and spelling city.
#the text is not parsed.



import pandas as pd
import os
path = '/Users/MK/GitHub/the_answer_is/data'
os.chdir(path)
train = pd.read_table('training_set.tsv',sep = '\t')
train.head()

import os
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

#train the model using wikipdeida data                
sentences = MySentences('/Users/MK/GitHub/the_answer_is/data/wikipedia_from_all_ck_words') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences,size=300, window=5, min_count=5, workers=4)
#procedure needed for deleting words not in the training set. 
def only_the_words_in_index( list, index ):
    output = []
    for a, s in enumerate(list):
        if s in index:
            output.append(list[a])
    return output

import numpy as np 
model = model
myanswer_list = []
myanswer_distance = pd.DataFrame(np.zeros(4).reshape(1,4), columns = ['A','B','C','D'])    #initialize dataframe to store my answers distance


for i in train.index.values:
    length = len(train)
    q = train.ix[i][1].split()
    a1 = train.ix[i][3].split()
    a2 = train.ix[i][4].split()
    a3 = train.ix[i][5].split()
    a4 = train.ix[i][6].split()
    
    q =  only_the_words_in_index( q, model.index2word)
    a1 = only_the_words_in_index( a1, model.index2word)
    a2 = only_the_words_in_index( a2, model.index2word)
    a3 = only_the_words_in_index( a3, model.index2word)
    a4 = only_the_words_in_index( a4, model.index2word)

    try:
        answer_similarity = np.array([['A',model.n_similarity(q, a1)],['B',model.n_similarity(q, a2)],
                                      ['C',model.n_similarity(q, a3)],['D',model.n_similarity(q, a4)]])
    except:
        print 'Error on ', i, ' and set lengths at random'
        answer_similarity = np.array([['A',np.random.rand()],['B',np.random.rand()],
                                      ['C',np.random.rand()],['D',np.random.rand()]])
        
        myanswer_distance.set_value(i, 'A', answer_similarity[0,1] )    #write down distance for each choice
        myanswer_distance.set_value(i, 'B', answer_similarity[1,1] )
        myanswer_distance.set_value(i, 'C', answer_similarity[2,1] )
        myanswer_distance.set_value(i, 'D', answer_similarity[3,1] )
        
        myanswer_index = answer_similarity[:,1].argmax()         #get the maxmimum similarity 
        myanswer = answer_similarity[myanswer_index][0]
        myanswer_list.append(myanswer)
        continue
    
    myanswer_distance.set_value(i, 'A', answer_similarity[0,1] )    #write down distance for each choice
    myanswer_distance.set_value(i, 'B', answer_similarity[1,1] )
    myanswer_distance.set_value(i, 'C', answer_similarity[2,1] )
    myanswer_distance.set_value(i, 'D', answer_similarity[3,1] )
    
    myanswer_index = answer_similarity[:,1].argmax()          #get the max similarity
    myanswer = answer_similarity[myanswer_index][0]
    myanswer_list.append(myanswer)
    
    print 'progress: ', i, '/', length 
#for printing out the distance
# myanswer_distance.to_csv('/Users/MK/GitHub/the_answer_is/data/answer/pure_ck12_word2vec_distance.csv', encoding='utf-8')    
train['ck_12_word2vec_answer'] = myanswer_list
    

train

train['ck_12_word2vec_correct'] = (train['correctAnswer'] == train['ck_12_word2vec_answer'])

print 'percent correct is ' , train['ck_12_word2vec_correct'].sum(axis =0) / (len(train) + 0.0)

train.to_csv('/Users/MK/GitHub/the_answer_is/data/answer/pure_ck12_word2vec_7000wiki.csv', encoding='utf-8')

myanswer_distance




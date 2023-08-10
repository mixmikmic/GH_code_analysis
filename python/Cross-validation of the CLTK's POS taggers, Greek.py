from nltk.corpus.reader import TaggedCorpusReader
from nltk.tag import AffixTagger
from nltk.tag import BigramTagger
from nltk.tag import tnt
from nltk.tag import TrigramTagger
from nltk.tag import UnigramTagger
from nltk.tokenize import wordpunct_tokenize
import math
import os
import pandas as pd
import random
from statistics import mean
from statistics import stdev
import time

full_training_set_rel = '~/greek_treebank_perseus/greek_training_set.pos'
full_training_set = os.path.expanduser(full_training_set_rel)

# This section's code is good, but it times out in IPython.
# Consider using it in separate scripts.
unigram_accuracies = []
bigram_accuracies = []
trigram_accuracies = []
backoff_accuracies = []
tnt_accuracies = []

with open(full_training_set) as f:
    training_set_string = f.read()

pos_set = training_set_string.split('\n\n')  # mk into a list

sentence_count = len(pos_set)  # 24825 
tenth = math.ceil(int(sentence_count) / int(10))

random.shuffle(pos_set)

def chunks(l, n):
    """Yield successive n-sized chunks from l.
    http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

# a list of 10 lists
ten_parts = list(chunks(pos_set, tenth))  # a list of 10 lists with ~2483 sentences each

#for counter in list(range(10)):
for counter, part in list(enumerate(ten_parts)):
    
    start = time.time()
    
    # map test list to part of given loop
    test_set = ten_parts[counter]  # or: test_set = part
    
    # filter out this loop's test index
    training_set_lists = [x for x in ten_parts if x is not ten_parts[counter]]
    
    # next concatenate the list together into 1 file ( http://stackoverflow.com/a/952952 )
    training_set = [item for sublist in training_set_lists for item in sublist]
        
    # save shuffled tests to file
    # there might be a way of getting 
    local_dir_rel = '~/cltk_data/user_data'
    local_dir = os.path.expanduser(local_dir_rel)
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)

    test_path = os.path.join(local_dir, 'test_greek.pos')
    with open(test_path, 'w') as f:
        f.write('\n\n'.join(test_set))

    train_path = os.path.join(local_dir, 'train_greek.pos')
    with open(train_path, 'w') as f:
        f.write('\n\n'.join(training_set))

    # read POS corpora
    train_reader = TaggedCorpusReader(local_dir, 'train_greek.pos')
    train_sents = train_reader.tagged_sents()

    test_reader = TaggedCorpusReader(local_dir, 'test_greek.pos')
    test_sents = test_reader.tagged_sents()
    
    print('Loop #' + str(counter))
    # make unigram tagger
    unigram_tagger = UnigramTagger(train_sents)
    # evaluate unigram tagger
    unigram_accuracy = None
    unigram_accuracy = unigram_tagger.evaluate(test_sents)
    unigram_accuracies.append(unigram_accuracy)
    print('Unigram:', unigram_accuracy)
    
    # make bigram tagger
    bigram_tagger = BigramTagger(train_sents)
    # evaluate bigram tagger
    bigram_accuracy = None
    bigram_accuracy = bigram_tagger.evaluate(test_sents)
    bigram_accuracies.append(bigram_accuracy)
    print('Bigram:', bigram_accuracy)
    
    # make trigram tagger
    trigram_tagger = TrigramTagger(train_sents)
    # evaluate trigram tagger
    trigram_accuracy = None
    trigram_accuracy = trigram_tagger.evaluate(test_sents)
    trigram_accuracies.append(trigram_accuracy)
    print('Trigram:', trigram_accuracy)
    
    # make 1, 2, 3-gram backoff tagger
    tagger1 = UnigramTagger(train_sents)
    tagger2 = BigramTagger(train_sents, backoff=tagger1)
    tagger3 = TrigramTagger(train_sents, backoff=tagger2)
    # evaluate trigram tagger
    backoff_accuracy = None
    backoff_accuracy = tagger3.evaluate(test_sents)
    backoff_accuracies.append(backoff_accuracy)
    print('1, 2, 3-gram backoff:', backoff_accuracy)
    
    # make tnt tagger
    tnt_tagger = tnt.TnT(N=100)  # N=1000 is default but but won't finish with Greek
    tnt_tagger.train(train_sents)
    # evaulate tnt tagger
    tnt_accuracy = None
    tnt_accuracy = tnt_tagger.evaluate(test_sents)
    tnt_accuracies.append(tnt_accuracy)
    print('TnT:', tnt_accuracy)
    
    print('Runtime:', time.time() - start)
    print()

final_accuracies_list = []
mean_accuracy_unigram = mean(unigram_accuracies)
standard_deviation_unigram = stdev(unigram_accuracies)
uni = {'unigram': {'mean': mean_accuracy_unigram, 'sd': standard_deviation_unigram}}
final_accuracies_list.append(uni)

mean_accuracy_bigram = mean(bigram_accuracies)
standard_deviation_bigram = stdev(bigram_accuracies)
bi = {'bigram': {'mean': mean_accuracy_bigram, 'sd': standard_deviation_bigram}}
final_accuracies_list.append(bi)

mean_accuracy_trigram = mean(trigram_accuracies)
standard_deviation_trigram = stdev(trigram_accuracies)
tri = {'trigram': {'mean': mean_accuracy_trigram, 'sd': standard_deviation_trigram}}
final_accuracies_list.append(tri)

mean_accuracy_backoff = mean(backoff_accuracies)
standard_deviation_backoff = stdev(backoff_accuracies)
back = {'1, 2, 3-gram backoff': {'mean': mean_accuracy_backoff, 'sd': standard_deviation_backoff}}
final_accuracies_list.append(back)

mean_accuracy_tnt = mean(tnt_accuracies)
standard_deviation_tnt = stdev(tnt_accuracies)
tnt = {'tnt': {'mean': mean_accuracy_tnt, 'sd': standard_deviation_tnt}}
final_accuracies_list.append(tnt)

final_dict = {}
for x in final_accuracies_list:
    final_dict.update(x)

df = pd.DataFrame(final_dict)
df


import os
from srs.utilities import Sentence
from srs.maxEntropyModel import load_labelled_sent, loadWordListDict, train, cond_prob, loadUsefulTrainingData
from srs.predictor import StaticPredictor
import matplotlib.pyplot as plt
import numpy as np
import json
get_ipython().magic('matplotlib inline')

static_traning_data_dir = os.path.abspath('../srs/static_training_data/')

sentences = loadUsefulTrainingData(static_traning_data_dir)

# static aspect counting
static_label_dict = {}
for sentence in sentences:
    if sentence.labeled_aspects not in static_label_dict:
        static_label_dict[sentence.labeled_aspects] = 1
    else:
        static_label_dict[sentence.labeled_aspects] += 1

#sorting
static_label_tups_sorted = sorted(static_label_dict.items(), key=lambda tup: -tup[1])
static_label_sorted = [tup[0] for tup in static_label_tups_sorted]
static_count_sorted = [tup[1] for tup in static_label_tups_sorted]

ind = np.arange(len(static_label_sorted))
width = 0.6
heights = static_count_sorted

plt.figure(figsize=[10,6])
ax = plt.gca()
ax.bar(ind, heights, width, color='r')
ax.set_xticks(ind+width/2)
ticks = ax.set_xticklabels(static_label_sorted, rotation=90)
plt.ylabel(u'Number of Sentences')

# set plotting params
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    

# set up predictor
staticPredictor = StaticPredictor()
params_file = '../srs/predictor_data/lambda_opt_1.txt'
static_aspect_list_file = '../srs/predictor_data/static_aspect_list.txt'
staticPredictor.loadParams(params_file)
staticPredictor.loadStaticAspectList(static_aspect_list_file)

staticPredictor.wordlist_dict = loadWordListDict('../srs/predictor_data/wordlist_dict_1.txt')

validation_set = sentences[:]
correct = 0.0
correct_dict = {}
incorrect_dict = {}
expect_dict = {}
for sentence in validation_set:
    predicted_aspect = staticPredictor.predict(sentence, cp_threshold=0)
    if predicted_aspect == sentence.labeled_aspects:
        correct += 1
        if predicted_aspect not in correct_dict:
            correct_dict[predicted_aspect] = [sentence]
        else:
            correct_dict[predicted_aspect].append(sentence)
    else:
        if predicted_aspect not in incorrect_dict:
            incorrect_dict[predicted_aspect] = [sentence]
        else:
            incorrect_dict[predicted_aspect].append(sentence)
        
        if sentence.labeled_aspects not in expect_dict:
            expect_dict[sentence.labeled_aspects] = [sentence]
        else:
            expect_dict[sentence.labeled_aspects].append(sentence)
        

accuracy = correct/len(sentences)
print accuracy

print "Precision:"
print '-----------'
for aspect in correct_dict:
    print aspect
    if aspect in incorrect_dict:
        print len(correct_dict[aspect])*1.0/(len(correct_dict[aspect]) + len(incorrect_dict[aspect]))
    else:
        print 1.0

print "Recall:"
print "----------"
for aspect in correct_dict:
    print aspect
    print len(correct_dict[aspect])*1.0/(len(correct_dict[aspect]) + len(expect_dict[aspect]))

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
word_list = ['cheaper', 'expensive', 'affordable', 'color', 
             'pretty', 'auto', 'modes', 'smart', 'focus','carry',
            'smaller', 'pocketable', 'bulky', 'photo', 'photos']
for i in range(len(word_list)):
    word = word_list[i]
    stemmedWord = stemmer.stem(word)
    print stemmedWord

aspect = "battery"
print len(correct_dict[aspect]) + len(incorrect_dict[aspect])
for sent in incorrect_dict[aspect]:
    print sent.content




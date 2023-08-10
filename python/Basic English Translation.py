# IMPORTS
import pandas as pd
import cPickle as pickle
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import gensim as gensim
import string
import re
import nltk.data
import time

# Convert basic english words to a list
basic_english_df = pd.read_csv('data/basic_english_wordlist.csv')
basic_english = [a for a in basic_english_df['WORD']]
# add the various conjugations of 'to be' and 'a'
basic_english.append('an')
basic_english.append('is')
basic_english.append('was')
basic_english.append('are')
basic_english.append('were')
basic_english.append('they')
basic_english[350] = 'big' # 'I' causing weird issues...
basic_english.append('she')
basic_english.append('hers')
basic_english.append('his')
basic_english.append('my')
basic_english.append('him')
basic_english.append('her')
basic_english.append('your')
basic_english.append('their')
basic_english.append('might')
basic_english.append('must')
basic_english.append('can')
basic_english.append('did')
basic_english.append('could')
basic_english.append('should')
basic_english.append('would')
basic_english.append('that')
basic_english.append('what')
basic_english.append('we')
basic_english.append('small')
basic_english[basic_english.index('colour')] = 'color'

# adding contractions...
contractions_df = pd.read_csv('data/contractions.csv', sep=' -')
contractions = [word for word in contractions_df['from']]
contractions[18] = "mightn't"

start = time.clock()
Google_model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
print '{:.2f}s'.format(time.clock() - start)

vocab_google = Google_model.vocab.keys()
print len(vocab_google)

try:
    my_dict = pickle.load(open('data/basic_english.pickle', "rb" ))
except:
    st = LancasterStemmer()
    stem_gn = [st.stem(key) for key in Google_model.vocab.keys()]
    stem_se = [st.stem(word) for word in basic_english]
    print 'No saved dictionary...'
    my_dict = {}
    threshold = 0.25
    for sim_in in xrange(len(basic_english)-1, 0, -1):
        print
        print basic_english[sim_in]
        print '**'*8
        indices = [i for i, s in enumerate(stem_gn) if stem_se[sim_in] == s]
        check = [i for i, s in enumerate(vocab_google) if basic_english[sim_in] == s]
        #print check, indices
        if len(check) > 0:
            for index in indices: 
                if Google_model.similarity(basic_english[sim_in], vocab_google[index]) >= threshold:
                    print '{} -> {}'.format(vocab_google[index], Google_model.similarity(basic_english[sim_in], vocab_google[index]))
                    my_dict[vocab_google[index].lower()] = [vocab_google[index].lower(), basic_english[sim_in].lower()]
        my_dict[basic_english[sim_in].lower()] = [basic_english[sim_in].lower(), basic_english[sim_in].lower()]
        
    my_dict['i'] = ['i','i'] # add 'I
    basic_english.append('i')
    for word in basic_english:
        wordy = word
        if len(word) <= 1:
            wordy = word+"'"
        for con in contractions:
            if wordy.lower() in con.lower()[0:len(wordy)]:
                my_dict[con.lower()] = [con.lower(), word.lower()]
    my_dict["am"] = ['am','am']
    my_dict["a"] = ['a','a']
#     with open('data/basic_english.pickle', 'wb') as handle:
#          pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

hold_dict = my_dict.copy()

import time as time
import sys 
from nltk import pos_tag, word_tokenize
from math import sqrt


def make_simple_english(input_text, threshold=0, dictionary=hold_dict, save_bypass=False):
    '''
    Return the input_text translated into simple english
    Input: String
    Output: String
    '''
    temp_dict = {}
    temp_dict = hold_dict.copy()
    if threshold == 0:
        threshold = 60.0/sqrt(len(input_text))
    done = 0
    # timer...
    start= time.clock()
    input_text = input_text.replace('—',' - ').replace("’"," ' ")
    input_text = ''.join([a if ord(a) < 128 else '' for a in list(input_text)])
    words = pos_tag(word_tokenize(input_text)) # makes a list of words...

    # These simply pass thru the model
    pass_thru = ['CD', # CD: numeral, cardinal
                 'EX', # EX: existential there
                 'FW', # FW: foreign word
                 'LS', # LS: list item marker
                 'NNP', # NNP: noun, proper, singular
                 'NNPS', # NNPS: noun, proper, plural
                 'PRP', # PRP: pronoun, personal
                 'SYM', # SYM: symbol
                 'TO', # TO: "to" as preposition or infinitive marker
                 'POS',
                 '$', # $: dollar
                 '(',
                 ')',
                 ',',
                 '.',
                 ':',
                 '"'
                ] 
    # make these Basic
    make_simple = ['CC', # CC: conjunction, coordinating
                   'DT', # DT: determiner
                   'IN', # IN: preposition or conjunction, subordinating
                   'JJ', # JJ: adjective or numeral, ordinal
                   'JJR', # JJR: adjective, comparative
                   'JJS', # JJR: adjective, comparative
                   'MD', # MD: modal auxiliary
                   'NN', # NN: noun, common, singular or mass
                   'NNS', # NNS: noun, common, plural
                   'PDT', # PDT: pre-determiner
                   'PDT', # PDT: pre-determiner
                   'PRP$', # PRP$: pronoun, possessive
                   'RB', # RB: adverb
                   'RBR', # RBR: adverb, comparative
                   'RBS', # RBS: adverb, superlative
                   'RP', # RP: particle
                   'UH', # UH: interjection
                   'VB', # VB: verb, base form
                   'VBD', # VBD: verb, past tense
                   'VBG', # VBG: verb, present participle or gerund
                   'VBN', # VBN: verb, past participle
                   'VBP', # VBP: verb, present tense, not 3rd person singular
                   'VBZ', # VBZ: verb, present tense, 3rd person singular
                   'WDT', # WDT: WH-determiner
                   'WP', # WP: WH-pronoun
                   'WP$', # WP$: WH-pronoun, possessive
                   'WRB' #WRB: Wh-adverb
                  ]
    done == 0
    count_replacements = 0
    lst_ret = []
    for word in words:
        if word[1] in pass_thru:
            # put it in and move on... it's proper or whatever
            lst_ret.append(word[0])
        else:
            # We have a word we need to replace...
            clean = word[0].strip(string.punctuation).lower() # bath it...
            # ...and bring it to the function
            if clean in temp_dict.keys():  # already simple... throw it in and move on
                lst_ret.append(retain_capitalization(temp_dict[clean][0], word[0]))
            elif clean != '': # not alread simply/basic...
                start_this = time.clock() # timing for testing
                try: # in case it fails...
                    lst = list(set(Google_model.most_similar(clean)))
                    done = 0
                    n = 0
                    while done == 0:
                        check = list(lst)[n][0]
                        n +=1
                        check_clean = check.strip(string.punctuation).lower()
                        if check_clean in temp_dict.keys():
                            done = 1
                            # add to dictionary...based on what's there, retaining grouping info
                            temp_dict[clean] = [temp_dict[check_clean][0], check_clean]
                            if save_bypass:
                                my_dict[clean.lower()] = [temp_dict[check_clean][0].lower(), check_clean.lower()]
                            # add to lst
                            lst_ret.append(retain_capitalization(temp_dict[clean][0], word[0]))
                            print "     {}: {} -> {} ({}s) {}".format(word, clean, temp_dict[check_clean][0].lower(), time.clock()-start_this, n)
                        else:
                            # add all similar words to that to the lst
                            if time.clock() - start_this < threshold:
                                [lst.append(a) for a in Google_model.most_similar(check, topn=3) if a not in lst]
                            else: # timeout!
                                done = 1
                                temp_dict[clean] = [clean.lower(), clean.lower()]
                                lst_ret.append(retain_capitalization(temp_dict[clean][0], word[0]))
                                # print "     {}: {} -> {} ({}s) {}".format(word, clean.lower(),  temp_dict[clean][0], time.clock()-start_this, n)         
                                # timeouts = add if training off simple wikipedia
                                if save_bypass:
                                    my_dict[clean] = [clean.lower(), clean.lower()]
                except:
                    lst_ret.append(retain_capitalization(word[0], word[0]))
                    temp_dict[word[0].lower()] = word[0].lower()
                    # print "     >{}: {} [->] {} ({}s)".format(word, clean, word[0], time.clock()-start_this)

    end = time.clock()
    print 'Time: {:.2f}s'.format(end-start)
    txt = replace_punctuation(' '.join(lst_ret))
    txt = txt.encode('utf-8')
    txt = re.sub("\xe2\x80\x93", "-", txt)
    return txt


def retain_capitalization(new_word, original_word):
    '''
    Checks the original_word for capitalization, if it has it, capitalizes the frst letter
    of new_word, returns new_word.
    '''
    if original_word[0] in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        lst = list(new_word)
        lst[0] = lst[0].upper()
        new_word = ''.join(lst)
    return new_word


def replace_punctuation(text):
    '''
    Tokenizing takes the punctuation as it's own item in the list.
    This takes the created string and replaces all 'end ?' with 'end?'
    '''
    text = text.replace(' .','.')
    text = text.replace(' ?','?')
    text = text.replace(' !','!')
    text = text.replace(' ,',',')
    text = text.replace(' ;',';')
    text = text.replace(' "','"')
    text = text.replace(" '","'")
    text = text.replace('( ','(')
    text = text.replace(' )',')')
    text = text.replace('$ ','$')
    text = text.replace(' *','*')
    return text

from bs4 import BeautifulSoup
import requests


r = requests.get('https://simple.wikipedia.org/wiki/Horse')
soup = BeautifulSoup(r.content, 'html.parser')
a = 0
ret = ''


tags = soup.find_all('p')
MyText = '\n'.join([tag.get_text() for tag in tags])


# print tags[tags.index('title=')+6:tags.index('/>')]
#tags.span.clear()
print MyText
#print tags[tags.index('class')]

hold_dict = my_dict.copy()

# Define a function to split a book into parsed sentences
def book_to_sentences(input_text, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(input_text.encode("ascii","replace").strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( book_to_wordlist( raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def book_to_wordlist(book_text, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    #  
    #  Decode from UTF-8
#     tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
#               if unicodedata.category(unichr(i)).startswith('P'))
#     book_text = book.text.translate(tbl)

    #
    # 3. Convert words to lower case and split themstring.decode('utf-8')
    words = book_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return words



# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
features = vectorizer.fit_transform([MyText])

# Numpy arrays are easy to work with, so convert the result to an 
# array
features = features.toarray()
vocab = vectorizer.get_feature_names()

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Load in sentences
print 'loading existant set of sentences'
try:
    sentences = pickle.load(open('data/sentences.pickle', "rb" ))
except:
    print 'load failed'
    sentences = []  # Initialize an empty list of sentences
print "Parsing sentences from training set"

    
MyText = MyText.encode('ascii', 'replace')
sentences += book_to_sentences(MyText, tokenizer)

with open('data/sentences.pickle', 'wb') as handle:
     pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

start = time.clock()
# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 4   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec

model_name = 'all_parsed'+str(num_features)+'features_'+str(min_word_count)+'min_word_count_'+str(context)+'context.npy'

print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, 
                          size=num_features, min_count = min_word_count, 
                          window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
# model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()

model.save('data/'+model_name)
print '{:.2f}s'.format(time.clock() - start)
print len(model.wv.vocab.keys())

make_simple_english(MyText, threshold=10.0, dictionary=my_dict, save_bypass=True)

hold = hold_dict.keys()
my = my_dict.keys()
for key in my_dict.keys():
    if key not in hold and key != my_dict[key][0]:
        print "my_dict['{}'] = {}".format(key, my_dict[key])

my_dict['adaptations'] = ['adaptations', u'adapt']
my_dict['replace'] = ['replace', u'replace']
my_dict['fabric'] = ['cloth', u'cloth']
my_dict['loads'] = ['loads', u'loading']
my_dict['filly'] = ['horse', u'mare']
my_dict['gelding'] = ['horse', u'colt']
my_dict['mammals'] = ['mammals', u'mammal']
my_dict['tough'] = [u'tough', u'tough']
my_dict['grazers'] = [u'herbavores', u'herbavore']
my_dict['carrying'] = [u'carrying', u'carry']
my_dict['racehorses'] = [u'horses', u'horses']
my_dict['agriculture'] = [u'farming', u'farming']
my_dict['preferred'] = [u'preferred', u'prefer']
my_dict['racing'] = ['racing', u'races']
my_dict['horsehair'] = ['hair', u'hair']
my_dict['cooler'] = [u'colder', u'cold']
my_dict['colt'] = ['horse', u'mare']
del my_dict['dressage']
my_dict['foal'] = ['horse', u'stallion']
my_dict['heavy'] = ['heavy', u'heavier']
my_dict['equestrianism'] = [u'', u'eventing']
my_dict['forests'] = ['woods', u'woods']
my_dict['showjumping'] = [u'jumping', u'jump']
my_dict['pets'] = [u'pets', u'pet']
my_dict['stallion'] = ['horse', u'horse']
my_dict['gelatin'] = [u'gelatin', u'gelatin']
my_dict['miles'] = ['miles', u'mile']
my_dict['mare'] = ['horse', u'stallion']
my_dict['browsers'] = ['browsers', u'browser']
my_dict['crowds'] = ['crowds', u'crowd']
my_dict['plain'] = ['simple', u'simple']
my_dict['ecological'] = ['ecological', u'ecology']
my_dict['leaves'] = [u'leave', u'leave']
my_dict['plaster'] = ['plaster', u'plaster']
my_dict['equine'] = [u'horse-like', u'horses']
my_dict['grazer'] = [u'herbavore', u'herbavore']
my_dict['domesticated'] = [u'tamed', u'tame']

hold = hold_dict.keys()
my = my_dict.keys()
for key in my_dict.keys():
    if key not in hold:
        print "my_dict['{}'] = {}".format(key, my_dict[key])

my_dict['broadsword'] = ['sword', 'sword']
my_dict['uniforms'] = [u'uniforms', u'uniform']
my_dict['uniform'] = [u'uniform', u'uniform']
my_dict['slashing'] = ['slashing', 'slash']
my_dict['stabbing'] = ['stabbing', 'stab']
my_dict['wielded'] = ['held', 'hold']
my_dict['protecting'] = ['protecting', 'protect']
my_dict['cutting'] = ['cutting', 'cut']

with open('data/basic_english.pickle', 'wb') as handle:
     pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

my_dict['gods'] = ['gosd', u'god']


my_dict['talk into']


sentences

from bs4 import BeautifulSoup
import requests


r = requests.get('https://simple.wikipedia.org/wiki/Alexandre_Dumas,_p%C3%A8re')
soup = BeautifulSoup(r.content, 'html.parser')
a = 0
ret = ''


tags = soup.find_all('p')
MyText = '\n'.join([tag.get_text() for tag in tags])


# print tags[tags.index('title=')+6:tags.index('/>')]
#tags.span.clear()
print MyText
#print tags[tags.index('class')]

from bs4 import BeautifulSoup
import requests


r = requests.get('https://simple.wikipedia.org/wiki/Main_Page')
soup = BeautifulSoup(r.content, 'html.parser')
a = 0
ret = ''

links = soup.find_all('a')
save = []
for link in links:
    try:
        if '/wiki/' in link['href'] and link['title'] in link['href']:
            save.append(link['href'])
    except:
        print ''

print save

save

links =  [u'/wiki/Vocabulary',
 u'/wiki/Democracy',
 u'/wiki/Execution',
 u'/wiki/Architecture',
 u'/wiki/Communication',
 u'/wiki/Electronics',
 u'/wiki/Engineering',
 u'/wiki/Farming',
 u'/wiki/Health',
 u'/wiki/Industry',
 u'/wiki/Medicine',
 u'/wiki/Transport',
 u'/wiki/Weather',
 u'/wiki/Anthropology',
 u'/wiki/Archaeology',
 u'/wiki/Geography',
 u'/wiki/Education',
 u'/wiki/History',
 u'/wiki/Language',
 u'/wiki/Philosophy',
 u'/wiki/Psychology',
 u'/wiki/Sociology',
 u'/wiki/Teaching',
 u'/wiki/Animation',
 u'/wiki/Art',
 u'/wiki/Book',
 u'/wiki/Cooking',
 u'/wiki/Custom',
 u'/wiki/Culture',
 u'/wiki/Dance',
 u'/wiki/Family',
 u'/wiki/Game',
 u'/wiki/Gardening',
 u'/wiki/Leisure',
 u'/wiki/Movie',
 u'/wiki/Music',
 u'/wiki/Radio',
 u'/wiki/Sport',
 u'/wiki/Theatre',
 u'/wiki/Travel',
 u'/wiki/Television',
 u'/wiki/Algebra',
 u'/wiki/Astronomy',
 u'/wiki/Biology',
 u'/wiki/Chemistry',
 u'/wiki/Ecology',
 u'/wiki/Geometry',
 u'/wiki/Mathematics',
 u'/wiki/Physics',
 u'/wiki/Statistics',
 u'/wiki/Zoology',
 u'/wiki/Copyright',
 u'/wiki/Economics',
 u'/wiki/Government',
 u'/wiki/Law',
 u'/wiki/Military',
 u'/wiki/Politics',
 u'/wiki/Trade',
 u'/wiki/Atheism',
 u'/wiki/Buddhism',
 u'/wiki/Christianity',
 u'/wiki/Esotericism',
 u'/wiki/Hinduism',
 u'/wiki/Islam',
 u'/wiki/Jainism',
 u'/wiki/Judaism',
 u'/wiki/Mythology',
 u'/wiki/Paganism',
 u'/wiki/Sect',
 u'/wiki/Sikhism',
 u'/wiki/Taoism',
 u'/wiki/Theology',
 u'/wiki/Horse',
 u'/wiki/France',
 u'/wiki/French_Revolution',
 u'/wiki/Sword',
 u'/wiki/Gun',
 u'/wiki/War',
 u'/wiki/Horse',
 u'/wiki/Alexandre_Dumas,_père'
 ]

len(links)




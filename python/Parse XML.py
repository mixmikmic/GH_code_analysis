import logging

# create logger
alogger = logging.getLogger(__name__)
alogger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
alogger.addHandler(ch)

# initialization: 
alogger.setLevel(logging.WARNING) 



alogger.setLevel(logging.WARNING)
from subprocess import check_output
def graphs2phones(s): 
    """
        Graphemes to Phonemes: 
        Takes a sentence, returns an array of graphemes strings (one per number of words in original sentence)
        Example(s): 
        > graphs2phones('hello world bla and ble')
        > graphs2phones(' wasuuuppp!')
    """
    phs = check_output(["speak", "-q", "-x",'-v', 'en-us',s]).decode('utf-8')
    logger.debug("Return {} strings: {}".format(len(phs.split()), phs))
    return [w for w in phs.strip().split(" ") if w != ' ']

from timeit import default_timer as timer
def take_time(code_snippet_as_string):
    """
        Measures the time it takes to execute the code snippet
        provided as string. 
        Returns: the value of the invocation + number of seconds it took. 
        Example(s): 
        > r, secs = take_time("2 + 2")
        > print("result = {}, time = {} secs".format(r, secs))
    """
    start = timer()
    r = eval(code_snippet_as_string)
    end = timer()
    return (r, end - start)


def key_value(i, phonemes_strs_augm, words_in_sent):
    if len(i) == 1:
        try:
            return (phonemes_strs_augm[i[0]], words_in_sent[math.floor(i[0]/2)])
        except IndexError as e:
            logger.error("Trying to do 'phonemes_strs_augm[i[0]]' :: ")
            logger.error("i[0] = {}".format(i[0]))
            logger.error("phonemes_strs_augm = {}".format(phonemes_strs_augm))
            logger.error("Trying to do 'words_in_sent[math.floor(i[0]/2)]' :: ")
            logger.error("math.floor(i[0]/2) = {}".format(math.floor(i[0]/2)))
            logger.error("words_in_sent = {}".format(words_in_sent))
            raise e
    else:
        return (' '.join(phonemes_strs_augm[i[0]:i[1] + 1]), words_in_sent[int(i[1]/2 - 1)])

def group_contiguous(idxs_phonemes_strs_orig):
    """
        Given a list of indexes produces a list of tuples, where a tuple in a certain position
        has 2 elements iif the indexes are contiguous.
        Example: 
        > group_contiguous([1, 2, 3, 5, 6, 8, 9])
        [(1,2), (3), (5), (6), (8,9)]
        Notice that if more than 2 elements are contiguous, they will be split.
        (eg, in the example above, (1,2,3) was converted to (1,2), (3)) 
    """
    idxs_phonemes_strs = idxs_phonemes_strs_orig
    r = []
    while len(idxs_phonemes_strs) > 0:
        if len(idxs_phonemes_strs) > 1 and (idxs_phonemes_strs[0] + 1 == idxs_phonemes_strs[1]):
            r += [[idxs_phonemes_strs[0], idxs_phonemes_strs[1]]]
            idxs_phonemes_strs = idxs_phonemes_strs[2:]
        else:
            r += [[idxs_phonemes_strs[0]]]
            idxs_phonemes_strs = idxs_phonemes_strs[1:]
    return r
    
    
import functools
    
def graphemes_to_phonemes(words_in_sent):
    """
        Takes a list of words and returns a list of tuples
        (grapheme: phoneme)
        Example:
        > graphemes_to_phonemes(["luis", "papa"])
        [('luis', "lj'u:Iz"), ('papa', "pa#p'A:")]
    """
    MAX_LENGTH_TO_SPEAK = 10 # if I give more than this, espeak fails to do a good job 
    # First step: generate all sounds of words as if they were "alone" (ie, not in a sentence)
    # We want to avoid a combination of words making only 1 sound
    # For example (depending on accent): "what's up?"
    # So in order to do that we'll introduce a word with a unique sound between the words, 
    # generate phonemes and then process them smartly: 
    # separator for words in sentence 
    separator = {"str": "XXX"}
    separator["sound"] = ''.join(graphs2phones(separator["str"]))    
    # 
    how_many_words = len(words_in_sent)
    num_batches = (how_many_words // MAX_LENGTH_TO_SPEAK) + int(how_many_words % MAX_LENGTH_TO_SPEAK != 0)
    result_dict = [] # {}
    for i in range(num_batches):
        logger.debug("{}: {} to {}".format(i, i * MAX_LENGTH_TO_SPEAK, (i + 1)*MAX_LENGTH_TO_SPEAK))
        words_in_batch = words_in_sent[i * MAX_LENGTH_TO_SPEAK: (i + 1)*MAX_LENGTH_TO_SPEAK]
        logger.debug("words_in_batch = {}".format(words_in_batch))
        sent_augm = ' '.join([w1 + ' ' + w2 for w1, w2 in list(zip([separator["str"]]*len(words_in_batch), words_in_batch))]) + " " + separator["str"]
        logger.debug("sent_augm = {}".format(sent_augm))
        phonemes_strs_augm = graphs2phones(sent_augm)
        logger.debug("phonemes_strs_augm = {}".format(phonemes_strs_augm))
        # there we go: all (indexes of) sounds that we are interested in. 
        seps_idxs = [i for i,v in enumerate(phonemes_strs_augm) if v.endswith(separator["sound"])]
        logger.debug("seps_idxs = {}".format(seps_idxs))
        how_many_separators = len(seps_idxs)
        logger.debug("how_many_separators = {}".format(how_many_separators))

        all_sounds = list(map(
            lambda t: ' '.join(phonemes_strs_augm[t[0] + 1: t[1]]),
            list(zip(seps_idxs[:-1], seps_idxs[1:]))))
        logger.debug("all sounds = {}".format(all_sounds))
        result_dict += list(zip(words_in_batch, all_sounds))
    return result_dict
      
    
def dict_phonemes_to_graphemes(words_in_sent) -> dict:
    as_phon_graph_list = graphemes_to_phonemes(words_in_sent)
    return {ph: graph for (graph, ph) in as_phon_graph_list}

graphemes_to_phonemes(words_in_sent = "this is one sentence".split())



import re

# A custom function to clean the text before sending it into the vectorizer
def clean_text(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ").replace("&#039;", "'").replace("&quot;", "\"")
    
    # replace twitter @mentions
#     mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
#     text = mentionFinder.sub("@MENTION", text)
    
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<").replace("<br>", " ").replace("<BR>", " ")
    
#     # lowercase
#     text = text.lower()

    return text
    

print(clean_text('HELLO<br> world'))
print(clean_text("I've been better"))
print(clean_text("I&#039;ve been better"))
print(clean_text("I&#039;ve been &quot;better&quot;"))

import xmltodict
from typing import List, Dict
class Formspring_Data_Parser():
    
    def __init__(self, path_to_xml):
        with open(path_to_xml) as fd:
            self.path = path_to_xml
            self.doc = xmltodict.parse(fd.read())

    def posts_for_id(self, an_id: int) -> List:
        r = self.doc['dataset']['FORMSPRINGID'][an_id]['POST']
        # is this a list or a dict?
        try:
            r[0]  # wil this throw an Exception?
            return r
        except:
            return list([r])

    def questions_answers_labels(self, an_id: int)->List[Dict]:
        some_posts = self.posts_for_id(an_id)
        r = []
        for post in some_posts:
            q_and_a_orig = post['TEXT']
            logger.debug("q and a orig = '{}'".format(q_and_a_orig))
            q_and_a = clean_text(post['TEXT'])
            logger.debug("q and a = '{}'".format(q_and_a))
            # parse question 
            beg_q = q_and_a.index("Q:") + 2
            end_q = q_and_a.index("A:")
            the_q_raw = q_and_a[beg_q:end_q].strip()
            graphs_and_phons = graphemes_to_phonemes(words_in_sent = the_q_raw.split())
            the_q_as_sounds = ' '.join([ph for _, ph in graphs_and_phons])
            the_q = []
            for s in the_q_as_sounds.split():
                try:
                    the_q.append(mw.sound_to_word(s))
                except:
                    logger.debug("sound '{}' is not recognized".format(s))
                    the_q.append('UNKNOWN')
            the_q = ' '.join(the_q)
            # parse answer 
            the_a = q_and_a[end_q + 2:].strip()
            logger.debug("QUESTION = '{}', ANSWER = '{}'".format(the_q, the_a))
            raw_labels = [lab['ANSWER'] for lab in post['LABELDATA']]
            logger.debug("raw labels = {}".format(raw_labels))
            labels = list(map(lambda txt: txt.lower(), [lab for lab in raw_labels if lab is not None]))
            all_votes_as_yes = [l for l in labels if l == 'yes']
            is_threat = (len(all_votes_as_yes) / len(labels)) >= 0.5
            r.append({
                "txt_orig": q_and_a_orig,
                "question_raw": the_q_raw,
                "question": the_q,
                "question_as_sounds": the_q_as_sounds, 
                "answer_raw": the_a,
                "answer": the_a,
                "answer_as_sounds": the_a, 
                "labels": labels,
                "threat": is_threat
            })
#             r.append({
#                 "orig": {
#                     "txt_orig": q_and_a_orig,
#                     "question_raw": the_q_raw,
#                     "answer_raw": the_a,
#                 }, 
#                 "question": {
#                     "question": the_q,
#                     "question_as_sounds": the_q_as_sounds, 
#                 }, 
#                 "answer": {
#                     "answer": the_a,
#                     "answer_as_sounds": the_a, 
#                 },
#                 "sentiment": {
#                     "labels": labels,
#                     "threat": is_threat
#                 }
#             })
        return r


xml_file_name = '/Users/luisd/Downloads/FormspringLabeledForCyberbullying/XMLMergedFile.xml'

parser = Formspring_Data_Parser(xml_file_name)


# alogger.setLevel(logging.DEBUG) # verbose 
alogger.setLevel(logging.WARNING) # silent 
all_of_them = parser.questions_answers_labels(an_id = 1)
all_of_them

common_words = "hello world how are you today? My oncle is rich"
graphs_phons = graphemes_to_phonemes(common_words.split())

# let's "for" it to have a detailed report on failures: 
common_words_back = []
for _, sound in graphs_phons:
    try:
        common_words_back.append(mw.sound_to_word(sound))
    except Exception as e:
        print(e)
        common_words_back.append('UNKNOWN')
print("common words = {}".format(common_words))
print("graphs_phons = {}".format(graphs_phons))
print("common_words_back = {}".format(common_words_back))

mw.sound_to_word("r'ItS")

check_output(["speak", "-q", "-x",'-v', 'en-us',"RICHE"]).decode('utf-8')

all_sounds = mw.sounds_dict.keys()

print("We have {} words in vocabulary, {} sounds".format(len(mw.model.vocab), len(all_sounds))) 







[e for e in all_of_them if e["threat"]]

all_of_them[0]

import gensim 
import bisect 
import numpy as np
from typing import List, Dict
import shelve

data_dir = "./data"

class ModelWrapper():
        
    default_shelf_filename = '{}/shelf_whole_google_news.shelf'.format(data_dir)
        
    def __init__(self, m, sounds_dict = None):
        if m is None:
            print("Loading model...")
            self.model = gensim.models.word2vec.KeyedVectors.load_word2vec_format('{}/GoogleNews-vectors-negative300.bin.gz'.format(data_dir), binary=True)
            print("Model succesfully loaded")
        else:
            print("[init] Model provided. If you want me to FORCE re-load it, call ModelWrapper's constructor with 'None'")
            self.model = m            
        # sort all the words in the model, so that we can auto-complete queries quickly
        print("Sort all the words in the model, so that we can auto-complete queries quickly...")
        self.orig_words = [gensim.utils.to_unicode(word) for word in self.model.index2word]
        indices = [i for i, _ in sorted(enumerate(self.orig_words), key=lambda item: item[1].lower())]
        self.all_words = [self.orig_words[i].lower() for i in indices]  # lowercased, sorted as lowercased
        self.orig_words = [self.orig_words[i] for i in indices]  # original letter casing, but sorted as if lowercased
        
        # sounds dictionary 
        if sounds_dict is None:
            print("Loading default sounds dictionary from '{}'...".format(self.default_shelf_filename))
            self.sounds_dict = shelve.open(self.default_shelf_filename, flag='r')  
            print("Sounds dictionary succesfully loaded")
        else:
            self.sounds_dict = sounds_dict
    
        
        
        
    def suggest(self, term):
        """
        For a given prefix, return 10 words that exist in the model start start with that prefix
        """
        prefix = gensim.utils.to_unicode(term).strip().lower()
        count = 10
        pos = bisect.bisect_left(self.all_words, prefix)
        result = self.orig_words[pos: pos + count]
        logger.info("suggested %r: %s" % (prefix, result))
        return result      
    
    def most_similar(self, positive, negative):
        """
            positive: an array of positive words
            negative: an array of negative words 
        """                
        try:
            result = self.model.most_similar(
                positive=[word.strip() for word in positive if word],
                negative=[word.strip() for word in negative if word],
                topn=5)
        except:
            result = []
        logger.info("similars for %s vs. %s: %s" % (positive, negative, result))
        return {'similars': result}    
    
    def vec_repr(self, word):
        """
            If 'word' belongs in the vocabulary, returns its 
            word2vec representation. Otherwise returns a vector of 0's
            of the same length of the other words. 
        """
        try:
            return self.model.word_vec(word)
        except KeyError:
            logger.debug("'{}' not in Model. Returning [0]'s vector.".format(word))
            return np.zeros(self.model.vector_size)
        
    def sound_to_word(self, a_sound: str) -> str:
        return self.sounds_dict[a_sound]
    # self.sound_repr(a_sound)["word"]

    def sound_to_vec(self, a_sound: str) -> str:
        return self.vec_repr(self.sound_to_word(a_sound))

    def sound_repr(self, a_sound: str) -> Dict:
        # w = self.sounds_dict[a_sound]
        return {'word': self.sound_to_word(a_sound), 'vec': self.sound_to_vec(a_sound)}  
    

mw = ModelWrapper(model)
model = mw.model # just cache in case I re-call this cell





from joblib import Parallel, delayed
import multiprocessing
    
# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10) 
def processInput(i):
    return i * i

num_cores = multiprocessing.cpu_count()
print("You have {} cores".format(num_cores))

inputs2 = list(range(1000)) 
orig_inputs2 = inputs2
def processInput2(i):
    diff_l = [a_i - b_i for a_i, b_i in zip(inputs2, orig_inputs2)]
    nonnon = [i for i, e in enumerate(diff_l) if e != 0]
    if len(nonnon) > 0:
        print("[i = {}] indexes that changed = {}".format(i, nonnon)) 
    r = inputs2[i]
    inputs2[i] = -1 
    return r

# results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
# results 
results = Parallel(n_jobs=num_cores)(delayed(processInput2)(i) for i,_ in enumerate(inputs2))
# print("results = {}".format(results)) 
# print("inputs2 = {}".format(inputs2)) 

inputs2 = list(range(3)) 
inputs2 - inputs2

[3, 4, 5] - [6, 7, 8]

inputs2[3] = -44

inputs2




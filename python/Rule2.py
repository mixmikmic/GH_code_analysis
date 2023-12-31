import collections
import math
import nltk
import pprint as pp
import re
import spacy
import sys
import time

from google_ngram_downloader import readline_google_store
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from nltk.wsd import lesk

# Some of this code was directly copied from Rule #5
# We use it to minimize the words we need to check for because it's a computationally heavy task
def google_most_common_words(n_most_common=10000):
    google_most_common_words_path = sys.path[1] + '/../Texts/google-10000-english-usa.txt'
    most_common_words = []
    with open(google_most_common_words_path, 'r') as f:
        for i in range(n_most_common):
            most_common_words.append(f.readline().strip())
    return most_common_words

def is_non_replaceable_word(word):
    return (word.isalpha() is False) or (word in most_common_words)

# Global variables
syllable_dict = cmudict.dict()
most_common_words = google_most_common_words()

# This number comes from Google's blog
# https://research.googleblog.com/2006/08/all-our-n-gram-are-belong-to-you.html
# TODO: If there's time, confirm this number
NGRAM_TOKEN_COUNT = 1024908267229

# Shout out to Quora for this snippet of code
# https://www.quora.com/Is-there-any-Google-Ngram-API-for-Python
def find_google_ngrams_word_count(word, time_function=False, verbose=False):
    if time_function == True:
        time1 = time.time()

    count = 2 # Set this to a minimum of 2 so we don't get a divide by zero error
    # TODO: Consider how we want to deal with capitalization
    fname, url, records = next(readline_google_store(ngram_len=1, indices=word[0]))
    # If we use the verbose settings, occaisionally print out the record
    verbosity_count = 1000000000
    earliest_year = 1950
    i = 0
    try:
        record = next(records)
        while record.ngram != word:
            record = next(records)
            if verbose == True and i%verbosity_count == 0:
                print(record)
            i += 1
        while record.ngram == word:
            if record.year >= earliest_year:
                count += record.match_count
                if verbose == True:
                    print(record)
            record = next(records)
    except StopIteration:
        pass
    # Default to 1 so our program doesn't crash
    if count == 0:
        count = 1
    if time_function == True:
        time2 = time.time()
    print('Total seconds for ' + word + ': ' + str(int((time2-time1))))
    return count

def find_frequency_score(word):
    unigram_count = find_google_ngrams_word_count(word, time_function=True)
    percent_occurrence = unigram_count/NGRAM_TOKEN_COUNT
    # Get the log of the frequency to make our number manageable
    freq_val = math.log(percent_occurrence)
    max_ngram_val = math.log(1/NGRAM_TOKEN_COUNT)
    relative_freq = ((freq_val - max_ngram_val)/(-max_ngram_val))
    return round(relative_freq, 5)

BIG_NUMBER = 18109831

def syllable_count(word):
    syllable_count = 0
    for word in word.split():
        if word in syllable_dict:
            # Shout out to StackOverflow for this snippet of code
            # http://stackoverflow.com/a/4103234/1031615
            syllable_count += [len(list(y for y in x if y[-1].isdigit())) for x in syllable_dict[word]][0]
            continue
        # If it's not in the dictionary count the number of vowels and ignore an e at the end not
        # preceded by another vowel. It's rough, but there will be few cases if any cases in which
        # a word is not in the CMU dictionary but in WordNet
        if word[-1] == 'e':
            word = word[:-1]
        word = re.sub(r'[^aeiou]', '', word)
        syllable_count += len(word)
    return max(syllable_count, 1)


def readability_for_word(word, ignore_common_words=False, use_ngrams=False):
    if word is None:
        return BIG_NUMBER 
    word = word.lower()
    # If it's in the top 10000 most common words, we assume it is readable enough
    if ignore_common_words is True and is_non_replaceable_word(word) is True:
        return 0
    syllables = syllable_count(word)
    if use_ngrams == False:
        return syllables
    freq_score = find_frequency_score(word)
    return syllables * freq_score

def synsets_for_tokens_in_tokenized_sentence(tokenized_sentence):
    sentence = [token.text for token in tokenized_sentence]
    synsets = [lesk(sentence, token.text, spacy_to_wordnet_pos(token.pos_)) for token in tokenized_sentence]
    for i in range(len(synsets)):
        # Get the lemmas of the word. Ignore if there is only one because it's just the root of the word
        if synsets[i] is not None and len(synsets[i].lemma_names()) > 1:
            synsets[i] = synsets[i].lemma_names()[1:]
        else:
            synsets[i] = None
    return synsets


def spacy_to_wordnet_pos(pos):
    # To see all the parts of speech spaCy uses, see the link below
    # http://polyglot.readthedocs.io/en/latest/POS.html
    if pos == 'ADJ':
        return wn.ADJ
    elif pos == 'ADV':
        return wn.ADV
    elif pos == 'NOUN':
        return wn.NOUN
    elif pos == 'VERB':
        return wn.VERB
    return None


# Returns an array of tuples. If the word cannot be replaced, the second value is the replacing word.
# If it cannot be replaced, it is None
def replaceable_word_in_tokenized_sentence(tokenized_sentence):
    sentence_words = [token.text for token in tokenized_sentence]
    sentence_alternatives = synsets_for_tokens_in_tokenized_sentence(tokenized_sentence)
    for i in range(len(sentence_alternatives)):
        alternatives_list = sentence_alternatives[i]
        if alternatives_list is not None:
            alternatives_list = [(readability_for_word(alt), alt) for alt in alternatives_list]
            sentence_alternatives[i] = min(alternatives_list)[1]
    # Get the minimum syllables among the alternatives
    
    words_and_alternatives = zip(tokenized_sentence, sentence_alternatives)
    replaceable_words = []
    
    
    for (token, alt) in words_and_alternatives:
        if readability_for_word(token.text, ignore_common_words=True) <= readability_for_word(alt):
            replaceable_words.append((token, None))
        else:
            replaceable_words.append((token, alt))
    return replaceable_words

def print_replaceable_words_marked_in_document(document, open_marker='{', close_marker='}'):
    checked_sentences = [replaceable_word_in_tokenized_sentence(sentence) for sentence in document.sents]
    new_document_text = ''
    for sent_array in checked_sentences:
        for word in sent_array:
            new_document_text += word[0].text_with_ws
            if word[1] is not None:
                new_document_text += open_marker + word[1].upper() + close_marker + ' '
    return new_document_text
                                         
def load_doc(filepath):
    # Open and read the file
    with open(filepath) as f:
        text = f.read()
    nlp = spacy.load('en')
    doc = nlp(text)
    return doc

def test_with_gladwell():
    file_path = sys.path[1] + '/../Rule3/gladwell_latebloomers.txt'
    gladwell_doc = load_doc(file_path)
    replaceable_words = print_replaceable_words_marked_in_document(gladwell_doc)
    print(replaceable_words)

def rule2_ranges_in_text(text, nlp=None):
    if nlp == None:
        nlp = spacy.load('en')
    document = nlp(text)
    checked_sentences = [replaceable_word_in_tokenized_sentence(sentence) for sentence in document.sents]
    checked_words = [word for sentence in checked_sentences for word in sentence]
    ranges = []
    character_count = 0
    for i in range(len(checked_words)):
        word, alt = checked_words[i]
        if alt is not None:
            ranges.append((character_count, len(word)))
        character_count += len(word.text_with_ws)
    return ranges

def marked_up_doc(document_str):
    replaceable_ranges = rule2_ranges_in_text(document_str)
    open_tag_indices = [(index, '<rule2>') for (index, length) in replaceable_ranges]
    close_tag_indices = [(index + length, '</rule2>') for (index, length) in replaceable_ranges]
    tag_dictionary = collections.defaultdict(list)
    for (index, tag) in  open_tag_indices + close_tag_indices:
        tag_dictionary[index].append(tag)

    new_document = ''
    character_count = 0
    for i in range(len(document_str)):
        index_tags = tag_dictionary[i]
        if len(index_tags) == 0:
            new_document += document_str[i]
            continue
        for tag in index_tags:
            new_document += tag
        new_document += document_str[i]
    return new_document

def is_closed_tag(tag):
    return tag[1] == '/'


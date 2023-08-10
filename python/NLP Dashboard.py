import nltk
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

stemmer = SnowballStemmer('english') # Load the SnoballStemmer
default_stopwords = set(stopwords.words('english')) # default stopwords of English Language
custom_stopwords = ['naga'] #Custom Stopword list usecase specific, ensure that you enter data in lower case
final_stopwords = default_stopwords.union(custom_stopwords) # Final Stopword list combination of default and custom

def filter_stopwords(words):
    filtered_words=[]
    for word in words:
        if word not in final_stopwords:
            filtered_words.append(word)
    return filtered_words

def stem_words(words):
    stemmed_words=[]
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
    
def tokenize(sentence):
    sentence=sentence.translate(str.maketrans('','',string.punctuation))
    tokens=nltk.word_tokenize(sentence)
    filtered_words=filter_stopwords(tokens)
    stemmed_words=stem_words(filtered_words)
    return stemmed_words
    

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(data, custom_tokenizer=tokenize, verbose=False):
    count_vectorizer = CountVectorizer(tokenizer=custom_tokenizer)
    print(count_vectorizer.fit_transform(data).todense())
    if True==verbose:
        print('\nFeature Extraction Summary ')
        print('###########################')
        print('Vocabulary: ')
        print(count_vectorizer.get_feature_names())
        print('Feature Vector Dimesion: ',len(count_vectorizer.get_feature_names()))
    return count_vectorizer,ct

from sklearn.feature_extraction.text import CountVectorizer



test_set=[]
test_set.append('Hello')
test_set.append('hi naga how are you man!!, What is your favourite number? Is it #9?')
ct,countv=extract_features(test_set)
print(ct.tr)


import findspark
findspark.init()
import pyspark

import re
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from itertools import groupby
import numpy as np
import math

sc = pyspark.SparkContext(appName="test")

filePath = 'hdfs://0.0.0.0:9000/user/bitnami/group_project_data/data_simple.xml'

raw_data = sc.textFile(filePath)

id_pattern = re.compile('<row Id=\"([\d]*)\"')
content_pattern = re.compile('Text=\"([\W\w]*)\"')
noise_pattern = re.compile('&[#]*[\w]*;')


def job_filter(input_str) :
    return id_pattern.search(input_str) and content_pattern.search(input_str)

def job_extract(input_str):
    postid = id_pattern.search(input_str).group(1)
    content = content_pattern.search(input_str).group(1)
    content = noise_pattern.sub('', content)
    return postid, content

def job_cleanup_format(input_tuple):
    postid, content = input_tuple
    return int(postid), content.strip()

records_with_content = raw_data.filter(job_filter)
raw_id_content = records_with_content.map(job_extract)
cleaned_id_content = raw_id_content.map(job_cleanup_format)

# see what happened now
cleaned_id_content.take(4)

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# to lower case, no punctuation, stemmed, filter stop words
# this python version is highly depend on nltk, may consider sth else later
def job_split_content(input_tuple):
    postid, input_str = input_tuple
    input_str = input_str.lower()
    raw_tokens = tokenizer.tokenize(input_str)
    stemmed_tokens = [stemmer.stem(token) for token in raw_tokens]
    stemmed_tokens = map(stemmer.stem, raw_tokens)
    stemmed_tokens_without_stopword = filter(lambda i: i not in stop_words, stemmed_tokens)
    return postid, list(stemmed_tokens_without_stopword)

# this step seems to use up a lot of time!
id_tokens =  cleaned_id_content.map(job_split_content)

# check first two records
id_tokens.take(2)

class CorpusWordsSet(pyspark.AccumulatorParam):
    def zero(self, value=set()):
        return set()
    
    def addInPlace(self, acc1, acc2):
        return acc1 | acc2;

corpus_words = sc.accumulator(set(), CorpusWordsSet())

def job_add_tokens_to_dict(records):
    _, tokens = records
    corpus_words.add( set(tokens))
    
document_count_rdd = id_tokens.map(job_add_tokens_to_dict)
DOCUMENT_COUNT = document_count_rdd.count() # force accumulator to run
    
corpus_words = list(corpus_words.value)
WORD_COUNT = len(corpus_words)

WORD_COUNT_broadcasted = sc.broadcast(WORD_COUNT)
corpus_words_broadcasted = sc.broadcast(corpus_words)

def job_word_to_index(input_tuple):
    postid, tokens = input_tuple
    content_indexed = [corpus_words_broadcasted.value.index(token) for token in tokens] # [index1, index2, index1, index1, index3], for example
    content_freq = dict()
    for key, grouped in groupby(content_indexed):
        content_freq[key] = len(list(grouped))
    indexs = set(content_freq.keys())
    result_list = list()
    for i in range(WORD_COUNT_broadcasted.value):
        if i in indexs:
            result_list.append(content_freq.get(i))
        else:
            result_list.append(0)
        
    return postid, np.array(result_list)

id_freq_dicts = id_tokens.map(job_word_to_index)
id_freq_dicts.take(2)

id_freq_dicts.cache()

idf_array = list()
def cal_run_time():
    # calculate this function's run time...
    for word in corpus_words:
        word_index = corpus_words.index(word)

        documents_frequency = id_freq_dicts.map(lambda i: i[1][word_index] > 0).filter(lambda i: i is True).count()
        word_idf = np.log(np.divide(DOCUMENT_COUNT, documents_frequency + 1))
        idf_array.append(word_idf)

get_ipython().run_line_magic('time', 'cal_run_time()')

def job_cal_document_freq(word_freq_array1, word_freq_array2):
    document_freq1 = word_freq_array1!=0
    document_freq2 = word_freq_array2!=0
    return document_freq1*1 + document_freq2

get_ipython().run_line_magic('time', 'document_freq = id_freq_dicts.map(lambda i:i[1]).reduce(job_cal_document_freq)')

idf_array = np.log(DOCUMENT_COUNT/document_freq)

idf_array

id_tfidf = id_freq_dicts.map(lambda i: (i[0], i[1] * idf_array))

id_tfidf.take(2)




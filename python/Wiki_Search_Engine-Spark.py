from operator import add
from pyspark import SparkContext, SparkConf

conf = SparkConf()        .setAppName("Wiki_Search_Engine")        .setMaster("local[*]")        .set("spark.driver.memory", "10g")        .set("spark.driver.maxResultSize", "4g")

sc = SparkContext(conf=conf)
rawData = sc.textFile(r"wiki\artile_per_line.txt")
flatten_words = rawData.flatMap(lambda x: x.split('\t')[1].split())
words_joint = flatten_words.map(lambda x: (x, 1)).reduceByKey(add)
print "Number of distinct words in the whole Wikipedia: ", words_joint.count()
# Number of distinct words in the whole Wikipedia:  26764007

import re
from nltk.corpus import stopwords
from spacy.en import English
import nltk
import spacy.attrs
import time
import codecs

fo = open(r'wiki\titles.txt')
raw_text = fo.read()
fo.close()

wiki_title_words = set(raw_text.split())
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
stop_words = set(stopwords.words("english"))

NLP = English()

fo = codecs.open(r'wiki\artile_per_line.txt', 'r', encoding='utf-8') # title \t document
out = codecs.open(r"Wiki\article_lemma.txt", 'w', encoding='utf-8')


for doc in NLP.pipe(fo, n_threads=4):    
    title_words = []
    passed_title = False
    for candidate in doc:
        if '\t' not in candidate.lemma_: # title \t document
            if passed_title:
                if (candidate.lemma_ not in stop_words) and (candidate.pos_ != u'PUNCT') and                     (candidate.lemma_ in english_vocab or candidate.lemma_ in wiki_title_words):

                    out.write(candidate.lemma_ + u' ')
            else:
                title_words.append(candidate.orth_)
        else:
            out.write(u''.join(title_words) + candidate.orth_)
            passed_title = True

    out.write('\n')

fo.close()
out.close()

from operator import add
from pyspark import SparkContext, SparkConf

conf = SparkConf()        .setAppName("Wiki_Search_Engine")        .setMaster("local[*]")        .set("spark.driver.memory", "10g")        .set("spark.driver.maxResultSize", "4g")

sc = SparkContext(conf=conf)
rawData = sc.textFile(r"wiki\article_lemma.txt")
flatten_words = rawData.flatMap(lambda x: x.split('\t')[1].split())
words_joint = flatten_words.map(lambda x: (x, 1)).reduceByKey(add)
print "Number of distinct lemmas in the whole Wikipedia: ", words_joint.count()
# Number of distinct lemmas in the whole Wikipedia:  561354

sc = SparkContext(conf=conf)
rawData = sc.textFile(r"wiki\articles_lemma.txt")
documents = rawData.map(lambda line : line.split('\t')[1].split())
titles = rawData.map(lambda line : line.split('\t')[0])
titles.cache()

hashingTF = HashingTF(20000000)  #20 Million hash buckets just to make sure it fits in memory
tf = hashingTF.transform(documents)
idf = IDF(minDocFreq=10).fit(tf)
tfidf = idf.transform(tf)
tfidf.cache()

QueryTF = hashingTF.transform(["tabriz"])
QueryHashValue = QueryTF.indices[0]
QueryRelevance = tfidf.map(lambda x: x[QueryHashValue])
zippedResults = QueryRelevance.zip(titles)
print "Top 10 related documents:"
for (k, v) in zippedResults.sortByKey(ascending=False).take(10):
    print v


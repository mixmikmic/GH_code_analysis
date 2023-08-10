def fun(x):
    return x**2

def add(x,y):
    return x+y

def maximum(x,y):
    return max(x,y)

def minimum(x,y):
    return min(x,y)

def prod(x,y):
    return x*y

def selective_division(x,y):
    return x/y if x>y else y/x 

L = [1,3,4,2,7]

s = 0
for a in L:
    s = s+a**2
print(s)


S = map(fun, L)
print(S)

selective_division(65, 89)

#print(L)
print(S)

print reduce(selective_division, S)

print reduce(add, map(fun, L))


print reduce(minimum, S)
#print reduce(maximum, S)

def is_even(x):
    return 1 if x%2==0 else 0

reduce(add, map(is_even, L))

A = ['Ahmet', 'Mehmet', 'veli'] 

def is_A(x):
    return 1 if x[0]=='A' else 0

reduce(lambda x,y: x+y, map(is_A, A))

reduce(add, A)

import sys
import numpy as np

#textFile = sc.textFile("data/books-eng/hamlet.txt")
textFile = sc.textFile("data/books-eng")
textFile.count()

textFile.first()

word = "CLAUDIUS"
textFile.filter(lambda lin: word in lin).count()

textFile.sample(withReplacement=False, fraction=0.05).first()

def prnt(x):
    print x

textFile.sample(withReplacement=False, fraction=0.05).take(10)

textFile.map(lambda line: len(line.split())).reduce(lambda a, b: a if (a > b) else b)

textFile.map(lambda line: (len(line.split()),line)).reduce(lambda a, b: a if (a[0] > b[0]) else b)

'abc df'.split()

wordCounts = textFile.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
wordCounts.collect()

textFile.cache()

words = textFile.flatMap(lambda line: line.split())
words_subset = words.filter(lambda x: x[0] in ['H','h'])
counts = words_subset.map(lambda w: (w,1)).reduceByKey(lambda a,b: a+b)
counts.collect()

import re

# Compile a regular expression that matches non-alphanumerics
pattern = re.compile('[\W_]+', re.UNICODE)

# Replace all non-alphanumerics with a space, then split into words
words = textFile.map(lambda line: pattern.sub(' ',line)).flatMap(lambda line: line.split())

#first_letters = set(['H','h','Q','q','s','S'])
first_letters = set(['A','a'])
# Count words that start with H
words_subset = words.filter(lambda x: x[0] in first_letters)
counts = words_subset.map(lambda w: (w,1)).reduceByKey(lambda a,b: a+b)
res = counts.collect()
for r,c in res:
    print r,c

import re

#textFile = sc.textFile("notes/data/books-eng/hamlet.txt")
textFile = sc.textFile("data/books-eng")

# Compile a regular expression that matches non-alphanumerics
pattern = re.compile(u'[\W0-9_]+', re.UNICODE)

# Replace all non-alphanumerics with a space, then split into words
words = textFile.map(lambda line: pattern.sub(' ',line)).flatMap(lambda line: line.split())
# Convert to lower case
words = words.map(lambda w: w.lower())

# Convert to a list of letters list('abc') = ['a','b','c']
letters = words.flatMap(lambda word: [pair[0]+pair[1] for pair in zip(list('_'+word),list(word+'_')) ] )
counts = letters.map(lambda w: (w,1)).reduceByKey(lambda a,b: a+b)
bigrams = counts.collect()

for r,c in sorted(bigrams,key=lambda x:x[1],reverse=True):
    print r,c
    
    

bigrams[0][1]

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt

# Reduction table
my2ascii_table = {
    ord(u'â'):"a",
    ord(u'ä'):"e",
    ord(u"à"):"a",
    ord(u"æ"):"a",
    ord(u'ç'):"c",
    ord(u"é"):"e",
    ord(u"è"):"e",
    ord(u"ê"):"e",
    ord(u"ë"):"e",
    ord(u'ğ'):"g",
    ord(u'ı'):"i",
    ord(u"î"):"i",
    ord(u'ï'):"i",
    ord(u'œ'):"o",
    ord(u"ô"):"o",
    ord(u'ö'):"o",
    ord(u'ş'):"s",
    ord(u'ù'):"u",
    ord(u"û"):"u",
    ord(u'ü'):"u",
    ord(u'ß'):"s"
    }


def letter2idx(x):
    if x=='_':
        i = 0
    else:
        i = ord(x)-ord('a')+1
    
    if i<0 or i>26:
        i = ord(my2ascii_table[ord(x)])-ord('a')+1
        
    return i

T = np.zeros((27,27))
# Convert bigrams to a transition matrix
for pair in bigrams:
    c = pair[1]
    s = list(pair[0])
    j = letter2idx(s[0])
    i = letter2idx(s[1])
    T[i,j] += c

plt.figure(figsize=(8,8))

alphabet=[chr(i+ord('a')) for i in range(26) ]
alphabet.insert(0,'_')
M = len(alphabet)

plt.imshow(T/np.sum(T,axis=0), interpolation='nearest', vmin=0,cmap='gray_r')
plt.xticks(range(M), alphabet)
plt.xlabel('x(t)')
plt.yticks(range(M), alphabet)
plt.ylabel('x(t-1)')
ax = plt.gca()
ax.xaxis.tick_top()
#ax.set_title(f, va='bottom')
plt.xlabel('x(t)')

plt.show()


import numpy as np
import pyspark

def sample(p):
    x, y = 2*np.random.rand()-1, 2*np.random.rand()-1
    return 1 if x*x + y*y < 1 else 0

NUM_SAMPLES = 1000000

count = sc.parallelize(xrange(0, NUM_SAMPLES)).map(sample).reduce(lambda a, b: a + b)

print("Pi is roughly %f" % (4.0 * count / NUM_SAMPLES))

from pyspark.sql import SparkSession

#    .config("spark.some.config.option", "some-value") \

spark = SparkSession     .builder     .appName("Python Spark SQL basic example")     .getOrCreate()

df = spark.read.json("data/products.json")
# Displays the content of the DataFrame to stdout
df.show()

#df.printSchema()
df.select("properties").show()


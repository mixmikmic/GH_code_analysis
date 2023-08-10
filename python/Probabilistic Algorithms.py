import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from ggplot import *

def event(counter):
    trials = np.random.binomial(counter, 0.5, 1)
    if (np.sum(trials) == counter):
        return True
    return False

def counting_RMorris():
    df = pd.DataFrame(data=np.nan, index=range(0,10000), columns=['index', 'counter'])
    counter = 0
    for i in range(0, 10000):
        if event(counter):
            counter = counter + 1
        df.loc[i] = [i + 1, counter]
    return df

runs = pd.DataFrame()
for r in range(0,10):
    df = counting_RMorris()
    df['run'] = str(r + 1)
    runs = runs.append(df)

ggplot(aes(x='index', y='counter', group='run', color='run'), data=runs) + scale_x_log(2) + geom_step()     + xlab('Events') + ylab('Expoent') + theme_bw()

import hashlib
import zlib

def hash_CRC32(s):
    return zlib.crc32(s) & 0xffffff

def hash_Adler32(s):
    return zlib.adler32(s) & 0xffffff

def hash_MD5(s):
    return int(hashlib.md5(s).hexdigest(), 16) & 0xffffff

def hash_SHA(s):
    return int(hashlib.sha1(s).hexdigest(), 16) & 0xffffff

hash_functions = [hash_CRC32, hash_Adler32, hash_MD5, hash_SHA]

import nltk
words = nltk.corpus.gutenberg.words('austen-persuasion.txt')
words = [x.lower().encode('utf-8') for x in words]
print(len(words))
words[1:8]

# note that this implementation is returning 2 to power of the first 1-bit
def least1(x, L):
    if x == 0:
        return 2**L
    return x & -x

subset = words[0:1000]
bitmap = 0
for w in subset:
    h = hash_CRC32(w)
    bitmap |= least1(h, 24)
print(bin(bitmap))

def cardinality_FM(bitmap):
    return least1(~bitmap, 24) / 0.77351

print(cardinality_FM(bitmap))

from collections import Counter
print(len(Counter(subset)))

subset = words[0:6000]
df = pd.DataFrame(data=np.nan, index=range(0, 5 * len(subset)), columns=['f', 'x', 'count'])
bitmaps = np.array([0] * 4)
s = set([])

for idx, w in enumerate(subset):
    s.add(w)
    for i, hash_function in enumerate(hash_functions):
        bitmaps[i] |= least1(hash_function(w), 24)
    
    df.loc[idx * 5] = ['True Counting', idx, len(s)]
    df.loc[idx * 5 + 1] = ['CRC32 Hash', idx, cardinality_FM(bitmaps[0])]
    df.loc[idx * 5 + 2] = ['Adler32 Hash', idx, cardinality_FM(bitmaps[1])]
    df.loc[idx * 5 + 3] = ['MD5 Hash', idx, cardinality_FM(bitmaps[2])]
    df.loc[idx * 5 + 4] = ['SHA Hash', idx, cardinality_FM(bitmaps[3])]

ggplot(aes(x='x', y='count', group='f', color='f'), data=df) + geom_step() + xlab('Words Processed') + ylab('Cardinality') + theme_bw()

def index_least1(x):
    if x == 0:
        return 0
    index = 1
    while x % 2 == 0:
        x >>= 1
        index += 1
    return index

def cardinality_LogLog(buckets):
    buckets = [index_least1(x) for x in buckets]
    return 0.39701 * len(buckets) * 2 ** (np.mean(buckets))

subset = words[0:6000]
df = pd.DataFrame(data=np.nan, index=range(0, 4 * len(subset)), columns=['f', 'x', 'count'])

s = set([])
bitmap = 0
buckets16 = np.array([0] * 16)
buckets64 = np.array([0] * 64)

for idx, w in enumerate(subset):
    s.add(w)
    hashed = hash_SHA(w)
    
    bitmap |= least1(hashed, 24)
    buckets16[hashed % 16] = max(buckets16[hashed % 16], least1(hashed >> 4, 24))
    buckets64[hashed % 64] = max(buckets64[hashed % 64], least1(hashed >> 6, 24))
    
    df.loc[idx * 4] = ['True Counting', idx, len(s)]
    df.loc[idx * 4 + 1] = ['SHA Hash', idx, cardinality_FM(bitmap)]
    df.loc[idx * 4 + 2] = ['LogLog (16 buckets)', idx, cardinality_LogLog(buckets16)]
    df.loc[idx * 4 + 3] = ['LogLog (64 buckets)', idx, cardinality_LogLog(buckets64)]

ggplot(aes(x='x', y='count', group='f', color='f'), data=df) + geom_step() + xlab('Words Processed') + ylab('Cardinality') + theme_bw()

def cardinality_HyperLogLog(buckets):
    buckets = [1 if bucket == 0 else 1 / (bucket << 1) for bucket in buckets]
    return 0.72134 * len(buckets)**2 / np.sum(buckets)

subset = words[0:15000]
df = pd.DataFrame(data=np.nan, index=range(0, 4 * len(subset)), columns=['f', 'x', 'count'])

s = set([])
bitmap = 0
buckets = np.array([0] * 64)

for idx, w in enumerate(subset):
    s.add(w)
    hashed = hash_SHA(w)
    
    bitmap |= least1(hashed, 24)
    buckets[hashed % 64] = max(buckets[hashed % 64], least1(hashed >> 6, 24))
    
    df.loc[idx * 4] = ['True Counting', idx, len(s)]
    df.loc[idx * 4 + 1] = ['SHA Hash', idx, cardinality_FM(bitmap)]
    df.loc[idx * 4 + 2] = ['LogLog (64 buckets)', idx, cardinality_LogLog(buckets)]
    df.loc[idx * 4 + 3] = ['HyperLogLog (64 buckets)', idx, cardinality_HyperLogLog(buckets)]

ggplot(aes(x='x', y='count', group='f', color='f'), data=df) + geom_step() + xlab('Words Processed') + ylab('Cardinality') + theme_bw()

train = words[0:1000] # ~400 unique elements
test = words[1000:15000]

s = set([])
for w in train:
    s.add(w)

N_sizes = [3000, 5000, 7000]
df = pd.DataFrame(data=np.nan, index=range(0, len(test) * len(N_sizes)), columns=['x', 'error', 'size'])
for idx, N in enumerate(N_sizes):
    bitmap = np.array([0] * N)
    for w in train:
        for hash_function in hash_functions:
            bitmap[hash_function(w) % N] = 1
    
    size = "N=" + str(N) + " (ratio: " + str(round(np.mean(bitmap), 3)) + ")"
    
    error = 0
    for i, w in enumerate(test):
        check = True
        for hash_function in hash_functions:
            if bitmap[hash_function(w) % N] == 0:
                check = False
                
        if check == True and w not in s:
            error += 1
            
        df.loc[idx * len(test) + i] = [i, error, size]

ggplot(aes(x='x', y='error', group='size', color='size'), data=df) + geom_step()     + xlab('Words Processed') + ylab('False Positives') + theme_bw()

N = 300
bitmap = np.array([0] * N)
subset = words[0:7000] # ~1500 unique words

for w in subset:
    for hash_function in hash_functions:
        bitmap[hash_function(w) % N] += 1

true_freq = nltk.FreqDist(subset).most_common(25)[5:]

df = pd.DataFrame(data=0, index=(0, 2 * (len(true_freq) - 1)), columns=['word', 'frequency', 'f'])
for idx, (w, true) in enumerate(true_freq):
    freq = -1
    for hash_function in hash_functions:
        cnt = bitmap[hash_function(w) % N]
        if freq == -1 or cnt < freq:
            freq = cnt
            
    df.loc[idx * 2] = [w, freq, 'Counting Bloom Filter']
    df.loc[idx * 2 + 1] = [w, true, 'True Frequency']

from plotnine import *
get_ipython().magic('matplotlib inline')
(ggplot(df, aes(x='word', y='frequency', fill='f')) + geom_bar(stat='identity', position='dodge') + theme_bw())


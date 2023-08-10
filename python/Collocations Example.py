import nltk;
import string;
import math;
import csv;

# read text file
text_path = "data/crime-and-punishment.txt";
with open(text_path) as f:
    text_raw = f.read().lower();

# remove punctuation
translate_table = dict((ord(char), None) for char in string.punctuation);
text_raw = text_raw.translate(translate_table);

# tokenize
tokens = nltk.word_tokenize(text_raw);
bigrams = nltk.bigrams(tokens);

# unigram/bigram frequencies
unigram_counts = nltk.FreqDist(tokens);
bigram_counts = nltk.FreqDist(bigrams);

# write to file
unigram_path = text_path + ".unigrams";
bigram_path = text_path + ".bigrams";

with open(unigram_path, "w") as f:
    writer = csv.writer(f);
    filtered = [ (w,c) for w,c in unigram_counts.items() if c > 1];
    writer.writerows(filtered);
    
with open(bigram_path, "w") as f:
    writer = csv.writer(f);
    filtered = [ (b[0], b[1],c) for b,c in bigram_counts.items() if c > 3];
    writer.writerows(filtered);

unigram_counts.most_common(20)

bigram_counts.most_common(20)

# compute pmi
pmi_bigrams = [];

for bigram,_ in bigram_counts.most_common(1000):
    w1, w2 = bigram;
    
    # compute pmi
    actual = bigram_counts[bigram];
    expected = unigram_counts[w1] * unigram_counts[w2];
    pmi = math.log( actual / expected );
    
    pmi_bigrams.append( (w1, w2, pmi) );

# sort pmi
pmi_sorted = sorted(pmi_bigrams, key=lambda x: x[2], reverse=True);

pmi_sorted[:30]

pmi_sorted[-30:]

unigram_path = "data/crime-and-punishment.txt.unigrams";
bigram_path = "data/crime-and-punishment.txt.bigrams";

with open(unigram_path) as f:
    reader = csv.reader(f);
    unigrams = { row[0] : int(row[1]) for row in csv.reader(f)}
    
with open(bigram_path) as f:
    reader = csv.reader(f);
    bigrams = { (row[0],row[1]) : int(row[2]) for row in csv.reader(f)}

bigrams


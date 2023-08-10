import json

# Read in bigram (letter-pair) counts
with open('bigrams.json') as bigram_file:
    bigrams = dict(json.load(bigram_file))

# Generate unigram (single-letter) counts from bigrams
unigrams = dict()
for k in bigrams:
    first_letter = k[0]
    if first_letter not in unigrams:
        unigrams[first_letter] = 0
    unigrams[first_letter] += bigrams[k]
    
unigrams

# Estimate P(X_0)
total_counts = sum([unigrams[c] for c in unigrams])
p_first_letter = {c: float(unigrams[c])/total_counts for c in unigrams}
p_first_letter

# Estimate P(X_t | X_t-1)
p_next_letter = {k: float(bigrams[k])/unigrams[k[0]] for k in bigrams}
print p_next_letter['aa']
print p_next_letter['ab']
print p_next_letter['ac']
print p_next_letter['ad']

# Compute P(x_1, x_2, ..., x_n) given P(X_1), P(X_t | X_t-1), and a particular x
from math import log
def word_log_prob(w):
    # Make a list of all bigrams appearing in the word
    w_bigrams = [w[i:(i+2)] for i in xrange(len(w)-1)]
    # Add up the total log probability for the first letter, and then each letter given the previous
    return log(p_first_letter[w[0]]) + sum([log(p_next_letter[b]) for b in w_bigrams])

print "log P('hi') = ",word_log_prob('hi')
print "log P('hello') = ",word_log_prob('hello')
print "log P('goodbye') = ",word_log_prob('goodbye')
print "log P('thththt') = ",word_log_prob('thththt')
print "log P('ooooooo') = ",word_log_prob('ooooooo')
print "log P('covfefe') = ",word_log_prob('covfefe')

def word_ok(w):
    if len(w) == 0:
        return False
    
    for i in xrange(len(w)-1):
        b = w[i:(i+2)]
        if b not in p_next_letter or p_next_letter[b] <= 0.0:
            return False
    return True
   
with open('english.json') as english_words:
    all_words = json.load(english_words)
filtered_words = list(filter(word_ok, all_words))
filtered_words[0:10]

for w in sorted(filtered_words, key=word_log_prob)[0:20]:
    print word_log_prob(w), w

for w in sorted(filtered_words, key=lambda k: word_log_prob(k)/len(k))[0:20]:
    print word_log_prob(w)/len(w), w

for w in sorted(filtered_words, key=word_log_prob, reverse=True)[0:20]:
    print word_log_prob(w), w

for w in sorted(filtered_words, key=lambda k: word_log_prob(k)/len(k), reverse=True)[0:20]:
    print word_log_prob(w)/len(w), w

seven_letter_words = list(filter(lambda w: len(w) == 7, filtered_words)) + ['covfefe']
words_by_prob = sorted(seven_letter_words, key=lambda w: word_log_prob(w))
rank = words_by_prob.index('covfefe')
print rank, "out of ", len(seven_letter_words)

words_by_prob[(rank-10):(rank+11)]


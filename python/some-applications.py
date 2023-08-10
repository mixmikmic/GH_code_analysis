from annoy import AnnoyIndex
import numpy as np

import spacy

nlp = spacy.load('en_core_web_md')

t = AnnoyIndex(50, metric="euclidean")
words = list()
lookup = dict()
for i, line in enumerate(open("cmudict-0.7b-simvecs", encoding="latin1")):
    word, vals_raw = line.split("  ")
    word = word.lower().strip("(012)")
    vals = np.array([float(x) for x in vals_raw.split(" ")])
    t.add_item(i, vals)
    words.append(word.lower())
    lookup[word.lower()] = i
t.build(100)

from collections import Counter
def nnslookup(t, nlp, words, vec, n=10):
    res = t.get_nns_by_vector(vec, n)
    batches = []
    current_batch = []
    last_vec = None
    for item in res:
        if last_vec is None or t.get_item_vector(item) == last_vec:
            current_batch.append(item)
            last_vec = t.get_item_vector(item)
        else:
            batches.append(current_batch[:])
            current_batch = []
            last_vec = None
    if len(current_batch) > 0:
        batches.append(current_batch[:])
    output = []
    for batch in batches:
        output.append(sorted(batch, key=lambda x: nlp.vocab[words[x]].prob, reverse=True)[0])
    return output
[words[i] for i in nnslookup(t, nlp, words, t.get_item_vector(lookup["roads"]))]

sentences = [
    "I am sitting in a room different from the one you are in now",
    "Double double toil and trouble fire burn and cauldron bubble",
    "Peter Piper picked a peck of pickled peppers",
    "Four score and seven years ago our fathers brought forth on this continent a new nation"
]
for s in sentences:
    vecs = np.array([t.get_item_vector(lookup[w.lower()]) for w in s.split()])
    mean = vecs.mean(axis=0)
    print(s, "\n\t→", ', '.join([words[i] for i in nnslookup(t, nlp, words, mean, 10)]))

def progress(src_vecs, op_vecs, n=10):
    for i in range(n+1):
        delta = i * (1.0/n)
        val = (src_vecs * (1.0-delta)) + (op_vecs * delta)
        yield val

s = "I am sitting in a room different from the one you are in now"
vecs = np.array([t.get_item_vector(lookup[w.lower()]) for w in s.split()])
print(s)
op_vecs = np.array([t.get_item_vector(lookup["humming"])]*len(vecs))
for res in progress(vecs, op_vecs, n=25):
    print(" ".join([words[nnslookup(t, nlp, words, i)[0]] for i in res]))

s = "Perhaps the best way to describe us is as a Center for the Recently Possible"
vecs = np.array([t.get_item_vector(lookup[w.lower()]) for w in s.split()])
print(s)
op_vecs = np.array([t.get_item_vector(lookup["error"])]*len(vecs))
for res in progress(vecs, op_vecs, n=25):
    print(" ".join([words[nnslookup(t, nlp, words, i)[0]] for i in res]))

import textwrap
alpha = "abcdefghijklmnopqrstuvwxyz"
last = ""
output = []
for a, b in zip(alpha[:-1], alpha[1:]):
    if a != last:
        output.append(a)
    last = a
    for res in progress(np.array([t.get_item_vector(lookup[a])]), np.array([t.get_item_vector(lookup[b])]), n=30):
        res = [words[t.get_nns_by_vector(i, n=1)[0]] for i in res][0]
        if res != last:
            output.append(res)
        last = res
    if b != last:
        output.append(b)
    last = b
print(textwrap.fill(", ".join(output), 65))

def analogy(t, w1, w2, w3):
    vec = (np.array(t.get_item_vector(w2)) - np.array(t.get_item_vector(w1))) + np.array(t.get_item_vector(w3))
    #return t.get_nns_by_vector(vec, 10)
    return nnslookup(t, nlp, words, vec, 10)

good_groups = [
    ["decide", "decision", "explode"], # explosion
    ["final", "finalize", "modern"],
    ["glory", "glorify", "liquid"],
    ["bite", "bitten", "shake"], # shaken
    ["leaf", "leaves", "calf"], # calves
    ["foot", "feet", "tooth"], # teeth
    ["automaton", "automata", "criterion"], # criteria
    ["four", "fourteen", "nine"], # nineteen
    ["light", "slide", "lack"], # slag
    ["whisky", "whimsy", "frisky"], # flimsy
    ["could", "stood", "calling"], # stalling
]
for w1, w2, w3 in good_groups:
    # uncomment for latex table formatting
    #print("%s & %s & %s & %s \\\\" % (w1, w2, w3, words[analogy(t, lookup[w1], lookup[w2], lookup[w3])[0]]))
    print("%s : %s :: %s : %s" % (w1, w2, w3, words[analogy(t, lookup[w1], lookup[w2], lookup[w3])[0]]))

def vecsum(t, w1, w2, mult=1.0):
    vec = (np.array(t.get_item_vector(w1)) + np.array(t.get_item_vector(w2))*mult)
    return nnslookup(t, nlp, words, vec)
def vecsub(t, w1, w2):
    vec = (np.array(t.get_item_vector(w1)) - np.array(t.get_item_vector(w2)))
    return nnslookup(t, nlp, words, vec)

[words[i] for i in vecsum(t, lookup["ate"], lookup["teen"])]

[words[i] for i in vecsum(t, lookup["miss"], lookup["sieve"])]

[words[i] for i in vecsum(t, lookup["sub"], lookup["marine"])]

[words[i] for i in vecsum(t, lookup["fizz"], lookup["theology"])]

[words[i] for i in vecsum(t, lookup["snack"], lookup["king"])]

[words[i] for i in vecsub(t, lookup["submarine"], lookup["sub"])]

[words[i] for i in vecsub(t, lookup["wordsworth"], lookup["word"])]

[words[i] for i in vecsub(t, lookup["lavender"], lookup["under"])]

[words[i] for i in vecsub(t, lookup["curiously"], lookup["lee"])]

[words[i] for i in vecsub(t, lookup["ingredients"], lookup["reed"])]

[words[i] for i in vecsub(t, lookup["disassociate"], lookup["diss"])]

text = """two roads diverged in a yellow wood
and sorry i could not travel both
and be one traveler long i stood
and looked down one as far as i could
to where it bent in the undergrowth
 
then took the other as just as fair
and having perhaps the better claim
because it was grassy and wanted wear
though as for that the passing there
had worn them really about the same
 
and both that morning equally lay
in leaves no step had trodden black
oh i kept the first for another day
yet knowing how way leads on to way
i doubted if i should ever come back
 
i shall be telling this with a sigh
somewhere ages and ages hence
two roads diverged in a wood and i
i took the one less traveled by
and that has made all the difference"""

for line in text.split("\n"):
    print(' '.join([words[vecsum(t, lookup[word], lookup["kiki"], 0.8)[0]] for word in line.split()]).capitalize())

for line in text.split("\n"):
    print(' '.join([words[vecsum(t, lookup[word], lookup["babu"], 0.8)[0]] for word in line.split()]).capitalize())

for line in text.split("\n"):
    print(' '.join([words[vecsum(t, lookup[word], lookup["road"], 0.95)[0]] for word in line.split()]).capitalize())

